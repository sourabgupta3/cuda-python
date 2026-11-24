# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare complete batch reads with direct-binding diagnostic lower bounds."""

import argparse
import ctypes as ct
import hashlib
import importlib.metadata
import json
import platform
import random
import statistics
import tempfile
import time
from pathlib import Path

from investigate import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="/mnt/nvme0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacities", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if min(args.rounds, args.count, args.size, *args.capacities) < 1:
        parser.error("capacities, size, rounds, and count must be positive")
    before = audit()
    if before["foreign_gpu_processes"] or before["io_processes"]:
        raise RuntimeError(f"Benchmark requires idle GPU/storage: {before}")

    import cupy as cp
    import numpy as np

    import cufile_python as api
    from cuda.bindings import cufile
    from cufile_python import cufile_driver

    root = Path(__file__).resolve().parents[1]
    report = {
        "arguments": vars(args) | {"output": str(args.output)},
        "method": "Submit plus blocking collection, reused queues, shuffled case order, warm file/device caches",
        "limitations": "Direct-binding probes omit ownership, buffer validation and public completion objects; not KvikIO",
        "python": platform.python_version(),
        "cuda_bindings": importlib.metadata.version("cuda-bindings"),
        "libcufile": cufile_driver.libcufile_version(),
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "source_hashes": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.joinpath("cufile_python").rglob("*.py")
        },
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "audits": [before],
        "rows": [],
    }
    rng = random.Random(571)
    with tempfile.TemporaryDirectory(dir=args.directory, prefix="cufile-batch-bench-") as directory:

        def run_capacity(capacity):
            size = args.size
            expected = np.arange(capacity * size, dtype=np.uint8)
            path = Path(directory) / str(capacity)
            path.write_bytes(expected.tobytes())
            target = cp.empty_like(cp.asarray(expected))
            cp.cuda.get_current_stream().synchronize()
            with api.CuFile(path) as file, api.Batch(capacity) as batch:
                operations = [
                    api.BatchOp(file, target[i * size : (i + 1) * size], "read", file_offset=i * size)
                    for i in range(capacity)
                ]

                def packed_params():
                    data = np.zeros(capacity, dtype=cufile.io_params_dtype)
                    for i in range(capacity):
                        item = data[i]
                        item["mode"] = cufile.BatchMode.BATCH
                        item["fh"] = file._handle._handle
                        item["opcode"] = cufile.Opcode.READ
                        item["cookie"] = i + 1
                        transfer = item["u"]["batch"]
                        transfer["dev_ptr_base"] = target.data.ptr + i * size
                        transfer["dev_ptr_offset"] = 0
                        transfer["file_offset"] = i * size
                        transfer["size_"] = size
                    return cufile.IOParams.from_data(data)

                params = packed_params()
                event_data = np.empty(capacity, dtype=cufile.io_events_dtype)
                events = cufile.IOEvents.from_data(event_data)
                count = ct.c_uint(capacity)
                handle = cufile.batch_io_set_up(capacity)
                try:

                    def direct(repack=False):
                        submitted = packed_params() if repack else params
                        cufile.batch_io_submit(handle, capacity, submitted.ptr, 0)
                        count.value = capacity
                        cufile.batch_io_get_status(handle, capacity, ct.addressof(count), events.ptr, 0)
                        assert count.value == capacity
                        total = 0
                        for i in range(capacity):
                            event = event_data[i]
                            assert event["status"] == cufile.Status.COMPLETE
                            assert event["ret"] == size
                            total += int(event["ret"])
                        return total

                    def complete():
                        batch.submit(operations)
                        return sum(result.get() for result in batch.wait_all())

                    cases = {"python": complete, "bindings_reused": direct, "bindings_repacked": lambda: direct(True)}
                    for iteration in range(args.rounds):
                        names = list(cases)
                        rng.shuffle(names)
                        for name in names:
                            call = cases[name]
                            target.fill(0)
                            cp.cuda.get_current_stream().synchronize()
                            for _ in range(5):
                                assert call() == capacity * size
                            samples = []
                            for _ in range(args.count):
                                start = time.perf_counter_ns()
                                value = call()
                                samples.append((time.perf_counter_ns() - start) / 1000)
                                assert value == capacity * size
                            np.testing.assert_array_equal(cp.asnumpy(target), expected)
                            report["rows"].append(
                                dict(capacity=capacity, case=name, round=iteration, samples_us=samples)
                            )
                finally:
                    cufile.batch_io_destroy(handle)

        for capacity in args.capacities:
            run_capacity(capacity)
    report["audits"].append(audit())
    if any(item["foreign_gpu_processes"] or item["io_processes"] for item in report["audits"]):
        raise RuntimeError("A competing workload invalidated this measurement")
    report["summary"] = []
    for capacity in args.capacities:
        for name in ("python", "bindings_reused", "bindings_repacked"):
            medians = [
                statistics.median(row["samples_us"])
                for row in report["rows"]
                if row["capacity"] == capacity and row["case"] == name
            ]
            report["summary"].append(
                dict(capacity=capacity, case=name, median_us=statistics.median(medians), round_medians_us=medians)
            )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

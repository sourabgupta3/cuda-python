# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controlled native-cuFile latency comparisons and diagnostic substitutions.

Probes run on an idle default stream in a known, current CUDA context.
Raw/stream I/O uses that caller context; cached worker metadata and skipped
retention are diagnostic substitutions, not public API changes.
All transfers use cuda.bindings; the reference package uses its existing backend.
"""

import argparse
import ctypes as ct
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import queue
import random
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import patch


def audit():
    smi, ps = shutil.which("nvidia-smi"), shutil.which("ps")
    if smi is None or ps is None:
        raise RuntimeError("nvidia-smi and ps are required for the benchmark audit")
    gpu_processes = subprocess.check_output(
        [smi, "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"], text=True
    ).strip()
    return {
        "time_ns": time.time_ns(),
        "gpu_processes": gpu_processes,
        "foreign_gpu_processes": [
            line for line in gpu_processes.splitlines() if line.split(",", 1)[0].strip() != str(os.getpid())
        ],
        "io_processes": [
            line.strip()
            for line in subprocess.check_output([ps, "-eo", "pid=,comm="], text=True).splitlines()
            if line.split()[-1] in {"fio", "gdsio", "gdsio_verify"}
        ],
    }


def measure(call, count, size):
    for _ in range(20):
        assert call() == size
    samples = []
    for _ in range(count):
        start = time.perf_counter_ns()
        value = call()
        samples.append((time.perf_counter_ns() - start) / 1000)
        assert value == size
    return {"median_us": statistics.median(samples), "samples_us": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="/mnt/nvme0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation", choices=("python", "reference", "paired"), default="paired")
    parser.add_argument("--probes", action="store_true")
    parser.add_argument("--trace", action="store_true", help="Record separate instrumented binding-call spans")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.rounds, args.count, args.size, args.threads) < 1:
        parser.error("rounds, count, size, and threads must be positive")
    if args.probes and args.implementation != "paired":
        parser.error("--probes requires paired implementations")

    before = audit()
    if before["foreign_gpu_processes"] or before["io_processes"]:
        raise RuntimeError(f"Benchmark requires an idle GPU/storage workload: {before}")
    import cupy as cp
    import numpy as np

    from cuda.bindings import cufile, driver

    packages = {}
    if args.implementation != "reference":
        packages["python"] = importlib.import_module("cufile_python")
    if args.implementation != "python":
        packages["reference"] = importlib.import_module("kvikio")
    for module in packages.values():
        importlib.import_module(module.__name__ + ".defaults")
    report = {
        "process_id": os.getpid(),
        "arguments": vars(args) | {"output": str(args.output)},
        "method": "Native GPU I/O; same file/allocation/stream; rotated deterministic random block order; no profiler",
        "audits": [before],
        "python": platform.python_version(),
        "cuda_bindings": importlib.metadata.version("cuda-bindings"),
        "libcufile": importlib.import_module(
            next(iter(packages.values())).__name__ + ".cufile_driver"
        ).libcufile_version(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "packages": {name: {"path": mod.__file__, "version": mod.__version__} for name, mod in packages.items()},
        "source_hashes": {
            str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).resolve().parents[1].joinpath("cufile_python").rglob("*.py")
        },
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": [],
    }
    with (
        tempfile.TemporaryDirectory(dir=args.directory, prefix="cufile-attribution-") as directory,
        ExitStack() as stack,
    ):
        path = Path(directory) / "data"
        expected = np.arange(args.size, dtype=np.uint8)
        path.write_bytes(expected.tobytes())
        target = cp.asarray(expected)
        stream = cp.cuda.Stream()
        cp.cuda.get_current_stream().synchronize()
        files, cases = {}, {}
        for name, module in packages.items():
            stack.enter_context(
                module.defaults.set(
                    {
                        "compat_mode": module.CompatMode.OFF,
                        "gds_threshold": 0,
                        "num_threads": args.threads,
                        "task_size": 4 * 1024**2,
                    }
                )
            )
            file = files[name] = stack.enter_context(module.CuFile(path, "r+"))
            cases.update(
                {
                    name + ".read": (partial(file.read, target), ()),
                    name + ".write": (partial(file.write, target), ()),
                    name + ".raw": (partial(file.raw_read, target), ()),
                    name + ".async": (lambda file=file: file.raw_read_async(target, stream.ptr).check_bytes_done(), ()),
                }
            )
        if args.probes:
            from cufile_python._lib import _runtime, arr, buffer, file_handle, future

            file = files["python"]
            handle, pointer = file._handle._handle, target.data.ptr
            view, context = arr.Array(target), _runtime.Context(arr.Array(target))

            class CachedContext(_runtime.Context):
                def __init__(self, _array):
                    self.context = context.context

            def direct_read():
                _runtime.cuda(driver.cuStreamSynchronize(0))
                return cufile.read(handle, pointer, args.size, 0, 0)

            def worker_read():
                with context:
                    return cufile.read(handle, pointer, args.size, 0, 0)

            def minimal_worker():
                _runtime.cuda(driver.cuStreamSynchronize(0))
                return _runtime.pool().submit(worker_read).result()

            requests, done_event, outcome = queue.SimpleQueue(), threading.Event(), []

            def slot_worker():
                while requests.get() is not None:
                    try:
                        outcome.append(worker_read())
                    except BaseException as error:
                        outcome.append(error)
                    finally:
                        done_event.set()

            slot_thread = threading.Thread(target=slot_worker, name="diagnostic-single-slot")
            slot_thread.start()

            def stop_slot():
                requests.put(None)
                slot_thread.join()

            stack.callback(stop_slot)

            def single_slot():
                _runtime.cuda(driver.cuStreamSynchronize(0))
                done_event.clear()
                requests.put(True)
                done_event.wait()
                value = outcome.pop()
                if isinstance(value, BaseException):
                    raise value
                return value

            class Storage(ct.Structure):
                _fields_ = [
                    ("size", ct.c_size_t),
                    ("offset", ct.c_int64),
                    ("dev_offset", ct.c_int64),
                    ("done", ct.c_ssize_t),
                ]

            reused = Storage(args.size, 0, 0, -1)

            def packed_stream(reuse=False):
                storage = reused if reuse else Storage(args.size, 0, 0, -1)
                base = ct.addressof(storage)
                cufile.read_async(
                    handle,
                    pointer,
                    base + Storage.size.offset,
                    base + Storage.offset.offset,
                    base + Storage.dev_offset.offset,
                    base + Storage.done.offset,
                    stream.ptr,
                )
                _runtime.cuda(driver.cuStreamSynchronize(stream.ptr))
                return storage.done

            def scalar_stream():
                size, offset, dev_offset, done = ct.c_size_t(args.size), ct.c_int64(0), ct.c_int64(0), ct.c_ssize_t(-1)
                cufile.read_async(
                    handle,
                    pointer,
                    ct.addressof(size),
                    ct.addressof(offset),
                    ct.addressof(dev_offset),
                    ct.addressof(done),
                    stream.ptr,
                )
                _runtime.cuda(driver.cuStreamSynchronize(stream.ptr))
                return done.value

            cases.update(
                {
                    "bindings.raw": (direct_read, ()),
                    "bindings.worker": (minimal_worker, ()),
                    "probe.worker.single_slot": (single_slot, ()),
                    "bindings.async_scalars": (scalar_stream, ()),
                    "bindings.async_packed": (packed_stream, ()),
                    "bindings.async_reused": (partial(packed_stream, True), ()),
                }
            )
            for method, call in (
                ("raw", partial(file.raw_read, target)),
                ("async", cases["python.async"][0]),
                ("read", cases["python.read"][0]),
            ):
                prepared = (
                    partial(file.raw_read, view)
                    if method == "raw"
                    else partial(file.read, view)
                    if method == "read"
                    else lambda: file.raw_read_async(view, stream.ptr).check_bytes_done()
                )
                cases[f"probe.{method}.prepared_view"] = (prepared, ())
                if method == "read":
                    cases["probe.read.cached_context"] = (call, ((file_handle, "Context", CachedContext),))
                    cases["probe.read.prepared_and_cached_context"] = (
                        prepared,
                        ((file_handle, "Context", CachedContext),),
                    )
            cases["probe.raw.pointer_lookup"] = (
                cases["python.raw"][0],
                ((buffer, "registered_pointer", lambda ptr: (ptr, 0)),),
            )
            cases["probe.async.global_retention"] = (
                cases["python.async"][0],
                (
                    (file_handle, "retain_stream", lambda _work: None),
                    (future, "release_stream", lambda _work: None),
                ),
            )
            cases["probe.raw.no_frontend"] = (partial(file._handle.read, target, None, 0, 0), ())
            cases["probe.async.no_frontend"] = (
                lambda: file._handle.read_async(target, None, 0, 0, stream.ptr).check_bytes_done(),
                (),
            )

        rng = random.Random(716)
        for round_index in range(args.rounds):
            order = list(cases)
            rng.shuffle(order)
            for name in order:
                call, patches = cases[name]
                comparison = None
                if name.startswith(("probe.", "bindings.")):
                    api = name.split(".")[1].split("_")[0]
                    api = "read" if api == "worker" else api
                    comparison = cases["python." + api][0]
                    before_case = measure(comparison, max(50, args.count // 2), args.size)
                with ExitStack() as scope:
                    for obj, attr, value in patches:
                        scope.enter_context(patch.object(obj, attr, value))
                    target.fill(0 if not name.endswith(".write") else 1)
                    if name.endswith(".write"):
                        target[:] = cp.asarray(expected)
                    cp.cuda.get_current_stream().synchronize()
                    result = measure(call, args.count, args.size)
                    np.testing.assert_array_equal(cp.asnumpy(target), expected)
                report["rows"].append({"round": round_index + 1, "case": name, **result})
                if comparison:
                    after_case = measure(comparison, max(50, args.count // 2), args.size)
                    report["rows"][-1].update(
                        control_before_us=before_case["median_us"],
                        control_after_us=after_case["median_us"],
                        paired_saving_us=(before_case["median_us"] + after_case["median_us"]) / 2 - result["median_us"],
                    )
            print(f"Completed round {round_index + 1}/{args.rounds}", flush=True)
            report["audits"].append(audit())
            if report["audits"][-1]["foreign_gpu_processes"] or report["audits"][-1]["io_processes"]:
                break
        os.fdatasync(next(iter(files.values())).fileno())
        assert path.read_bytes() == expected.tobytes()
        report["audits"].append(audit())
        if args.trace:
            report["trace_note"] = (
                "Instrumented wall spans include bindings and GIL reacquisition; not native device execution time."
            )
            report["trace"] = {}
            for name in (
                "python.raw",
                "python.read",
                "python.async",
                "bindings.raw",
                "bindings.worker",
                "bindings.async_scalars",
            ):
                if name not in cases:
                    continue
                events, rows = [], []

                def timed(original, label, events=events):
                    def run(*values):
                        begin = time.perf_counter_ns()
                        value = original(*values)
                        events.append((label, begin, time.perf_counter_ns()))
                        return value

                    return run

                with ExitStack() as scope:
                    for module, attr in ((cufile, "read"), (cufile, "read_async"), (driver, "cuStreamSynchronize")):
                        scope.enter_context(patch.object(module, attr, timed(getattr(module, attr), attr)))
                    for _ in range(200):
                        events.clear()
                        begin = time.perf_counter_ns()
                        assert cases[name][0]() == args.size
                        end = time.perf_counter_ns()
                        native = next(event for event in events if event[0] in {"read", "read_async"})
                        rows.append(
                            {
                                "before_io_us": (native[1] - begin) / 1000,
                                "io_call_us": (native[2] - native[1]) / 1000,
                                "after_io_us": (end - native[2]) / 1000,
                                "total_us": (end - begin) / 1000,
                                "sync_calls": [
                                    (event[2] - event[1]) / 1000
                                    for event in events
                                    if event[0] == "cuStreamSynchronize"
                                ],
                            }
                        )
                report["trace"][name] = rows
        report["libraries"] = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if any(name in line for name in ("libcufile.so", "libkvikio.so"))
            }
        )
    report["valid_comparison"] = all(
        not item["foreign_gpu_processes"] and not item["io_processes"] for item in report["audits"]
    )
    report["summary"] = {
        name: {
            "median_us": statistics.median(row["median_us"] for row in report["rows"] if row["case"] == name),
            "round_medians_us": [row["median_us"] for row in report["rows"] if row["case"] == name],
            "paired_savings_us": [
                row["paired_saving_us"] for row in report["rows"] if row["case"] == name and "paired_saving_us" in row
            ],
        }
        for name in cases
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["valid_comparison"]:
        raise RuntimeError(f"Another workload was detected; exclude this run: {args.output}")
    print(json.dumps({name: value["median_us"] for name, value in report["summary"].items()}, indent=2))


if __name__ == "__main__":
    main()

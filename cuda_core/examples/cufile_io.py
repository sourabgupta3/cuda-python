# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verified native synchronous and CUDA-stream file I/O using core resources."""

import argparse
import hashlib
import json
import pathlib
import tempfile
import time

from cuda.core import Device, VirtualMemoryResource, VirtualMemoryResourceOptions, cufile


def run_application(directory: pathlib.Path, size: int, iterations: int) -> dict:
    if size < 256 or size % 256 or iterations < 1:
        raise ValueError("size must be a positive multiple of 256 and iterations must be positive")
    expected = bytes(range(256)) * (size // 256)
    device = Device(0)
    device.set_current()
    stream = device.create_stream()
    memory = VirtualMemoryResource(device, VirtualMemoryResourceOptions(gpu_direct_rdma=True))
    results = {}
    try:
        with tempfile.TemporaryDirectory(prefix="cufile-native-", dir=directory) as temporary:
            root = pathlib.Path(temporary)
            source = root / "input.bin"
            source.write_bytes(expected)
            with (
                cufile.Driver(),
                memory.allocate(size, stream=stream) as allocation,
                cufile.RegisteredBuffer(allocation) as buffer,
                cufile.RegisteredStream(stream) as io_stream,
                cufile.File(source) as reader,
                cufile.IOParams(buffer, size) as read_params,
                cufile.IOParams(buffer, size) as write_params,
            ):
                for mode in ("sync", "stream"):
                    elapsed = 0.0
                    output = root / f"{mode}.bin"
                    with cufile.File(output, "w+") as writer:
                        for iteration in range(-1, iterations):
                            begin = time.perf_counter()
                            if mode == "sync":
                                received = reader.read(read_params)
                                written = writer.write(write_params)
                            else:
                                # The supplied stream orders the read before the write.
                                reader.read_async(read_params, io_stream)
                                writer.write_async(write_params, io_stream)
                                stream.sync()
                                received = read_params.bytes_done
                                written = write_params.bytes_done
                            duration = time.perf_counter() - begin
                            assert received == written == size
                            if iteration >= 0:
                                elapsed += duration
                            assert output.read_bytes() == expected
                    results[mode] = {
                        "verified": True,
                        "bytes_per_transfer": size,
                        "iterations": iterations,
                        "seconds": elapsed,
                        "MiB_per_second": (size * 2 * iterations / 1024**2) / elapsed,
                        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    }
    finally:
        stream.sync()
        stream.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=pathlib.Path, default=pathlib.Path("/mnt/nvme0"))
    parser.add_argument("--size-mib", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if min(args.size_mib, args.iterations) < 1:
        parser.error("size and iterations must be positive")
    result = run_application(args.directory, args.size_mib * 1024**2, args.iterations)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native multi-file cuFile batches using cuda.core resource ownership."""

import argparse
import json
import pathlib
import tempfile

from cuda.core import Device, VirtualMemoryResource, VirtualMemoryResourceOptions, cufile


def run_application(directory: pathlib.Path) -> dict:
    device = Device(0)
    device.set_current()
    stream = device.create_stream()
    resource = VirtualMemoryResource(device, VirtualMemoryResourceOptions(gpu_direct_rdma=True))
    block = 4096
    expected = [bytes(range(256)) * 16, bytes(reversed(range(256))) * 16]
    try:
        with tempfile.TemporaryDirectory(prefix="cufile-core-batch-", dir=directory) as temporary:
            paths = [pathlib.Path(temporary) / f"input-{index}.bin" for index in range(2)]
            outputs = [pathlib.Path(temporary) / f"output-{index}.bin" for index in range(2)]
            for path, content in zip(paths, expected):
                path.write_bytes(content)
            with (
                resource.allocate(block * 2, stream=stream) as allocation,
                cufile.RegisteredBuffer(allocation) as registered,
                cufile.File(paths[0]) as first,
                cufile.File(paths[1]) as second,
                cufile.File(outputs[0], "w") as out_first,
                cufile.File(outputs[1], "w") as out_second,
                cufile.Batch(2) as batch,
            ):
                batch.submit(
                    [
                        cufile.BatchIOParams(first, registered, "read", block, cookie="first"),
                        cufile.BatchIOParams(second, registered, "read", block, buffer_offset=block, cookie="second"),
                    ]
                )
                reads = batch.wait_all(timeout=10)
                batch.submit(
                    [
                        cufile.BatchIOParams(out_first, registered, "write", block, cookie="first"),
                        cufile.BatchIOParams(
                            out_second, registered, "write", block, buffer_offset=block, cookie="second"
                        ),
                    ]
                )
                writes = batch.wait_all(timeout=10)
            assert [path.read_bytes() for path in outputs] == expected
            return {
                "verified": True,
                "files": 2,
                "bytes_read": sum(item.result() for item in reads),
                "bytes_written": sum(item.result() for item in writes),
            }
    finally:
        stream.sync()
        stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=pathlib.Path, default=pathlib.Path("/mnt/nvme0"))
    args = parser.parse_args()
    print(json.dumps(run_application(args.directory), indent=2))


if __name__ == "__main__":
    main()

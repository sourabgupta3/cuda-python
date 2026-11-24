# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native batch I/O across files, with host/GPU buffers and queue reuse."""

import argparse
import tempfile
from dataclasses import replace
from pathlib import Path

import cupy as cp
import numpy as np

import cufile_python as cufile


def main(directory):
    with tempfile.TemporaryDirectory(dir=directory) as root:
        root = Path(root)
        size = 4096
        expected = np.arange(size, dtype=np.uint8)
        (root / "input.bin").write_bytes(expected.tobytes() * 2)
        host = bytearray(size)
        gpu = cp.empty(size, dtype=cp.uint8)
        cp.cuda.get_current_stream().synchronize()
        with (
            cufile.CuFile(root / "input.bin") as source,
            cufile.CuFile(root / "output.bin", "w+") as target,
            cufile.Batch(2) as batch,
        ):
            operations = [
                cufile.BatchOp(source, host, "read", cookie="host"),
                cufile.BatchOp(source, gpu, "read", file_offset=size, cookie="gpu"),
            ]
            batch.submit(operations)
            # libcufile 1.19 host batches require blocking collection.
            results = batch.wait_all()
            assert {item.cookie for item in results} == {"host", "gpu"}
            assert all(item.get() == size for item in results)
            assert host == expected.tobytes()
            np.testing.assert_array_equal(cp.asnumpy(gpu), expected)

            # GPU-only batches can also collect available results without waiting.
            batch.submit([operations[1]])
            available = batch.poll(max_complete=1)
            remaining = batch.wait_all(timeout=10)
            results = available + remaining  # Each completion appears once.
            assert len(results) == 1
            assert results[0].get() == size

            # Reuse the queue with changed entries after collecting the old batch.
            operations[0] = replace(operations[0], file=target, opcode="write")
            operations[1] = replace(operations[1], file=target, opcode="write")
            batch.submit(operations)
            assert [item.get() for item in batch.wait_all()] == [size, size]
        assert (root / "output.bin").read_bytes() == expected.tobytes() * 2
        assert batch.closed

        # Explicit close is also supported; complete the batch before the file.
        file = cufile.CuFile(root / "output.bin")
        batch = cufile.Batch(1)
        try:
            batch.submit([cufile.BatchOp(file, host, "read")])
            assert batch.wait_all()[0].get() == size
        finally:
            try:
                batch.close()
            finally:
                file.close()
    print("Batch read/write, mixed buffers, polling, reuse, and explicit close passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="/mnt/nvme0")
    main(parser.parse_args().directory)

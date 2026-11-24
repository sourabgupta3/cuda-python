# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refill a cuFile batch as individual reads complete.

Requires working native partial resubmission. The wrapper rejects rolling
submission on libcufile 1.19 because of a native memory-corruption defect.
"""

import argparse
import json
import pathlib
import tempfile
from dataclasses import replace

from cuda.core import cufile


def run_application(directory: pathlib.Path) -> dict:
    block, count = 4096, 12
    expected = [bytes([index]) * block for index in range(count)]
    with tempfile.TemporaryDirectory(prefix="cufile-rolling-", dir=directory) as temporary:
        paths = [pathlib.Path(temporary) / name for name in ("a.bin", "b.bin")]
        for path in paths:
            path.write_bytes(b"".join(expected))
        buffers = [bytearray(block), bytearray(block)]
        with cufile.File(paths[0]) as first, cufile.File(paths[1]) as second, cufile.Batch(2) as batch:
            entries = [
                cufile.BatchIOParams(file, buffer, "read", block, file_offset=slot * block, cookie=(slot, slot))
                for slot, (file, buffer) in enumerate(zip((first, second), buffers))
            ]
            batch.submit(entries)
            next_job, completed, refills = 2, 0, 0
            while batch.is_active:
                available = batch.poll(min_complete=1, max_complete=1, timeout=10)
                if not available:
                    raise TimeoutError("No batch completion within 10 seconds")
                for completion in available:
                    assert completion.result() == block
                    slot, job = completion.cookie
                    assert buffers[slot] == expected[job]
                    completed += 1
                    if next_job < count:
                        # Only this completed slot is changed; the other request stays outstanding.
                        entries[slot] = replace(entries[slot], file_offset=next_job * block, cookie=(slot, next_job))
                        refills += int(batch.is_active)
                        batch.submit([entries[slot]])
                        next_job += 1
            assert batch.wait_all() == ()
    assert completed == count
    return {"verified": True, "transfers": completed, "bytes_read": completed * block, "rolling_refills": refills}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=pathlib.Path, default=pathlib.Path("/mnt/nvme0"))
    print(json.dumps(run_application(parser.parse_args().directory), indent=2))

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KvikIO-style local I/O using the cufile_python import alias."""

import argparse
import tempfile
from pathlib import Path

import cupy as cp
import numpy as np

import cufile_python as kvikio
import cufile_python.defaults
from cufile_python import buffer, cufile_driver


def main(directory):
    with tempfile.TemporaryDirectory(dir=directory) as root:
        root = Path(root)
        path = root / "data.bin"
        source = cp.arange(1024**2, dtype=cp.uint8)
        target = cp.empty_like(source)
        expected = cp.asnumpy(source)
        with kvikio.SummaryMonitor() as monitor, kvikio.defaults.set("num_threads", 4):
            # Blocking and worker-pool operations use the same file object.
            with kvikio.CuFile(path, "w+") as file:
                assert file.write(source) == source.nbytes
                future = file.pread(target)
                assert future.get() == target.nbytes
                host = np.empty_like(expected)
                assert file.read(host) == host.nbytes
                np.testing.assert_array_equal(host, expected)
                assert file.pwrite(target).get() == target.nbytes
            cp.testing.assert_array_equal(source, target)

            # Explicit driver and stream-registration ownership.
            cufile_driver.driver_open()
            stream = cp.cuda.Stream()
            try:
                buffer.memory_register(source)
                try:
                    kvikio.stream_register(stream.ptr, 0)
                    try:
                        file = kvikio.CuFile(path, "r+")
                        try:
                            completion = file.raw_write_async(source, stream.ptr)
                            assert completion.check_bytes_done() == source.nbytes
                        finally:
                            file.close()
                    finally:
                        kvikio.stream_deregister(stream.ptr)
                finally:
                    buffer.memory_deregister(source)
            finally:
                cufile_driver.driver_close()

            np.testing.assert_array_equal(np.fromfile(path, dtype=np.uint8), expected)
        print(kvikio.__file__)
        print(monitor.get().report())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="/mnt/nvme0")
    main(parser.parse_args().directory)

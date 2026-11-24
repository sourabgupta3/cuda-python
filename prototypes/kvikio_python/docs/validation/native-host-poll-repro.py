# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Flushed progress output helps diagnose the native polling race.
# ruff: noqa: T201
"""Direct-bindings reproducer for libcufile 1.19 host polling; run in a separate process.

This deliberately invokes the native race guarded by cufile_python.Batch.
It uses no cufile_python code. It is diagnostic evidence, not a passing test.
"""

import ctypes as ct
import os
import tempfile
import time
from pathlib import Path

import cupy as cp
import numpy as np

from cuda.bindings import cufile


class Timespec(ct.Structure):
    _fields_ = [("tv_sec", ct.c_long), ("tv_nsec", ct.c_long)]


block = 4096
expected = np.arange(block, dtype=np.uint8)
source = cp.ndarray(block * 3, dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(cp.cuda.Memory(block * 3), 0))
target = cp.ndarray(block * 3, dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(cp.cuda.Memory(block * 3), 0))
source.set(np.tile(expected, 3))
host = bytearray(b"!" * block * 3)
host_pointer = ct.addressof(ct.c_char.from_buffer(host))
cp.cuda.get_current_stream().synchronize()
print("Native version:", cufile.get_version(), flush=True)
cufile.driver_open()
with tempfile.TemporaryDirectory(dir="/mnt/nvme0") as root:
    descriptors, handles = [], []
    for name in ("a", "b"):
        fd = os.open(Path(root) / name, os.O_RDWR | os.O_CREAT | os.O_DIRECT, 0o600)
        desc = cufile.Descr()
        desc.type = cufile.FileHandleType.OPAQUE_FD
        desc.handle.fd = fd
        desc.fs_ops = 0
        descriptors.append(fd)
        handles.append(cufile.handle_register(desc.ptr))
    events = cufile.IOEvents(2)
    interval = Timespec(0, 0)
    failures = 0
    for iteration in range(100):
        batch = cufile.batch_io_set_up(2)
        for reading in (False, True):
            params = cufile.IOParams(2)
            for index in range(2):
                item = params[index]
                item.mode = cufile.BatchMode.BATCH
                item.fh = handles[index]
                item.opcode = cufile.Opcode.READ if reading else cufile.Opcode.WRITE
                item.cookie = index + 1
                gpu_pointer = target.data.ptr if reading else source.data.ptr
                cpu_pointer = host_pointer + block if reading else expected.ctypes.data
                item.u.batch.dev_ptr_base = gpu_pointer + block if index == 0 else cpu_pointer
                item.u.batch.dev_ptr_offset = 0
                item.u.batch.file_offset = block if index == 0 else 0
                item.u.batch.size_ = block
            cufile.batch_io_submit(batch, 2, params.ptr, 0)
            pending = 2
            first = True
            while pending:
                count = ct.c_uint(1 if first else pending)
                first = False
                cufile.batch_io_get_status(batch, 0, ct.addressof(count), events.ptr, ct.addressof(interval))
                for i in range(count.value):
                    event = events[i]
                    if event.status != cufile.Status.COMPLETE or event.ret != block:
                        failures += 1
                        print(
                            "Failure:",
                            iteration,
                            reading,
                            event.cookie,
                            event.status,
                            ct.c_ssize_t(event.ret).value,
                            flush=True,
                        )
                pending -= count.value
                if pending:
                    time.sleep(0.0005)
        cufile.batch_io_destroy(batch)
    for handle in handles:
        cufile.handle_deregister(handle)
    for fd in descriptors:
        os.close(fd)
cufile.driver_close()
print("Failures:", failures, flush=True)
raise SystemExit(bool(failures))

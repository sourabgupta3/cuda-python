# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import threading

from ._runtime import Context, cuda, driver, native
from .arr import asarray

_registered = {}
_lock = threading.RLock()


def memory_register(buf):
    arr = asarray(buf)
    if not arr.cuda:
        raise ValueError("Non-CUDA buffers not supported")
    with Context(arr):
        base, size = cuda(driver.cuMemGetAddressRange(arr.ptr))
        with _lock:
            native("buf_register", int(base), size, 0)
            _registered[int(base)] = (size, arr)


def memory_deregister(buf):
    arr = asarray(buf)
    with Context(arr):
        base, _ = cuda(driver.cuMemGetAddressRange(arr.ptr))
        with _lock:
            native("buf_deregister", int(base))
            _registered.pop(int(base), None)


def registered_pointer(ptr):
    with _lock:
        for base, (size, _) in _registered.items():
            if base <= ptr < base + size:
                return base, ptr - base
    return ptr, 0


def bounce_buffer_free():
    # This backend has no Python-owned POSIX staging buffers.
    return 0

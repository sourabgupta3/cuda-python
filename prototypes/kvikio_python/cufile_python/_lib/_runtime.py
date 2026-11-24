# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import atexit
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

from cuda.bindings import cufile, driver


def cuda(result):
    status, *values = result
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA error: {status.name} ({int(status)})")
    return values[0] if len(values) == 1 else tuple(values)


def native(name, *args):
    try:
        return getattr(cufile, name)(*args)
    except cufile.cuFileError as exc:
        raise RuntimeError(str(exc)) from exc


class Context:
    """Resolved context handle; push/pop state belongs to each calling thread."""

    def __init__(self, arr):
        self.context = None
        if arr.cuda and arr.nbytes:
            cuda(driver.cuInit(0))
            result = driver.cuPointerGetAttribute(driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_CONTEXT, arr.ptr)
            self.context = None if result[0] == driver.CUresult.CUDA_ERROR_INVALID_VALUE else cuda(result)
            if self.context is None or not int(self.context):
                self.context = cuda(driver.cuCtxGetCurrent())
                accessible = False
                if int(self.context):
                    value = driver.cuPointerGetAttribute(
                        driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_POINTER, arr.ptr
                    )
                    if value[0] != driver.CUresult.CUDA_ERROR_INVALID_VALUE:
                        accessible = int(cuda(value)) == arr.ptr
                if not accessible:
                    ordinal = cuda(
                        driver.cuPointerGetAttribute(
                            driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, arr.ptr
                        )
                    )
                    with _lock:
                        if ordinal not in _primary_contexts:
                            _primary_contexts[ordinal] = cuda(driver.cuDevicePrimaryCtxRetain(ordinal))
                        self.context = _primary_contexts[ordinal]

    def __enter__(self):
        if self.context:
            cuda(driver.cuCtxPushCurrent(self.context))
        return self

    def __exit__(self, kind, error, traceback):
        if self.context:
            try:
                cuda(driver.cuCtxPopCurrent())
            except RuntimeError as cleanup:
                if error is None:
                    raise
                error.add_note(f"Restoring CUDA context failed: {cleanup}")


_lock = threading.RLock()
_pool = None
_pending_streams = set()
_pending_batches = set()
_primary_contexts = {}
_device_pools = {}


def pool(device=None):
    global _pool
    from . import defaults

    size = defaults.thread_pool_nthreads()
    with _lock:
        if device is not None:
            if device not in _device_pools:
                _device_pools[device] = ThreadPoolExecutor(
                    max_workers=size, thread_name_prefix=f"cufile-python-device-{device}"
                )
            return _device_pools[device]
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=size, thread_name_prefix="cufile-python")
        return _pool


def reset_pool():
    global _pool
    with _lock:
        previous, _pool = _pool, None
        devices = list(_device_pools.values())
        _device_pools.clear()
    if previous is not None:
        previous.shutdown(wait=True)
    for executor in devices:
        executor.shutdown(wait=True)


def retain_stream(work):
    with _lock:
        _pending_streams.add(work)


def release_stream(work):
    with _lock:
        _pending_streams.discard(work)


def retain_batch(batch):
    with _lock:
        _pending_batches.add(batch)


def release_batch(batch):
    with _lock:
        _pending_batches.discard(batch)


def shutdown():
    reset_pool()
    for batch in tuple(_pending_batches):
        try:
            batch.close()
        except RuntimeError as exc:
            warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
    for work in tuple(_pending_streams):
        try:
            work.check_bytes_done()
        except RuntimeError as exc:
            warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
    for ordinal in _primary_contexts:
        cuda(driver.cuDevicePrimaryCtxRelease(ordinal))
    _primary_contexts.clear()


atexit.register(shutdown)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private ownership and context adapters for the optional Linux cuFile API."""

from cpython.buffer cimport Py_buffer, PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_WRITABLE
from cpython.ref cimport Py_INCREF, Py_DECREF
from libc.stdint cimport intptr_t
from libc.errno cimport errno
from posix.types cimport off_t
from cuda.bindings cimport cydriver, cycufile
from cuda.core._memory._buffer cimport Buffer, Buffer_check_open
from cuda.core._stream cimport Stream, Stream_accept, Stream_check_open
from cuda.core._event cimport Event
from cuda.core._rt cimport (
    ContextHandle, DevicePtrHandle, StreamHandle, EventHandle, as_intptr, as_cu,
    create_context_handle_ref, get_primary_context, get_stream_context,
    get_current_context, io_enter_context, io_exit_context, attach_rollback_failure,
    create_event_handle_for_stream, get_last_error,
)
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

import math
import operator
import os
import threading
import numpy as np
from cuda.core._memoryview import StridedMemoryView


def pin_forever(obj):
    """Retain uncertain in-flight ownership even during interpreter teardown."""
    Py_INCREF(obj)


def unpin(obj):
    """Release a pending-work pin only after quiescence has been established."""
    Py_DECREF(obj)


ctypedef ssize_t (*_ReadFn)(cycufile.CUfileHandle_t, void*, size_t, off_t, off_t) noexcept nogil
ctypedef ssize_t (*_WriteFn)(cycufile.CUfileHandle_t, const void*, size_t, off_t, off_t) noexcept nogil


cdef class NativeTransfers:
    """Immutable function table resolved by cuda.bindings, as in core's runtime.

    Direct invocation captures errno before any Python/GIL machinery can
    overwrite it. The generated Python binding otherwise interprets POSIX
    errno values as cuFile enum values, which can itself raise ValueError.
    """
    cdef _ReadFn read_fn
    cdef _WriteFn write_fn

    def __init__(self):
        from cuda.bindings._internal import cufile as loader
        from cuda.bindings._internal.utils import FunctionNotFoundError
        cdef intptr_t read_address = loader._inspect_function_pointer("__cuFileRead")
        cdef intptr_t write_address = loader._inspect_function_pointer("__cuFileWrite")
        if not read_address or not write_address:
            raise FunctionNotFoundError("libcufile must provide cuFileRead and cuFileWrite")
        self.read_fn = <_ReadFn>read_address
        self.write_fn = <_WriteFn>write_address

    def transfer(self, intptr_t handle, intptr_t ptr, size_t size, long long file_offset,
                 long long buffer_offset, bint write):
        cdef ssize_t result
        cdef int error_number
        with nogil:
            if write:
                result = self.write_fn(<cycufile.CUfileHandle_t>handle, <const void*>ptr,
                                      size, file_offset, buffer_offset)
            else:
                result = self.read_fn(<cycufile.CUfileHandle_t>handle, <void*>ptr,
                                     size, file_offset, buffer_offset)
            error_number = errno
        if result < 0:
            from cuda.core.cufile._runtime import CuFileError, check_count
            operation = "write" if write else "read"
            if result == -1:
                cause = OSError(error_number, os.strerror(error_number))
                error = CuFileError(
                    f"cuFile {operation} failed: {cause.strerror}", errno=error_number,
                    strerror=cause.strerror, operation=operation, native_result=result)
                raise error from cause
            check_count(result, operation=operation)
        return result


cdef class _ContextScope:
    cdef ContextHandle context
    cdef cydriver.CUcontext previous
    cdef int changed

    def __enter__(self):
        if self.context:
            with nogil:
                HANDLE_RETURN(io_enter_context(self.context, &self.previous, &self.changed))
        return self

    def __exit__(self, exc_type, exc, traceback):
        cdef cydriver.CUresult status
        if self.context:
            with nogil:
                status = io_exit_context(self.previous, self.changed, cydriver.CUDA_SUCCESS)
            if status != cydriver.CUDA_SUCCESS:
                if exc is None:
                    HANDLE_RETURN(status)
                else:
                    attach_rollback_failure(b"cuCtxSetCurrent", status, b"restoring the caller's context after cuFile")
        return False


cdef _ContextScope _scope(ContextHandle ctx):
    cdef _ContextScope result = _ContextScope()
    result.context = ctx
    return result


def _contiguous(shape, strides, itemsize):
    if strides is None or 0 in shape:
        return True
    for dimensions in (range(len(shape)), reversed(range(len(shape)))):
        expected = itemsize
        for axis in dimensions:
            if shape[axis] > 1 and strides[axis] != expected:
                break
            expected *= shape[axis]
        else:
            return True
    return False


cdef class BufferLease:
    cdef DevicePtrHandle pointer_owner
    cdef ContextHandle context_owner
    cdef object owner
    cdef Py_buffer view
    cdef bint has_view
    cdef readonly intptr_t ptr
    cdef readonly size_t nbytes
    cdef readonly bint is_device
    cdef readonly bint readonly

    def __init__(self, obj, *, writable=False):
        cdef Buffer buf
        cdef cydriver.CUcontext ctx = NULL
        cdef cydriver.CUdevice device
        cdef cydriver.CUmemorytype memory_type
        cdef cydriver.CUresult status
        cdef int flags = PyBUF_ANY_CONTIGUOUS
        if isinstance(obj, Buffer):
            buf = obj
            Buffer_check_open(buf)
            self.pointer_owner = buf._h_ptr
            self.owner = buf._owner
            self.ptr = as_intptr(self.pointer_owner)
            self.nbytes = buf._size
            self.is_device = buf.is_device_accessible and not buf.is_host_accessible
        elif hasattr(obj, "__cuda_array_interface__"):
            interface = obj.__cuda_array_interface__
            shape = tuple(operator.index(n) for n in interface["shape"])
            if any(n < 0 for n in shape):
                raise ValueError("Negative buffer dimensions")
            itemsize = np.dtype(interface["typestr"]).itemsize
            if not _contiguous(shape, interface.get("strides"), itemsize):
                raise ValueError("cuFile requires a C- or F-contiguous buffer")
            self.owner = StridedMemoryView.from_cuda_array_interface(obj, stream_ptr=-1)
            self.readonly = self.owner.readonly
            if writable and self.readonly:
                raise ValueError("Read destination must be writable")
            self.ptr = self.owner.ptr
            self.nbytes = math.prod(shape) * itemsize
            self.is_device = True
        else:
            if writable:
                flags |= PyBUF_WRITABLE
            try:
                PyObject_GetBuffer(obj, &self.view, flags)
            except BufferError as exc:
                raise ValueError("cuFile requires a contiguous writable destination or contiguous source") from exc
            self.has_view = True
            self.ptr = <intptr_t>self.view.buf
            self.nbytes = self.view.len
            self.readonly = self.view.readonly
        if self.is_device and self.nbytes:
            with nogil:
                HANDLE_RETURN(cydriver.cuPointerGetAttribute(&memory_type, cydriver.CU_POINTER_ATTRIBUTE_MEMORY_TYPE, self.ptr))
                HANDLE_RETURN(cydriver.cuPointerGetAttribute(&ctx, cydriver.CU_POINTER_ATTRIBUTE_CONTEXT, self.ptr))
                HANDLE_RETURN(cydriver.cuPointerGetAttribute(&device, cydriver.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, self.ptr))
            if ctx != NULL:
                self.context_owner = create_context_handle_ref(ctx)
            else:
                self.context_owner = get_primary_context(device)

    def __dealloc__(self):
        if self.has_view:
            PyBuffer_Release(&self.view)

    def context(self):
        return _scope(self.context_owner)

    @property
    def context_id(self):
        return as_intptr(self.context_owner) if self.context_owner else 0

cdef class StreamLease:
    cdef StreamHandle stream_owner
    cdef ContextHandle context_owner
    cdef readonly intptr_t ptr
    cdef object thread_id

    def __init__(self, obj, BufferLease buffer=None):
        cdef Stream stream
        cdef cydriver.CUcontext ctx = NULL
        if not isinstance(obj, Stream) and not hasattr(obj, "__cuda_stream__"):
            self.ptr = operator.index(obj)
            if self.ptr < 0:
                raise ValueError("Stream handle cannot be negative")
            self.context_owner = buffer.context_owner if buffer is not None and buffer.context_owner else get_current_context()
        else:
            stream = Stream_accept(obj, allow_stream_protocol=True)
            Stream_check_open(stream)
            self.stream_owner = stream._h_stream
            self.ptr = as_intptr(self.stream_owner)
            self.context_owner = get_stream_context(self.stream_owner)
            if not self.context_owner:
                self.context_owner = buffer.context_owner if buffer is not None and buffer.context_owner else get_current_context()
        if self.ptr == 2:
            self.thread_id = threading.get_ident()
        if not self.context_owner:
            raise RuntimeError("A CUDA context is required for cuFile stream operations")
        if self.ptr not in (0, 1, 2):
            with _scope(self.context_owner):
                with nogil:
                    HANDLE_RETURN(cydriver.cuStreamGetCtx(<cydriver.CUstream>self.ptr, &ctx))
            if buffer is not None and buffer.context_owner and ctx != as_cu(buffer.context_owner):
                raise ValueError("Stream and buffer belong to different CUDA contexts")
            self.context_owner = create_context_handle_ref(ctx)

    def context(self):
        return _scope(self.context_owner)

    @property
    def context_id(self):
        return as_intptr(self.context_owner) if self.context_owner else 0

    def sync(self):
        if self.thread_id is not None and self.thread_id != threading.get_ident():
            raise RuntimeError("A per-thread default stream must be completed on its submitting thread")
        with self.context():
            with nogil:
                HANDLE_RETURN(cydriver.cuStreamSynchronize(<cydriver.CUstream>self.ptr))

    def record_event(self):
        cdef EventHandle handle
        with self.context():
            handle = create_event_handle_for_stream(
                <cydriver.CUstream>self.ptr, cydriver.CUevent_flags.CU_EVENT_DISABLE_TIMING)
            if not handle:
                HANDLE_RETURN(get_last_error())
                raise RuntimeError("internal cuda.core error, please report: empty cuFile completion event")
            event = Event._from_handle(handle)
            with nogil:
                HANDLE_RETURN(cydriver.cuEventRecord(as_cu(handle), <cydriver.CUstream>self.ptr))
        return event

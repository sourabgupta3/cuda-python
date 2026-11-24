# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained Python buffer and CUDA Array Interface views; no compiled adapter."""

import ctypes as ct
import math
import operator
import sys


class _PyBuffer(ct.Structure):
    _fields_ = [
        ("buf", ct.c_void_p),
        ("obj", ct.c_void_p),
        ("len", ct.c_ssize_t),
        ("itemsize", ct.c_ssize_t),
        ("readonly", ct.c_int),
        ("ndim", ct.c_int),
        ("format", ct.c_char_p),
        ("shape", ct.POINTER(ct.c_ssize_t)),
        ("strides", ct.POINTER(ct.c_ssize_t)),
        ("suboffsets", ct.POINTER(ct.c_ssize_t)),
        ("internal", ct.c_void_p),
    ]


_get = ct.pythonapi.PyObject_GetBuffer
_get.argtypes = [ct.py_object, ct.POINTER(_PyBuffer), ct.c_int]
_get.restype = ct.c_int
_release = ct.pythonapi.PyBuffer_Release
_release.argtypes = [ct.POINTER(_PyBuffer)]
_release.restype = None


class Array:
    def __init__(self, obj):
        self.obj, self._view = obj, None
        iface = getattr(obj, "__cuda_array_interface__", None)
        self.cuda = iface is not None
        if self.cuda:
            if iface.get("mask") is not None:
                raise NotImplementedError("mask attribute not supported")
            self.ptr, self.readonly = iface["data"]
            typestr = iface["typestr"]
            if not typestr:
                raise ValueError("Expected nonempty typestr")
            # NumPy is optional except for structured or uncommon dtype descriptions.
            try:
                self.itemsize = int(typestr[2:])
            except ValueError:
                import numpy as np

                self.itemsize = np.dtype(typestr).itemsize
            self.shape = tuple(iface["shape"])
            self.strides = iface.get("strides")
            if self.strides is not None and len(self.strides) != len(self.shape):
                raise ValueError("The length of shape and strides must be equal")
        else:
            view = _PyBuffer()
            _get(obj, ct.byref(view), 0x11C)  # PyBUF_FULL_RO, retained until release.
            self._view = view
            if view.suboffsets:
                raise NotImplementedError("Suboffsets are not supported")
            self.ptr, self.readonly, self.itemsize = view.buf, bool(view.readonly), view.itemsize
            self.shape = tuple(view.shape[i] for i in range(view.ndim))
            self.strides = tuple(view.strides[i] for i in range(view.ndim)) if view.strides else None
        self.ndim = len(self.shape)
        self.nbytes = self.itemsize * math.prod(self.shape)
        self.c_contiguous = self._contiguous(False)
        self.f_contiguous = self._contiguous(True)
        self.contiguous = self.c_contiguous or self.f_contiguous

    def _contiguous(self, fortran):
        if self.strides is None:
            return not fortran or self.ndim <= 1
        stride = self.itemsize
        indices = range(self.ndim) if fortran else reversed(range(self.ndim))
        for i in indices:
            if self.shape[i] > 1 and self.strides[i] != stride:
                return False
            stride *= self.shape[i]
        return True

    def __del__(self):
        if self._view is not None and not sys.is_finalizing():
            _release(ct.byref(self._view))
            self._view = None


def asarray(obj):
    return obj if isinstance(obj, Array) else Array(obj)


def parse(buf, size=None, accept_host=True, writable=False, offset=0):
    arr = asarray(buf)
    if not arr.contiguous:
        raise ValueError("Array must be contiguous")
    if not accept_host and not arr.cuda:
        raise ValueError("Non-CUDA buffers not supported")
    if writable and arr.readonly:
        raise ValueError("Destination buffer is read-only")
    size = arr.nbytes if size is None else operator.index(size)
    offset = operator.index(offset)
    if size < 0 or offset < 0:
        raise OverflowError("can't convert negative value to size_t")
    if size + offset > arr.nbytes:
        raise ValueError("Size is greater than the size of the buffer")
    return arr, size

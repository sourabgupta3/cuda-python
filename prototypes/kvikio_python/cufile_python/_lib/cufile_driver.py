# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ctypes as ct

from ._runtime import native


class _Nvfs(ct.Structure):
    _fields_ = [
        ("major_version", ct.c_uint),
        ("minor_version", ct.c_uint),
        ("poll_thresh_size", ct.c_size_t),
        ("max_direct_io_size", ct.c_size_t),
        ("dstatusflags", ct.c_uint),
        ("dcontrolflags", ct.c_uint),
    ]


class _Props(ct.Structure):
    _fields_ = [
        ("nvfs", _Nvfs),
        ("fflags", ct.c_uint),
        ("max_device_cache_size", ct.c_uint),
        ("per_buffer_cache_size", ct.c_uint),
        ("max_device_pinned_mem_size", ct.c_uint),
        ("max_batch_io_size", ct.c_uint),
        ("max_batch_io_timeout_msecs", ct.c_uint),
    ]


def _query():
    result = _Props()
    native("driver_get_properties", ct.addressof(result))
    return result


def libcufile_version():
    return native("get_version")


def driver_open():
    native("driver_open")


def driver_close():
    native("driver_close")


class DriverProperties:
    def __init__(self):
        self._props = None

    def _get(self):
        if self._props is None:
            self._props = _query()
        return self._props

    @property
    def is_gds_available(self):
        try:
            props = self._get()
            return bool(props.nvfs.major_version or props.nvfs.minor_version)
        except (RuntimeError, ImportError, OSError):
            return False

    @property
    def major_version(self):
        return self._get().nvfs.major_version

    @property
    def minor_version(self):
        return self._get().nvfs.minor_version

    @property
    def allow_compat_mode(self):
        return bool(self._get().nvfs.dcontrolflags & 2)

    @property
    def poll_mode(self):
        return bool(self._get().nvfs.dcontrolflags & 1)

    @poll_mode.setter
    def poll_mode(self, value):
        native("driver_set_poll_mode", value, self.poll_thresh_size)
        props = self._get()
        props.nvfs.dcontrolflags = (props.nvfs.dcontrolflags & ~1) | int(bool(value))

    @property
    def poll_thresh_size(self):
        return self._get().nvfs.poll_thresh_size

    @poll_thresh_size.setter
    def poll_thresh_size(self, value):
        native("driver_set_poll_mode", self.poll_mode, value)
        self._get().nvfs.poll_thresh_size = value

    @property
    def max_device_cache_size(self):
        return self._get().max_device_cache_size

    @max_device_cache_size.setter
    def max_device_cache_size(self, value):
        self._get()
        native("driver_set_max_cache_size", value)
        self._get().max_device_cache_size = value

    @property
    def max_pinned_memory_size(self):
        return self._get().max_device_pinned_mem_size

    @max_pinned_memory_size.setter
    def max_pinned_memory_size(self, value):
        self._get()
        native("driver_set_max_pinned_mem_size", value)
        self._get().max_device_pinned_mem_size = value

    @property
    def per_buffer_cache_size(self):
        return self._get().per_buffer_cache_size

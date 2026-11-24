# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch parameter storage, native completion collection, and owner lifetimes."""

import ctypes as ct
import math
import operator
import sys
import threading
import time
import warnings
from contextlib import contextmanager

import numpy as np

from ._runtime import cufile, native, release_batch, retain_batch
from .arr import parse
from .buffer import registered_pointer


def _integer(value, name, minimum=0, maximum=(1 << 63) - 1):
    value = operator.index(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _timeout(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("timeout must be finite and nonnegative, or None")
    return value


class _Timespec(ct.Structure):
    _fields_ = [("tv_sec", ct.c_long), ("tv_nsec", ct.c_long)]


class BatchHandle:
    def __init__(self, capacity):
        self._handle = None
        self._lock = threading.Lock()
        self._pending = {}
        self._files = set()
        self._params = self._events = None
        self._events_data = None
        self._broken = False
        self.capacity = _integer(capacity, "capacity", 1, (1 << 32) - 1)
        self._version = native("get_version")
        self._handle = native("batch_io_set_up", self.capacity)

    @property
    def closed(self):
        return self._handle is None

    @property
    def pending(self):
        return len(self._pending)

    def check(self):
        if self.closed:
            raise RuntimeError("Batch is closed")
        if self._broken:
            raise RuntimeError("Batch completion is uncertain after a native failure; its resources remain retained")

    @contextmanager
    def _exclusive(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent operations on one Batch are not supported")
        try:
            yield
        finally:
            self._lock.release()

    def submit(self, operations):
        from ..batch import BatchOp
        from ..cufile import CuFile

        with self._exclusive():
            self.check()
            if self.pending:
                raise RuntimeError("Collect all batch completions before submitting again")
            # Limit consumption of generators, including accidentally infinite ones.
            entries = []
            for entry in operations:
                if len(entries) == self.capacity:
                    raise ValueError("Operation count exceeds batch capacity")
                entries.append(entry)
            if not entries:
                raise ValueError("A batch submission must contain at least one operation")
            # Use the bindings' ABI-derived dtype and public from_data interface.
            # Structured array fields avoid repeated nested recarray wrappers.
            data = np.zeros(len(entries), dtype=cufile.io_params_dtype)
            params = cufile.IOParams.from_data(data)
            pending, files, regions = {}, set(), []
            for index, op in enumerate(entries):
                if not isinstance(op, BatchOp):
                    raise TypeError("Each operation must be a BatchOp")
                if not isinstance(op.file, CuFile):
                    raise TypeError("BatchOp.file must be a CuFile")
                opcode = {"read": cufile.Opcode.READ, "write": cufile.Opcode.WRITE}.get(op.opcode, op.opcode)
                if opcode not in (cufile.Opcode.READ, cufile.Opcode.WRITE):
                    raise ValueError("opcode must be read, write, or a native Opcode value")
                write = opcode == cufile.Opcode.WRITE
                file = op.file._handle
                file._check(write)
                offset = _integer(op.dev_offset, "dev_offset")
                file_offset = _integer(op.file_offset, "file_offset")
                arr, size = parse(op.buf, op.size, writable=not write, offset=offset)
                _integer(size, "size", 1)
                _integer(file_offset + size, "file_offset + size")
                start, end = arr.ptr + offset, arr.ptr + offset + size
                regions.append((arr.cuda, start, end, not write))
                item = data[index]
                item["mode"] = cufile.BatchMode.BATCH
                item["fh"], item["opcode"], item["cookie"] = file._handle, opcode, index + 1
                # Host offsets are applied to the pointer: native 1.19 can ignore
                # dev_ptr_offset for host memory. GPU registration bases stay fixed.
                base, relative = registered_pointer(start) if arr.cuda else (start, 0)
                transfer = item["u"]["batch"]
                transfer["dev_ptr_base"] = base
                transfer["dev_ptr_offset"] = relative
                transfer["file_offset"] = file_offset
                transfer["size_"] = size
                pending[index + 1] = (arr, op.cookie, size)
                files.add(file)
            device, end, destination_end = None, 0, 0
            for current_device, lo, hi, destination in sorted(regions):
                if current_device != device:
                    device, end, destination_end = current_device, 0, 0
                if lo < (end if destination else destination_end):
                    raise ValueError("Batch transfers overlap a destination buffer")
                end = max(end, hi)
                if destination:
                    destination_end = max(destination_end, hi)
            if self._events is None:
                self._events_data = np.empty(self.capacity, dtype=cufile.io_events_dtype)
                self._events = cufile.IOEvents.from_data(self._events_data)
            self._params, self._pending, self._files = params, pending, files
            for file in files:
                file._batches.add(self)
            retain_batch(self)
            try:
                native("batch_io_submit", self._handle, len(entries), params.ptr, 0)
            except BaseException:
                # A native failure may follow partial acceptance. Releasing the
                # submitted pointers without known completion would permit DMA UAF.
                self._broken = True
                raise

    def _retire(self):
        for file in self._files:
            file._batches.discard(self)
        self._files.clear()
        self._params = None
        release_batch(self)

    def _collect(self, maximum, blocking=False):
        from ..batch import BatchCompletion

        if not self.pending:
            return ()
        if not blocking and self._version == 1190 and any(not arr.cuda for arr, _, _ in self._pending.values()):
            # Native status collection can release hostMemHandle before its
            # worker sets active. Waiting for every entry avoids that race.
            raise NotImplementedError(
                "Host-buffer batch polling and timed waits are unsafe with libcufile 1.19; use wait_all()"
            )
        maximum = min(maximum, self.pending)
        count, interval = ct.c_uint(maximum), _Timespec(0, 0)
        try:
            native(
                "batch_io_get_status",
                self._handle,
                maximum if blocking else 0,
                ct.addressof(count),
                self._events.ptr,
                0 if blocking else ct.addressof(interval),
            )
            if count.value > maximum:
                raise RuntimeError("Native batch returned too many completions")
            completed, seen = [], set()
            terminal = {
                int(status): status
                for status in (
                    cufile.Status.COMPLETE,
                    cufile.Status.CANCELED,
                    cufile.Status.FAILED,
                    cufile.Status.INVALID,
                    cufile.Status.TIMEOUT,
                )
            }
            for index in range(count.value):
                event = self._events_data[index]
                token, status = int(event["cookie"]), int(event["status"])
                if token not in self._pending or token in seen or status not in terminal:
                    raise RuntimeError("Invalid native batch completion cookie or status")
                seen.add(token)
                _, cookie, size = self._pending[token]
                result = ct.c_ssize_t(int(event["ret"])).value
                if result > size:
                    raise RuntimeError("Native batch completion exceeds submitted size")
                completed.append(BatchCompletion(token - 1, cookie, terminal[status], result))
        except BaseException:
            self._broken = True
            raise
        for token in seen:
            del self._pending[token]
        if not self.pending:
            self._retire()
        return tuple(completed)

    def poll(self, max_complete=None):
        with self._exclusive():
            self.check()
            maximum = (
                self.capacity if max_complete is None else _integer(max_complete, "max_complete", 1, self.capacity)
            )
            return self._collect(maximum)

    def _wait_all(self, timeout):
        deadline = None if timeout is None else time.monotonic() + timeout
        completed = []
        try:
            while self.pending:
                completed.extend(self._collect(self.pending, blocking=deadline is None))
                if self.pending and deadline is not None:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError("Batch is still active; pending resources remain retained")
                    time.sleep(min(left, 0.0005))
        except BaseException as error:
            error.completions = tuple(sorted(completed, key=lambda item: item.index))
            raise
        return tuple(sorted(completed, key=lambda item: item.index))

    def wait_all(self, timeout=None):
        timeout = _timeout(timeout)
        with self._exclusive():
            self.check()
            return self._wait_all(timeout)

    def drain(self, timeout=None):
        """File.close uses this to finish work before releasing native handles."""
        completed = self.wait_all(timeout)
        for item in completed:
            try:
                item.get()
            except RuntimeError as error:
                error.completions = completed
                raise

    def cancel(self):
        with self._exclusive():
            self.check()
            if not self.pending:
                return
            if self._version == 1190:
                raise NotImplementedError(
                    "Active batch cancellation is unsupported with libcufile 1.19; use wait_all()"
                )
            try:
                native("batch_io_cancel", self._handle)
            except BaseException:
                self._broken = True
                raise

    def close(self, *, timeout=None):
        timeout = _timeout(timeout)
        with self._exclusive():
            if self.closed:
                return
            self.check()
            completed = self._wait_all(timeout)
            native("batch_io_destroy", self._handle)
            self._handle = self._events = self._events_data = None
            for item in completed:
                try:
                    item.get()
                except RuntimeError as error:
                    error.completions = completed
                    raise

    def __del__(self):
        if not sys.is_finalizing():
            try:
                self.close()
            except (OSError, RuntimeError) as error:
                warnings.warn(f"Batch cleanup: {error}", RuntimeWarning, stacklevel=2)

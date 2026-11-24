# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native cuFile batches spanning multiple file handles."""

import ctypes
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from cuda.bindings.cufile import Status

from ._buffers import _BufferArg
from ._file import File
from ._runtime import CuFileError, _ScopedResource, call, integer, lib, runtime, transfer_error


class IOOpcode(StrEnum):
    """Read/write operation name; a value is required and has no default."""

    READ = "read"
    WRITE = "write"


@dataclass(frozen=True)
class BatchIOParams:
    """One batch transfer. File and buffer ownership is acquired by submit().

    file, buf, opcode, and size are required. Both byte offsets default to 0;
    cookie defaults to None. No operation or byte count is inferred.
    """

    file: File
    buf: object
    opcode: IOOpcode | str
    size: int
    file_offset: int = 0
    buffer_offset: int = 0
    cookie: object = None

    def __post_init__(self):
        if not isinstance(self.file, File):
            raise TypeError("file must be a File")
        object.__setattr__(self, "opcode", IOOpcode(self.opcode))
        integer(self.size, "size", minimum=1)
        integer(self.file_offset, "file_offset")
        integer(self.buffer_offset, "buffer_offset")


@dataclass(frozen=True)
class IOCompletion:
    """Copied per-operation completion; independent of native array reuse.

    Batch produces these objects. For manual construction, index, cookie,
    status, and bytes_transferred are required; error and native_result
    default to None. status uses cuda.bindings.cufile.Status directly.
    index increases across submissions within one Batch. result() returns the
    count or raises the stored error.
    """

    index: int
    cookie: object
    status: Status
    bytes_transferred: int
    error: CuFileError | None = None
    native_result: int | None = None

    def result(self) -> int:
        if self.error is not None:
            raise self.error
        return self.bytes_transferred


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


def _timeout(value):
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("timeout must be finite and nonnegative, or None")
    return result


class _BatchRequest:
    """Retain one request until GetStatus returns its terminal completion."""

    observed = False

    def __init__(self, batch, index, state, argument, operation, native):
        self.batch = batch
        self.index = index
        self.state = state
        self.argument = argument
        self.operation = operation
        self.native = native

    def wait(self):
        if self.batch is not None:
            self.batch.wait_all()

    def release(self):
        self.batch = None
        state, argument = self.state, self.argument
        self.state = self.argument = self.operation = self.native = None
        state.work.discard(self)
        if argument.record is not None:
            argument.record.work.discard(self)
        try:
            argument.release()
        finally:
            state.release()


class Batch(_ScopedResource):
    """Native queue accepting requests up to its outstanding capacity.

    submit(entries) packs and enqueues new requests, including while others are
    outstanding. poll() returns newly collected completions once. wait_all()
    waits for the remaining requests; it never repeats earlier poll results.
    If a wait raises, its completions attribute carries results collected by that
    call. libcufile 1.19 rolling submission is rejected before native mutation.
    Timeouts are seconds. Create a File or Driver before native batch setup.
    Mutating one batch concurrently is unsupported.
    """

    def __init__(self, capacity: int) -> None:
        self._handle = None
        self._pending = {}
        self._events = None
        self._uncertain = False
        self._next_index = 0
        self._context = None
        self._capacity = integer(capacity, "capacity", minimum=1, maximum=(1 << 32) - 1)
        runtime.check()
        self._handle = call("batch_io_set_up", self.capacity)
        runtime.track(self)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_closed(self) -> bool:
        return self._handle is None

    @property
    def is_active(self) -> bool:
        return bool(self._pending)

    def _check(self):
        runtime.check(allow_shutdown=True)
        if self.is_closed:
            raise RuntimeError("Batch has been closed")
        if self._uncertain:
            raise RuntimeError("Batch submission state is uncertain; its resources were retained")

    def _release(self, requests):
        error = None
        for request in requests:
            self._pending.pop(request.index + 1, None)
            try:
                request.release()
            except Exception as exc:
                if error is None:
                    error = exc
        if not self._pending:
            runtime.remove_work(self)
            self._context = None
        if error is not None:
            raise error

    def submit(self, operations: Iterable[BatchIOParams]) -> None:
        runtime.check()
        self._check()
        operations = tuple(operations)
        if not 1 <= len(operations) <= self.capacity - len(self._pending):
            raise ValueError("Operation count must be between 1 and the available batch capacity")
        if self.is_active and call("get_version") == 1190:
            raise NotImplementedError(
                "Rolling batch submission is disabled with libcufile 1.19: partial resubmission can corrupt "
                "memory. Collect remaining completions with wait_all() before submitting new entries."
            )
        integer(self._next_index + len(operations), "request index")
        native = lib().IOParams(len(operations))
        requests = []
        contexts = {r.argument.lease.context_id for r in self._pending.values()} - {0}
        try:
            for offset, operation in enumerate(operations):
                if not isinstance(operation, BatchIOParams):
                    raise TypeError("Each operation must be BatchIOParams")
                operation.file._check_open()
                argument = _BufferArg(
                    operation.buf, operation.size, operation.buffer_offset, writable=operation.opcode == "read"
                )
                try:
                    operation.file._state.acquire()
                except BaseException:
                    argument.release()
                    raise
                request = _BatchRequest(
                    self, self._next_index + offset, operation.file._state, argument, operation, native
                )
                requests.append(request)
                if argument.lease.context_id:
                    contexts.add(argument.lease.context_id)
                item = native[offset]
                item.mode = lib().BatchMode.BATCH
                item.fh = request.state.handle
                item.opcode = lib().Opcode.READ if operation.opcode == "read" else lib().Opcode.WRITE
                item.cookie = request.index + 1
                # libcufile 1.19 can ignore dev_ptr_offset for registered host memory.
                host_offset = 0 if argument.lease.is_device else argument.offset
                item.u.batch.dev_ptr_base = argument.ptr + host_offset
                item.u.batch.size_ = argument.size
                item.u.batch.file_offset = operation.file_offset
                item.u.batch.dev_ptr_offset = argument.offset - host_offset
            if len(contexts) > 1:
                raise ValueError("Outstanding batch buffers must belong to one CUDA context")
            context = next(
                (
                    r.argument.lease.context()
                    for r in (*self._pending.values(), *requests)
                    if r.argument.lease.context_id
                ),
                requests[0].argument.lease.context(),
            )
            if self._events is None:
                self._events = lib().IOEvents(self.capacity)
        except BaseException as exc:
            for request in requests:
                try:
                    request.release()
                except Exception as cleanup:
                    exc.add_note(f"Batch submission rollback also failed: {cleanup}")
            raise
        was_active = self.is_active
        previous_context, self._context = self._context, context
        for request in requests:
            self._pending[request.index + 1] = request
            request.state.work.add(request)
            request.argument.attach(request)
        if not was_active:
            runtime.add_work(self)
        attempted = submitted = False
        try:
            with context:
                attempted = True
                call("batch_io_submit", self._handle, len(requests), native.ptr, 0)
                submitted = True
                self._next_index += len(requests)
        except BaseException as exc:
            if not attempted:
                try:
                    self._release(requests)
                except Exception as cleanup:
                    exc.add_note(f"Batch submission rollback also failed: {cleanup}")
                self._context = previous_context
            elif not submitted:
                self._uncertain = True
                runtime.quarantine(self)
            raise

    def poll(
        self, min_complete: int = 0, max_complete: int | None = None, timeout: float | None = 0.0
    ) -> tuple[IOCompletion, ...]:
        self._check()
        minimum = integer(min_complete, "min_complete", maximum=self.capacity)
        maximum = (
            self.capacity
            if max_complete is None
            else integer(max_complete, "max_complete", minimum=1, maximum=self.capacity)
        )
        if minimum > maximum:
            raise ValueError("min_complete cannot exceed max_complete")
        timeout = _timeout(timeout)
        if not self.is_active:
            return ()
        maximum = min(maximum, len(self._pending))
        minimum = min(minimum, maximum)
        count = ctypes.c_uint(maximum)
        interval = None if timeout is None else _Timespec(int(timeout), int((timeout % 1) * 1_000_000_000))
        polled = False
        error = None
        try:
            with self._context:
                call(
                    "batch_io_get_status",
                    self._handle,
                    minimum,
                    ctypes.addressof(count),
                    self._events.ptr,
                    0 if interval is None else ctypes.addressof(interval),
                )
                polled = True
        except BaseException as exc:
            if not polled:
                self._uncertain = True
                runtime.quarantine(self)
                raise
            error = exc
        completions = []
        statuses = {
            int(s): s for s in (Status.COMPLETE, Status.CANCELED, Status.FAILED, Status.INVALID, Status.TIMEOUT)
        }
        for offset in range(count.value):
            event = self._events[offset]
            token = int(event.cookie)
            request = self._pending.get(token)
            status = statuses.get(int(event.status))
            if request is None or status is None:
                self._uncertain = True
                runtime.quarantine(self)
                invalid = RuntimeError("Invalid cookie or nonterminal batch completion")
                invalid.completions = tuple(completions)
                raise invalid
            value = ctypes.c_ssize_t(int(event.ret)).value
            failure = None
            if value < 0:
                failure = transfer_error(value, operation=f"batch[{request.index}]")
            elif status != Status.COMPLETE:
                failure = CuFileError(
                    f"Batch operation {request.index}: {status.name}, native result {value}",
                    operation=f"batch[{request.index}]",
                    native_result=value,
                )
            completions.append(
                IOCompletion(
                    request.index, request.operation.cookie, status, max(value, 0), failure, native_result=value
                )
            )
            try:
                self._release((request,))
            except Exception as exc:
                if error is None:
                    error = exc
        if error is not None:
            error.completions = tuple(completions)
            raise error
        return tuple(completions)

    def wait_all(self, timeout: float | None = None) -> tuple[IOCompletion, ...]:
        self._check()
        timeout = _timeout(timeout)
        deadline = None if timeout is None else time.monotonic() + timeout
        completions = []
        try:
            while self.is_active:
                interval = None if deadline is None else max(deadline - time.monotonic(), 0)
                completions.extend(self.poll(len(self._pending), timeout=interval))
                if self.is_active and deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Batch is still active; its resources remain retained")
        except BaseException as exc:
            completions.extend(getattr(exc, "completions", ()))
            exc.completions = tuple(sorted(completions, key=lambda item: item.index))
            raise
        result = tuple(sorted(completions, key=lambda item: item.index))
        for completion in result:
            if completion.error is not None:
                completion.error.completions = result
                raise completion.error
        return result

    def cancel(self) -> None:
        self._check()
        if self.is_active:
            if call("get_version") == 1190:
                raise NotImplementedError(
                    "Native batch cancellation is disabled with libcufile 1.19: "
                    "unreaped completions can be lost. Drain the batch with wait_all()."
                )
            with self._context:
                call("batch_io_cancel", self._handle)

    def close(self, *, timeout: float | None = None) -> None:
        if self.is_closed:
            return
        self._check()
        error = None
        if self.is_active:
            try:
                self.wait_all(timeout=timeout)
            except TimeoutError:
                raise
            except Exception as exc:
                if self.is_active:
                    runtime.quarantine(self)
                    raise
                error = exc
        call("batch_io_destroy", self._handle)
        self._handle = None
        self._events = self._context = None
        if error is not None:
            raise error

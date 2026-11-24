# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exclusive native registrations retained while wrapper-managed I/O uses them."""

import threading

from cuda.bindings import driver
from cuda.core._cufile_native import BufferLease, StreamLease
from cuda.core._utils.cuda_utils import handle_return

from ._runtime import CuFileError, _ScopedResource, _wait_for_work, call, integer, lib, runtime

_memory = {}
_streams = {}
_lock = threading.RLock()


class _Registration:
    def __init__(self, lease, key, kind, *, flags=0):
        self.lease = lease
        self.key = key
        self.kind = kind
        self.flags = flags
        self.users = 0
        self.retired = False
        self.registered = False
        self.creating = True
        self.destroying = False
        self.work = set()
        runtime.track(self)

    def release(self):
        with _lock:
            self.retired = True
        error = None
        for work in tuple(self.work):
            try:
                _wait_for_work(work)
            except Exception as exc:
                if error is None:
                    error = exc
        self._destroy()
        if error is not None:
            raise error

    def _destroy(self):
        with _lock:
            if not self.retired or self.users or self.lease is None or self.destroying:
                return
            self.destroying = True
        try:
            if self.registered:
                with self.lease.context():
                    call("buf_deregister" if self.kind == "memory" else "stream_deregister", self.key)
                    self.registered = False
        except BaseException:
            with _lock:
                self.destroying = False
            raise
        with _lock:
            registry = _memory if self.kind == "memory" else _streams
            registry.pop(self.key if self.kind == "memory" else self.registry_key, None)
            self.lease = None
            self.destroying = False

    def close(self):
        if self.lease is not None:
            self.release()


def _register_memory(lease, size, flags):
    try:
        base = lease.ptr
        if lease.is_device:
            with lease.context():
                memory_type = handle_return(
                    driver.cuPointerGetAttribute(driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE, base)
                )
                if memory_type == driver.CUmemorytype.CU_MEMORYTYPE_DEVICE:
                    capable = handle_return(
                        driver.cuPointerGetAttribute(
                            driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE, base
                        )
                    )
                    if not capable:
                        raise NotImplementedError(
                            "Device-buffer registration requires GPUDirect-capable memory, "
                            "including in compatibility mode; use unregistered I/O for this allocation"
                        )
        runtime.check()
        with _lock:
            if any(base < record.key + record.size and record.key < base + size for record in _memory.values()):
                raise CuFileError(
                    "Buffer range is already registered or overlaps an existing registration",
                    lib().OpError.MEMORY_ALREADY_REGISTERED,
                    operation="buf_register",
                )
            record = _Registration(lease, base, "memory", flags=flags)
            record.size = size
            _memory[base] = record
        return _finish_registration(record, "buf_register", base, size, flags)
    finally:
        # A retained exception traceback must not keep an extra buffer export.
        lease = None


def _register_stream(stream, flags):
    lease = StreamLease(stream)
    runtime.check()
    with _lock:
        key = (lease.context_id, lease.ptr, threading.get_ident() if lease.ptr == 2 else None)
        if key in _streams:
            lease = None
            raise CuFileError(
                "Stream is already registered",
                lib().OpError.INVALID_VALUE,
                operation="stream_register",
            )
        record = _Registration(lease, lease.ptr, "stream", flags=flags)
        record.registry_key = key
        _streams[key] = record
    lease = None
    return _finish_registration(record, "stream_register", record.key, flags)


def _finish_registration(record, function, *args):
    try:
        with record.lease.context():
            call(function, *args)
            record.registered = True
        record.creating = False
    except BaseException as exc:
        try:
            record.close()
        except Exception as cleanup:
            runtime.quarantine(record)
            exc.add_note(f"Registration rollback also failed: {cleanup}")
        raise
    return record


class RegisteredBuffer(_ScopedResource):
    """Scoped cuFile registration retaining an existing host or device buffer.

    Each object owns one native registration. Duplicate or overlapping tracked
    buffer ranges raise CuFileError; registrations are never shared or resized.
    close() drains pending I/O before deregistration. Registrations created
    elsewhere are not adopted.

    size defaults to the supplied buffer's byte length, not the underlying
    allocation's capacity. Explicit sizes must be positive and fit the buffer.
    flags defaults to 0; the native argument is reserved and only 0 is supported.
    The buffer is required; there is no implicit allocation.

    Device memory without CUDA's GPUDirect capability raises NotImplementedError
    before native registration in both GDS and compatibility modes. This check
    uses pointer attributes, including for buffers owned by other libraries.
    Such allocations can still be passed directly to unregistered I/O.
    """

    def __init__(self, buf: object, size: int | None = None, *, flags: int = 0) -> None:
        self._record = None
        self._lease = None
        self._flags = integer(flags, "flags", maximum=0)
        try:
            self._lease = BufferLease(buf)
            self._size = integer(
                self._lease.nbytes if size is None else size, "size", minimum=1, maximum=self._lease.nbytes
            )
            self._record = _register_memory(self._lease, self._size, self._flags)
        except BaseException:
            self._lease = None
            raise

    @property
    def size(self) -> int:
        """Registered byte count, also available after close()."""
        return self._size

    @property
    def flags(self) -> int:
        """The native registration flags; currently always 0."""
        return self._flags

    @property
    def is_closed(self) -> bool:
        return self._record is None

    def close(self) -> None:
        if self._record is not None:
            self._record.release()
            self._record = None
            self._lease = None


class RegisteredStream(_ScopedResource):
    """Scoped cuFile registration retaining an existing CUDA stream.

    Each object owns one native registration. Registering the same tracked
    stream again raises CuFileError, even with matching flags. close() drains
    pending I/O before deregistration and leaves the CUDA stream open.
    Registrations made outside this wrapper are not adopted.

    flags defaults to 0: size and offsets are read at execution time. Bits
    0x1, 0x2, and 0x4 fix buffer offset, file offset, and size at submission.
    Bit 0x8 promises 4096-byte alignment of the address, offsets, and size.
    Combine bits with |; 0xF selects all four promises. The stream is required.
    """

    def __init__(self, stream: object, flags: int = 0) -> None:
        self._record = None
        self._flags = integer(flags, "flags", maximum=15)
        self._record = _register_stream(stream, self._flags)

    @property
    def flags(self) -> int:
        """The native registration bitmask supplied at construction."""
        return self._flags

    @property
    def is_closed(self) -> bool:
        return self._record is None

    def close(self) -> None:
        if self._record is not None:
            self._record.release()
            self._record = None


class _BufferArg:
    def __init__(self, buf, size, buffer_offset, *, writable):
        self.record = None
        if isinstance(buf, RegisteredBuffer):
            if buf.is_closed:
                raise RuntimeError("Buffer registration has been closed")
            self.lease = buf._lease
            self.nbytes = buf.size
        else:
            self.lease = BufferLease(buf, writable=writable)
            self.nbytes = self.lease.nbytes
        if writable and self.lease.readonly:
            raise ValueError("Read destination must be writable")
        self.offset = integer(buffer_offset, "buffer_offset", maximum=self.nbytes)
        self.size = self.nbytes - self.offset if size is None else integer(size, "size")
        if self.offset + self.size > self.nbytes:
            raise ValueError("Transfer exceeds the supplied buffer")
        self.ptr = self.lease.ptr
        with _lock:
            if isinstance(buf, RegisteredBuffer):
                record = buf._record
            else:
                start = self.ptr + self.offset
                end = start + self.size
                record = next(
                    (r for r in _memory.values() if start < r.key + r.size and (r.key < end or start == end == r.key)),
                    None,
                )
            if record is not None:
                if record.retired or record.creating:
                    raise RuntimeError("Buffer registration is being created or released")
                offset = self.ptr + self.offset - record.key
                if offset < 0 or offset + self.size > record.size:
                    raise ValueError("Transfer exceeds the registered buffer range")
                self.offset = offset
                self.ptr = record.key
                self.record = record
                record.users += 1
        if self.record is None:
            self.ptr += self.offset
            self.offset = 0

    def attach(self, work):
        if self.record is not None:
            self.record.work.add(work)

    def release(self):
        try:
            if self.record is not None:
                record, self.record = self.record, None
                with _lock:
                    record.users -= 1
                record._destroy()
        finally:
            self.lease = None


def _stream_arg(stream, buffer):
    record = None
    if isinstance(stream, RegisteredStream):
        if stream.is_closed:
            raise RuntimeError("Stream registration has been closed")
        record = stream._record
        if buffer.context_id and record.lease.context_id != buffer.context_id:
            raise ValueError("Stream and buffer belong to different CUDA contexts")
        return record.lease, record
    lease = StreamLease(stream, buffer)
    key = (lease.context_id, lease.ptr, threading.get_ident() if lease.ptr == 2 else None)
    with _lock:
        record = _streams.get(key)
        if record is not None and (record.retired or record.creating):
            raise RuntimeError("Stream registration is being created or released")
        return lease, record

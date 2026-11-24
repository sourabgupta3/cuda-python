# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native synchronous and CUDA-stream file transfers."""

import ctypes
import os
import threading

from ._buffers import _BufferArg, _stream_arg
from ._runtime import CuFileError, _ScopedResource, _wait_for_work, call, check_count, integer, lib, runtime


def _native_transfer(handle, ptr, size, file_offset, buffer_offset, write):
    return runtime.get_native_io().transfer(handle, ptr, size, file_offset, buffer_offset, write)


class _FileState:
    def __init__(self, path, mode, flags):
        modes = {
            "r": os.O_RDONLY,
            "r+": os.O_RDWR,
            "w": os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            "w+": os.O_RDWR | os.O_CREAT | os.O_TRUNC,
        }
        if mode not in modes:
            raise ValueError("Supported file modes are r, r+, w, and w+")
        flags = integer(flags, "flags", maximum=(1 << 31) - 1)
        supported = os.O_DIRECT | os.O_SYNC | os.O_DSYNC | os.O_EXCL | os.O_CLOEXEC
        if flags & ~supported:
            raise ValueError(
                "flags supports O_DIRECT, O_SYNC, O_DSYNC, O_EXCL, and O_CLOEXEC; "
                "use mode to select access, creation, and truncation"
            )
        if flags & os.O_EXCL and not modes[mode] & os.O_CREAT:
            raise ValueError("O_EXCL requires mode w or w+")
        self.fd = self.handle = None
        self.closed = False
        self.users = 0
        self.lock = threading.RLock()
        self.destroying = False
        self.work = set()
        runtime.check()
        try:
            self.fd = os.open(path, modes[mode] | flags | os.O_CLOEXEC, 0o666)
            descriptor = lib().Descr()
            descriptor.type = lib().FileHandleType.OPAQUE_FD
            descriptor.handle.fd = self.fd
            descriptor.fs_ops = 0
            self.handle = call("handle_register", descriptor.ptr)
            runtime.track(self)
        except BaseException as exc:
            self.closed = True
            try:
                self._destroy()
            except Exception as cleanup:
                runtime.quarantine(self)
                exc.add_note(f"File construction rollback also failed: {cleanup}")
            raise

    def acquire(self):
        runtime.check()
        with self.lock:
            if self.closed:
                raise RuntimeError("File has been closed")
            self.users += 1

    def release(self):
        with self.lock:
            self.users -= 1
        self._destroy()

    def _destroy(self):
        with self.lock:
            if not self.closed or self.users or self.destroying:
                return
            self.destroying = True
        try:
            if self.handle is not None:
                call("handle_deregister", self.handle)
                self.handle = None
            if self.fd is not None:
                fd, self.fd = self.fd, None
                os.close(fd)
        finally:
            with self.lock:
                self.destroying = False

    def close(self):
        self.closed = True
        error = None
        for work in tuple(self.work):
            try:
                _wait_for_work(work)
            except Exception as exc:
                if not work.observed and error is None:
                    error = exc
        self._destroy()
        if error is not None:
            raise error


class _Work:
    def __init__(self, state, argument, operation):
        self.state = state
        self.argument = argument
        self.operation = operation
        self.stream = None
        self.registration = None
        self.event = None
        self.storage = None
        self.finished = False
        self.observed = False
        self.error = None
        self.count = 0
        self.lock = threading.RLock()
        state.acquire()
        try:
            state.work.add(self)
            argument.attach(self)
            runtime.add_work(self)
        except BaseException:
            state.work.discard(self)
            if argument.record is not None:
                argument.record.work.discard(self)
            state.release()
            raise

    def _release(self):
        self.finished = True
        self.state.work.discard(self)
        if self.argument.record is not None:
            self.argument.record.work.discard(self)
        error = None
        try:
            self.argument.release()
        except Exception as exc:
            error = exc
        if self.registration is not None:
            self.registration.work.discard(self)
        try:
            self.state.release()
        except Exception as exc:
            if error is None:
                error = exc
        if self.event is not None:
            try:
                self.event.close()
            except Exception as exc:
                if error is None:
                    error = exc
            self.event = None
        self.argument = None
        self.stream = None
        self.storage = None
        runtime.remove_work(self)
        if error is not None:
            raise error

    def wait(self):
        with self.lock:
            if not self.finished:
                if self.stream is not None:
                    try:
                        if self.event is None:
                            self.stream.sync()
                        else:
                            self.event.sync()
                    except Exception:
                        runtime.quarantine(self)
                        raise
                self._complete()
            if self.error is not None:
                raise self.error
            return self.count

    def _complete(self):
        try:
            self.count = check_count(self.storage[-1].value, operation=self.operation)
        except Exception as exc:
            self.error = exc
        try:
            self._release()
        except Exception as cleanup:
            if self.error is None:
                self.error = cleanup
            else:
                self.error.add_note(f"Transfer cleanup also failed: {cleanup}")

    def result(self):
        with self.lock:
            if not self.finished:
                if self.event is None or not self.event.is_done:
                    raise RuntimeError("Stream I/O result is not ready")
                self._complete()
            self.observed = True
            if self.error is not None:
                raise self.error
            return self.count


class IOParams(_ScopedResource):
    """Arguments for one synchronous or CUDA-stream transfer at a time.

    File.read/write return the completed byte count directly. For
    File.read_async/write_async, wait for the CUDA stream or an event recorded
    after the transfer, then read bytes_done. This property is only for stream
    I/O and does not wait. Stream registration flags determine which arguments
    can change while pending. close() waits for unfinished stream I/O before
    releasing resources.

    buf is required. size defaults to None (bytes remaining after the buffer
    offset); file_offset and buffer_offset default to 0 bytes. No stream or
    result is associated at construction.
    """

    def __init__(self, buf: object, size: int | None = None, *, file_offset: int = 0, buffer_offset: int = 0) -> None:
        self._closed = True
        self._work = self._storage = None
        self._offset_bias = 0
        self._buf = buf
        self._size = None if size is None else integer(size, "size")
        self._file_offset = integer(file_offset, "file_offset")
        self._buffer_offset = integer(buffer_offset, "buffer_offset")
        self._closed = False

    @property
    def buf(self) -> object:
        return self._buf

    @property
    def size(self) -> int | None:
        return self._size if self._storage is None else self._storage[0].value

    @size.setter
    def size(self, value: int | None) -> None:
        self._set_argument(0, value)

    @property
    def file_offset(self) -> int:
        return self._file_offset if self._storage is None else self._storage[1].value

    @file_offset.setter
    def file_offset(self, value: int) -> None:
        self._set_argument(1, value)

    @property
    def buffer_offset(self) -> int:
        return self._buffer_offset if self._storage is None else self._storage[2].value - self._offset_bias

    @buffer_offset.setter
    def buffer_offset(self, value: int) -> None:
        self._set_argument(2, value)

    def _set_argument(self, index, value):
        if self._closed:
            raise RuntimeError("IOParams has been closed")
        names = ("size", "file_offset", "buffer_offset")
        if value is None and index == 0:
            if self._work is not None and not self._work.finished:
                raise ValueError("A pending stream transfer needs a bounded size")
            self._file_offset = self.file_offset
            self._buffer_offset = self.buffer_offset
            self._size = None
            self._storage = None
            return
        value = integer(value, names[index])
        work = self._work
        if work is not None and not work.finished:
            flags = work.registration.flags if work.registration is not None else 0
            if flags & (4, 2, 1)[index]:
                raise ValueError(f"{names[index]} was fixed at submission by the stream registration flags")
            if index == 2 and work.argument.record is None and value != self.buffer_offset:
                raise ValueError("Deferred buffer offsets require registered memory")
            size = value if index == 0 else self.size
            offset = value if index == 2 else self.buffer_offset
            file_offset = value if index == 1 else self.file_offset
            if size > work.max_size or offset + size > work.argument.nbytes:
                raise ValueError("Deferred arguments exceed the submitted size or supplied buffer")
            record = work.argument.record
            if record is not None and (
                offset + self._offset_bias < 0 or offset + self._offset_bias + size > record.size
            ):
                raise ValueError("Deferred arguments exceed the registered buffer range")
            if flags & 8 and (
                size % 4096 or file_offset % 4096 or (work.argument.ptr + offset + self._offset_bias) % 4096
            ):
                raise ValueError("Deferred arguments violate the stream alignment flag")
        setattr(self, "_" + names[index], value)
        if self._storage is not None:
            self._storage[index].value = value + (self._offset_bias if index == 2 else 0)

    @property
    def bytes_done(self) -> int:
        """Return a completed stream byte count or raise the transfer error.

        This property does not wait for I/O. Read it after stream or event
        completion; earlier access raises RuntimeError. Repeated reads use
        the saved result, which remains readable after close(). Synchronous
        methods return counts directly and clear the previous stream result.
        """
        if self._work is None:
            raise RuntimeError("No stream I/O has been submitted with these parameters")
        return self._work.result()

    @property
    def is_closed(self) -> bool:
        return self._closed

    def _check_idle(self):
        if self._closed:
            raise RuntimeError("IOParams has been closed")
        if self._work is not None and not self._work.finished:
            self._work.result()

    def close(self) -> None:
        if not self._closed:
            if self._work is not None and not self._work.finished:
                self._work.observed = True
                try:
                    self._work.wait()
                finally:
                    if self._work.finished:
                        self._closed = True
                        self._buf = None
            else:
                self._closed = True
                self._buf = None


def _record_completion(stream):
    return stream.record_event()


def _transfer(state, argument, opcode, size, file_offset, buffer_offset):
    if not size:
        return 0
    with argument.lease.context():
        return _native_transfer(state.handle, argument.ptr, size, file_offset, buffer_offset, opcode == "write")


class File(_ScopedResource):
    """An open local file and its native cuFile registration.

    All read/write methods take IOParams. Synchronous methods return the byte
    count directly; stream methods store their result in IOParams.bytes_done.
    All offsets and counts are bytes; callers establish
    producer readiness and order subsequent GPU consumers explicitly.

    File-handle registration relies on native implicit driver initialization.
    This object never calls driver-open or driver-close. Keep an explicit Driver
    open until its dependent I/O and resource cleanup finish.

    ``mode`` defaults to r (read an existing file). Other modes are r+
    (read/write existing), w (write, create/truncate), and w+ (read/write,
    create/truncate). Creation uses 0o666 subject to the process umask.
    Keyword-only ``flags`` supplies additional os.open flags and defaults to
    os.O_DIRECT; pass 0 to omit it. Supported flags are os.O_DIRECT, os.O_SYNC,
    os.O_DSYNC, os.O_EXCL (with w/w+), and os.O_CLOEXEC. The descriptor always
    has O_CLOEXEC. Unsupported flags are rejected before opening the file.
    """

    def __init__(self, path: os.PathLike[str] | str, mode: str = "r", *, flags: int = os.O_DIRECT) -> None:
        self._state = None
        try:
            self._state = _FileState(os.fspath(path), mode, flags)
        except OSError as exc:
            error = CuFileError(
                str(exc), errno=exc.errno, filename=exc.filename, strerror=exc.strerror, operation="open"
            )
            raise error from exc

    @property
    def is_closed(self) -> bool:
        return self._state is None or self._state.closed

    def close(self) -> None:
        """Wait for pending file I/O, then deregister and close the native handle."""
        if self._state is not None:
            self._state.close()

    def _check_open(self):
        runtime.check()
        if self.is_closed:
            raise RuntimeError("File has been closed")

    def fileno(self) -> int:
        """Return a borrowed file descriptor, for example for os.fsync()."""
        self._check_open()
        return self._state.fd

    def _sync(self, opcode, params):
        self._check_open()
        if not isinstance(params, IOParams):
            raise TypeError("params must be IOParams")
        params._check_idle()
        file_offset = integer(params.file_offset, "file_offset")
        argument = _BufferArg(params.buf, params.size, params.buffer_offset, writable=opcode == "read")
        try:
            # A synchronous submission replaces any previous stream result.
            # Keep the current argument values when detaching stream storage.
            params._size, params._file_offset, params._buffer_offset = (params.size, file_offset, params.buffer_offset)
            params._work = params._storage = None
            params._offset_bias = 0
            return _transfer(self._state, argument, opcode, argument.size, file_offset, argument.offset)
        finally:
            argument.release()

    def read(self, params: IOParams) -> int:
        """Synchronously read using params and return the completed byte count.

        params.size=None transfers the bytes remaining after buffer_offset.
        A short read, including EOF, returns the actual count. Errors raise
        directly. This operation does not set params.bytes_done or synchronize
        a GPU stream.
        """
        return self._sync("read", params)

    def write(self, params: IOParams) -> int:
        """Synchronously write using params; return actual bytes or raise the error.

        This operation does not set params.bytes_done or synchronize a GPU stream.
        """
        return self._sync("write", params)

    def _async(self, opcode, params, stream_arg):
        self._check_open()
        if not isinstance(params, IOParams):
            raise TypeError("params must be IOParams")
        params._check_idle()
        file_offset = params.file_offset
        buffer_offset = params.buffer_offset
        argument = _BufferArg(params.buf, params.size, buffer_offset, writable=opcode == "read")
        # Registered memory keeps its base and a deferred offset. For
        # unregistered memory the adapter folds the offset into the address.
        params._offset_bias = argument.offset - buffer_offset
        work = None
        submitted = False
        try:
            stream, registration = _stream_arg(stream_arg, argument.lease)
            if (
                registration is not None
                and registration.flags & 8
                and (argument.size % 4096 or file_offset % 4096 or (argument.ptr + argument.offset) % 4096)
            ):
                raise ValueError("Stream alignment flag requires 4096-byte aligned buffer, offset, and size")
            work = _Work(self._state, argument, f"{opcode}_async")
            work.stream, work.registration = stream, registration
            work.max_size = argument.size
            if registration is not None:
                registration.work.add(work)
            work.storage = (
                ctypes.c_size_t(argument.size),
                ctypes.c_int64(file_offset),
                ctypes.c_int64(argument.offset),
                ctypes.c_ssize_t(0),
            )
            params._storage = work.storage
            if argument.size:
                with stream.context():
                    call(
                        work.operation,
                        self._state.handle,
                        argument.ptr,
                        *(ctypes.addressof(item) for item in work.storage),
                        stream.ptr,
                    )
                    submitted = True
                work.event = _record_completion(stream)
            else:
                work.stream = None
                work._release()
        except BaseException as exc:
            try:
                if work is not None:
                    if submitted:
                        work.wait()
                    else:
                        work._release()
                else:
                    argument.release()
            except Exception as cleanup:
                exc.add_note(f"Asynchronous submission rollback also failed: {cleanup}")
            raise
        params._work = work

    def read_async(self, params: IOParams, stream: object) -> None:
        """Enqueue a read; read params.bytes_done after stream/event completion."""
        self._async("read", params, stream)

    def write_async(self, params: IOParams, stream: object) -> None:
        """Enqueue a write; read params.bytes_done after stream/event completion."""
        self._async("write", params, stream)

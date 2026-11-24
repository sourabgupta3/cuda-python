# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ctypes as ct
import fcntl
import io
import mmap
import operator
import os
import sys
import warnings
from concurrent.futures import wait

from . import defaults
from ._runtime import Context, cuda, cufile, driver, native, pool, retain_stream
from .arr import parse
from .future import IOFuture, IOFutureStream


def _unsigned(value):
    value = operator.index(value)
    if value < 0:
        raise OverflowError("can't convert negative value to size_t")
    return value


def _flags(mode):
    mode = str(mode).replace("b", "")
    if mode not in ("r", "r+", "w", "w+", "a", "a+"):
        raise ValueError("Unknown file open flag")
    flags = os.O_RDWR if "+" in mode else os.O_RDONLY if mode == "r" else os.O_WRONLY
    if mode.startswith("w"):
        flags |= os.O_CREAT | os.O_TRUNC
    if mode.startswith("a"):
        flags |= os.O_CREAT | os.O_APPEND
    return flags


class CuFile:
    def __init__(self, file_path, flags="r"):
        self._fd = self._direct = self._handle = None
        self._pending, self._streams = set(), set()
        self._batches = set()
        self.path = os.fspath(file_path)
        self.flags = _flags(flags)
        try:
            self._fd = os.open(self.path, self.flags | os.O_CLOEXEC, 0o644)
            self._pool_device = os.fstat(self._fd).st_dev if defaults.thread_pool_per_block_device else None
            self._direct = os.open(self.path, (self.flags & ~os.O_TRUNC) | os.O_DIRECT | os.O_CLOEXEC, 0o644)
            descriptor = cufile.Descr()
            descriptor.type = cufile.FileHandleType.OPAQUE_FD
            descriptor.handle.fd = self._direct
            descriptor.fs_ops = 0
            self._handle = native("handle_register", descriptor.ptr)
        except (ImportError, OSError, RuntimeError) as exc:
            self.close()
            raise RuntimeError(f"Unable to open file {self.path}: {exc}") from exc

    def close(self):
        failure = None
        for batch in tuple(self._batches):
            try:
                batch.drain()
            except RuntimeError as exc:
                if batch.pending:
                    raise
                failure = failure or exc
        for future in tuple(self._pending):
            try:
                future.result()
            except Exception as exc:
                failure = failure or exc
        for work in tuple(self._streams):
            try:
                work.check_bytes_done()
            except RuntimeError as exc:
                if not work.finished:
                    raise
                failure = failure or exc
        if self._handle is not None:
            native("handle_deregister", self._handle)
            self._handle = None
        for name in ("_direct", "_fd"):
            fd = getattr(self, name, None)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)
        if failure is not None:
            raise failure

    def __del__(self):
        if not sys.is_finalizing():
            try:
                self.close()
            except (OSError, RuntimeError) as exc:
                warnings.warn(f"CuFile cleanup: {exc}", RuntimeWarning, stacklevel=2)

    def closed(self):
        return self._fd is None

    def fileno(self):
        return -1 if self.closed() else self._fd

    def open_flags(self):
        if self.closed():
            raise RuntimeError("File handle is closed")
        return fcntl.fcntl(self._fd, fcntl.F_GETFL)

    def is_direct_io_supported(self):
        return self._direct is not None

    def _check(self, write):
        if self.closed():
            raise RuntimeError("File handle is closed")
        access = self.flags & os.O_ACCMODE
        if (write and access == os.O_RDONLY) or (not write and access == os.O_WRONLY):
            raise RuntimeError("Operation not permitted")

    def _transfer(self, arr, size, file_offset, offset, write, context):
        with context:
            return self._transfer_native(arr, size, file_offset, offset, write)

    def _transfer_native(self, arr, size, file_offset, offset, write):
        try:
            # Registered allocation bases must remain stable across chunked views.
            from .buffer import registered_pointer

            base, relative = registered_pointer(arr.ptr + offset)
            return native("write" if write else "read", self._handle, base, size, file_offset, relative)
        except OSError as exc:
            raise RuntimeError(str(exc)) from exc

    def _transfer_observed(self, arr, size, file_offset, offset, write, context, finish):
        # Finish observation inside the task, before its future becomes ready.
        try:
            count = self._transfer(arr, size, file_offset, offset, write, context)
        except Exception as exc:
            finish(0, exc)
            raise
        finish(count, None)
        return count

    def _transfer_chunks(self, arr, size, file_offset, first, stride, task_size, write, context, finish):
        count, failure = 0, None
        for offset in range(first, size, stride):
            try:
                count += self._transfer(
                    arr, min(task_size, size - offset), file_offset + offset, offset, write, context
                )
            except Exception as exc:
                # Drain every scheduled chunk before reporting the first failure.
                if failure is None:
                    failure = exc
        if finish is not None:
            finish(count if failure is None else 0, failure)
        if failure is not None:
            raise failure
        return count

    def _parallel(self, buf, size, file_offset, task_size, write):
        from .statistics import observe

        arr, size = parse(buf, size, writable=not write)
        file_offset = _unsigned(file_offset)
        task_size = defaults.task_size() if not task_size else _unsigned(task_size)
        context = Context(arr)
        finish = observe(size, write, "gds")
        try:
            self._check(write)
            if arr.cuda and size:
                with context:
                    cuda(driver.cuStreamSynchronize(0))
            if size == 0:
                if finish is not None:
                    finish(0, None)
                return IOFuture()
            single_task = size <= task_size
            executor = pool(self._pool_device)
            chunk_count = (size + task_size - 1) // task_size
            jobs = min(chunk_count, defaults.thread_pool_nthreads())
            # Each job owns consecutive chunks, preserving locality and native sizes.
            futures = []
            try:
                if single_task:
                    transfer = self._transfer if finish is None else self._transfer_observed
                    extra = () if finish is None else (finish,)
                    futures.append(executor.submit(transfer, arr, size, file_offset, 0, write, context, *extra))
                else:
                    for index in range(jobs):
                        first = index * chunk_count // jobs * task_size
                        end = min(size, (index + 1) * chunk_count // jobs * task_size)
                        futures.append(
                            executor.submit(
                                self._transfer_chunks,
                                arr,
                                end,
                                file_offset,
                                first,
                                task_size,
                                task_size,
                                write,
                                context,
                                finish if jobs == 1 else None,
                            )
                        )
            except BaseException:
                wait(futures)
                raise
            result = IOFuture(futures, finish=None if jobs == 1 else finish)
            self._pending.add(result.future)
            result.future.add_done_callback(self._pending.discard)
            return result
        except (OSError, RuntimeError, ValueError) as exc:
            if finish is not None:
                finish(0, exc)
            raise

    def pread(self, buf, size, file_offset, task_size):
        return self._parallel(buf, size, file_offset, task_size, False)

    def pwrite(self, buf, size, file_offset, task_size):
        return self._parallel(buf, size, file_offset, task_size, True)

    def _raw(self, buf, size, file_offset, dev_offset, write):
        from .statistics import observe

        arr, size = parse(buf, size, accept_host=False, writable=not write, offset=dev_offset)
        finish = observe(size, write, "gds")
        try:
            self._check(write)
            if size:
                cuda(driver.cuStreamSynchronize(0))
            value = self._transfer_native(arr, size, _unsigned(file_offset), _unsigned(dev_offset), write)
        except (OSError, RuntimeError) as exc:
            if finish is not None:
                finish(0, exc)
            raise
        if finish is not None:
            finish(value, None)
        return value

    def read(self, buf, size, file_offset, dev_offset):
        return self._raw(buf, size, file_offset, dev_offset, False)

    def write(self, buf, size, file_offset, dev_offset):
        return self._raw(buf, size, file_offset, dev_offset, True)

    def _async(self, buf, size, file_offset, dev_offset, stream, write):
        arr, size = parse(buf, size, accept_host=False, writable=not write, offset=dev_offset)
        self._check(write)
        work = IOFutureStream(arr, size, _unsigned(file_offset), _unsigned(dev_offset), stream, self)
        self._streams.add(work)
        retain_stream(work)
        try:
            from .buffer import registered_pointer

            base, relative = registered_pointer(arr.ptr + dev_offset)
            work.dev_offset.value = relative
            native(
                "write_async" if write else "read_async",
                self._handle,
                base,
                ct.addressof(work.size),
                ct.addressof(work.file_offset),
                ct.addressof(work.dev_offset),
                ct.addressof(work.result),
                int(stream),
            )
        except (RuntimeError, OSError, ValueError) as exc:
            try:
                work.abort(exc)
            except RuntimeError as cleanup:
                exc.add_note(f"Stream drain failed; argument storage remains retained: {cleanup}")
            raise
        return work

    def read_async(self, buf, size, file_offset, dev_offset, st):
        return self._async(buf, size, file_offset, dev_offset, st, False)

    def write_async(self, buf, size, file_offset, dev_offset, st):
        return self._async(buf, size, file_offset, dev_offset, st, True)


def _fd(file):
    if isinstance(file, (os.PathLike, str)):
        try:
            return os.open(file, os.O_RDONLY), True
        except OSError as exc:
            raise RuntimeError(f"Unable to open file: {exc}") from exc
    if isinstance(file, int):
        return file, False
    if isinstance(file, io.IOBase):
        return file.fileno(), False
    raise ValueError("The type of file must be os.PathLike, str, int, or io.IOBase")


def get_page_cache_info(file, offset, length):
    fd, owned = _fd(file)
    try:
        offset, length = _unsigned(offset), _unsigned(length)
        size = os.fstat(fd).st_size
        if offset >= size:
            return 0, 0
        end = size if length == 0 else min(size, offset + length)
        start = offset // mmap.PAGESIZE * mmap.PAGESIZE
        count = (end - start + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        with mmap.mmap(fd, end - start, access=mmap.ACCESS_COPY, offset=start) as mapping:
            storage = (ct.c_char * (end - start)).from_buffer(mapping)
            vector = (ct.c_ubyte * count)()
            libc = ct.CDLL(None, use_errno=True)
            libc.mincore.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p]
            if libc.mincore(ct.addressof(storage), end - start, vector):
                raise OSError(ct.get_errno(), os.strerror(ct.get_errno()))
            del storage
            return sum(bool(v & 1) for v in vector), count
    finally:
        if owned:
            os.close(fd)


def drop_file_page_cache(file, offset, length, sync_first):
    fd, owned = _fd(file)
    try:
        if sync_first:
            os.fdatasync(fd)
        os.posix_fadvise(fd, _unsigned(offset), _unsigned(length), os.POSIX_FADV_DONTNEED)
    finally:
        if owned:
            os.close(fd)


def drop_system_page_cache(reclaim_dentries_and_inodes, sync_first):
    import subprocess

    if sync_first:
        os.sync()
    command = ["/sbin/sysctl", f"vm.drop_caches={3 if reclaim_dentries_and_inodes else 1}"]
    for prefix in ([], ["sudo", "-n"]):
        try:
            result = subprocess.run(prefix + command, capture_output=True, check=False)  # noqa: S603 - fixed sysctl command; no shell.
        except FileNotFoundError:
            continue
        if result.returncode == 0:
            return True
    return False

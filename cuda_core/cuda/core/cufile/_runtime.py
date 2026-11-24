# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared transfer adapters and cleanup for wrapper-owned cuFile resources."""

import atexit
import operator
import os
import sys
import threading
import warnings
import weakref

from cuda.core import CUDAError, CUDAWarning


class CuFileError(CUDAError):
    """A cuFile or filesystem failure with machine-readable error information.

    status/native_code identify a cuFile OpError, not a CUDA status or errno.
    status_name is the bindings' symbolic name, or None for an unknown status.
    Filesystem failures supply errno, filename, and strerror when available.
    Missing metadata is None. This exception derives from CUDAError, not OSError;
    original binding/filesystem exceptions remain available as __cause__.
    """

    def __init__(
        self,
        message: str,
        native_code: int | None = None,
        *,
        cuda_error: int | None = None,
        errno: int | None = None,
        filename: str | None = None,
        strerror: str | None = None,
        operation: str | None = None,
        native_result: int | None = None,
    ) -> None:
        super().__init__(message)
        self.native_code = None if native_code is None else operator.index(native_code)
        self.cuda_error = cuda_error
        self.operation = operation
        self.native_result = native_result
        self.errno = None if errno is None else operator.index(errno)
        self.filename = filename
        self.strerror = strerror

    @property
    def status(self) -> int | None:
        """Native cuFile status; an alias of native_code, matching bindings naming."""
        return self.native_code

    @property
    def status_name(self) -> str | None:
        """Bindings OpError name, or None when no known cuFile status is available."""
        if self.native_code is None:
            return None
        try:
            return lib().OpError(self.native_code).name
        except ValueError:
            return None


def integer(value: object, name: str, *, minimum: int = 0, maximum: int = (1 << 63) - 1) -> int:
    result = operator.index(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def lib():
    from cuda.bindings import cufile

    return cufile


def _wait_for_work(work):
    wait_all = getattr(work, "wait_all", None)
    return work.wait() if wait_all is None else wait_all()


def call(name, *args):
    backend = lib()
    try:
        function = getattr(backend, name) if isinstance(name, str) else name
        return function(*args)
    except backend.cuFileError as exc:
        error = CuFileError(f"cuFile {name}: {exc}", int(exc.status), cuda_error=exc.cuda_error, operation=name)
        raise error from exc


def transfer_error(value: int, *, operation: str | None = None) -> CuFileError:
    if value == -1:
        return CuFileError(
            f"cuFile {operation or 'transfer'} failed; the deferred result does not provide errno",
            operation=operation,
            native_result=value,
        )
    error = CuFileError(
        f"cuFile {operation or 'transfer'} failed with result {value}",
        -value,
        operation=operation,
        native_result=value,
    )
    if error.status_name is not None:
        error.args = (f"{error.args[0]} ({error.status_name})",)
    return error


def check_count(value: int, *, operation: str | None = None) -> int:
    if value < 0:
        raise transfer_error(value, operation=operation)
    return value


def warn_cleanup(exc: Exception) -> None:
    warnings.warn(f"cuFile cleanup failed: {exc}", CUDAWarning, stacklevel=2)


class _Runtime:
    def __init__(self):
        self.pid = os.getpid()
        self.lock = threading.RLock()
        self.stopping = False
        self.native_io = None
        self.resources = weakref.WeakSet()
        self.pending = set()
        self.quarantined = []
        atexit.register(self.shutdown)

    def check(self, *, allow_shutdown=False):
        if self.pid != os.getpid():
            raise RuntimeError("cuFile cannot be used in a forked child; use a spawned process")
        if self.stopping and not allow_shutdown:
            raise RuntimeError("cuFile runtime is shutting down")

    def get_native_io(self):
        """Resolve transfer functions without opening or owning the cuFile driver."""
        self.check()
        with self.lock:
            if self.native_io is None:
                from cuda.core._cufile_native import NativeTransfers

                self.native_io = NativeTransfers()
            return self.native_io

    def quarantine(self, obj):
        # An intentionally leaked Python reference also survives module teardown.
        from cuda.core._cufile_native import pin_forever

        with self.lock:
            if any(resource is obj for resource in self.quarantined):
                return
            pin_forever(obj)
            self.quarantined.append(obj)

    def track(self, resource):
        with self.lock:
            self.resources.add(resource)

    def add_work(self, work):
        from cuda.core._cufile_native import pin_forever

        with self.lock:
            self.pending.add(work)
            pin_forever(work)

    def remove_work(self, work):
        from cuda.core._cufile_native import unpin

        with self.lock:
            present = work in self.pending
            self.pending.discard(work)
        if present:
            unpin(work)

    def shutdown(self):
        if self.pid != os.getpid() or self.stopping:
            return
        self.stopping = True
        from ._driver import Driver

        with self.lock:
            pending, resources = tuple(self.pending), tuple(self.resources)
        for work in pending:
            try:
                _wait_for_work(work)
            except Exception as exc:
                warn_cleanup(exc)
        for resource in resources:
            if isinstance(resource, Driver):
                continue
            try:
                resource.close()
            except Exception as exc:
                warn_cleanup(exc)
        if not self.quarantined:
            # Explicit Driver objects close after their dependent resources.
            for resource in resources:
                if not isinstance(resource, Driver):
                    continue
                try:
                    resource.close()
                except Exception as exc:
                    warn_cleanup(exc)


runtime = _Runtime()


class _ScopedResource:
    def __enter__(self):
        if self.is_closed:
            raise RuntimeError("Resource has been closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as cleanup:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(f"cuFile cleanup also failed: {cleanup}")
            else:
                warn_cleanup(cleanup)
        return False

    def __del__(self):
        if not sys.is_finalizing():
            try:
                self.close()
            except Exception as exc:
                warn_cleanup(exc)

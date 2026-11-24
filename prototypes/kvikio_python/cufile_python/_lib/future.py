# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ctypes as ct
import threading
from concurrent.futures import Future

from ._runtime import cuda, driver, release_stream


class IOFuture:
    def __init__(self, futures=(), *, finish=None, value=0):
        self._used = False
        if len(futures) == 1 and finish is None:
            self.future = futures[0]
            return
        self.future = Future()
        self._lock = threading.Lock()
        self._remaining = len(futures)
        self._total, self._error = 0, None
        self._finish = finish
        if not futures:
            self.future.set_result(value)
        for item in futures:
            item.add_done_callback(self._complete)

    def _complete(self, item):
        with self._lock:
            try:
                self._total += item.result()
            except Exception as exc:
                if self._error is None:
                    self._error = exc
            self._remaining -= 1
            if self._remaining:
                return
            if self._finish is not None:
                self._finish(self._total, self._error)
            if self._error is None:
                self.future.set_result(self._total)
            else:
                self.future.set_exception(self._error)
            self._finish = None

    def get(self):
        if self._used:
            raise RuntimeError("std::future_error: No associated state")
        self._used = True
        return self.future.result()

    def done(self):
        if self._used:
            raise RuntimeError("std::future_error: No associated state")
        return self.future.done()


class IOFutureStream:
    def __init__(self, arr, size, file_offset, dev_offset, stream, owner):
        self.arr, self.owner = arr, owner
        self.stream = stream
        self.size = ct.c_size_t(size)
        self.file_offset = ct.c_int64(file_offset)
        self.dev_offset = ct.c_int64(dev_offset)
        self.result = ct.c_ssize_t(-1)
        self.finished = False
        self._error = None

    def abort(self, error):
        # A submission error must not free argument storage before stream work ends.
        cuda(driver.cuStreamSynchronize(self.stream))
        self._complete(error)

    def _complete(self, error=None):
        self._error = error
        self.finished = True
        self.owner._streams.discard(self)
        self.arr = self.owner = None
        release_stream(self)

    def check_bytes_done(self):
        if not self.finished:
            cuda(driver.cuStreamSynchronize(self.stream))
            value = self.result.value
            self._complete(RuntimeError(f"cuFile async I/O failed: {value}") if value < 0 else None)
        if self._error is not None:
            raise self._error
        return self.result.value

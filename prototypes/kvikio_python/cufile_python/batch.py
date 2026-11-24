# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native cuFile batches, extending the KvikIO-compatible Python API."""

from dataclasses import dataclass

from cuda.bindings.cufile import Opcode, Status

from ._lib.batch import BatchHandle
from .cufile import CuFile

__all__ = ["Batch", "BatchOp", "BatchCompletion", "Opcode", "Status"]


@dataclass(frozen=True, slots=True)
class BatchOp:
    """One transfer, with the same byte-count/offset conventions as CuFile.

    file, buf, and opcode are required. opcode is "read", "write", or a native
    Opcode value. size=None means the entire contiguous Python buffer view;
    specify size when dev_offset is nonzero. Both offsets default to zero.
    dev_offset also applies to host buffers. cookie is arbitrary user metadata.
    Reuse an entry after completion, or use dataclasses.replace to change it.
    """

    file: CuFile
    buf: object
    opcode: Opcode | str
    size: int | None = None
    file_offset: int = 0
    dev_offset: int = 0
    cookie: object = None


@dataclass(frozen=True, slots=True)
class BatchCompletion:
    """A copied completion whose fields remain valid after Batch.close().

    index identifies the entry within its submission; cookie is the BatchOp's
    user value. status is cuda.bindings.cufile.Status, without invented states.
    native_result preserves the signed native byte count or error result.
    get() returns the count or raises RuntimeError, and can be called again.
    """

    index: int
    cookie: object
    status: Status
    native_result: int

    def get(self) -> int:
        if self.status != Status.COMPLETE or self.native_result < 0:
            error = RuntimeError(
                f"cuFile batch entry {self.index}: {self.status.name}, native result {self.native_result}"
            )
            error.status = self.status
            error.native_result = self.native_result
            raise error
        return self.native_result


class Batch:
    """A reusable native batch queue with one outstanding submission at a time.

    capacity is the maximum entries per submission. Construction sets up the
    native queue. Open a CuFile or the native driver first. The queue accepts
    different files and contiguous host/device buffers; callers provide the
    appropriate current CUDA context and finish producer GPU work before submit.
    No worker pool, implicit driver reference, or context switching is added.

    poll() returns newly collected completions without waiting. wait_all()
    waits for remaining completions, ordered by index, and never repeats a poll
    result. Per-entry errors are raised by completion.get(). A timeout or native
    collection error exposes already collected results in error.completions.
    On libcufile 1.19, batches containing host buffers require wait_all() without
    a timeout: native nonblocking collection can free host state prematurely.

    close() drains remaining work, reports otherwise unobserved entry errors,
    and destroys the queue. It is idempotent after success. File.close() also
    drains a batch using that file; collect results before either close to keep
    them. Concurrent operations on one Batch, and nesting with on the same
    object, are rejected. Native submission/status failures retain owners when
    completion cannot be established; the affected queue cannot be reused.
    """

    def __init__(self, capacity: int):
        self._handle = BatchHandle(capacity)
        self._entered = False

    @property
    def capacity(self) -> int:
        return self._handle.capacity

    @property
    def closed(self) -> bool:
        return self._handle.closed

    @property
    def pending(self) -> int:
        """Number of submitted entries whose completion has not been collected."""
        return self._handle.pending

    def submit(self, operations) -> None:
        """Validate and pack BatchOp entries, then submit them without waiting.

        Collect all previous completions before resubmitting. One batch's
        entries may execute in any order; do not submit dependent or overlapping
        destination transfers together. Keep external registrations valid until
        completion. Entries, buffers, and file handles are retained internally.
        """
        self._handle.submit(operations)

    def poll(self, max_complete: int | None = None) -> tuple[BatchCompletion, ...]:
        """Collect ready results; host-buffer polling is disabled on libcufile 1.19."""
        return self._handle.poll(max_complete)

    def wait_all(self, timeout: float | None = None) -> tuple[BatchCompletion, ...]:
        """Collect remaining results; timeout is a host-side deadline in seconds."""
        return self._handle.wait_all(timeout)

    def cancel(self) -> None:
        """Request cancellation, then collect terminal results normally.

        Cancellation is best effort. Active cancellation on libcufile 1.19 is
        rejected: that version can discard unreaped completions. Use wait_all().
        """
        self._handle.cancel()

    def close(self, *, timeout: float | None = None) -> None:
        self._handle.close(timeout=timeout)

    def __enter__(self):
        self._handle.check()
        if self._entered:
            raise RuntimeError("Cannot nest with blocks on the same Batch")
        self._entered = True
        return self

    def __exit__(self, kind, error, traceback):
        try:
            self.close()
        except (RuntimeError, OSError) as cleanup:
            if error is None:
                raise
            error.add_note(f"Batch cleanup also failed: {cleanup}")
        finally:
            self._entered = False

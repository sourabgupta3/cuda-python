# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ctypes as ct
import gc
import threading
import weakref
from collections import deque
from dataclasses import replace

import numpy as np
import pytest

import cufile_python as api
from cuda.bindings import cufile as native
from cufile_python._lib import _runtime
from cufile_python._lib import batch as backend
from cufile_python.batch import Opcode, Status

BLOCK = 4096


class ControlledBatch:
    """Deterministic native boundary for completion/failure/lifetime tests."""

    def __init__(self, monkeypatch, version=1200):
        self.ready = deque()
        self.submissions = []
        self.calls = []
        self.version = version
        self.fail = None
        self.return_count = None
        self.entered = self.release = None
        monkeypatch.setattr(backend, "native", self)

    def __call__(self, name, *args):
        self.calls.append(name)
        if self.fail == name:
            raise RuntimeError(f"injected {name} failure")
        if name == "get_version":
            return self.version
        if name == "batch_io_set_up":
            return 123
        if name == "batch_io_submit":
            _, count, ptr, _ = args
            params = native.IOParams.from_ptr(ptr, size=count)
            self.submissions.append(
                [
                    (
                        int(params[i].fh),
                        int(params[i].opcode),
                        int(params[i].cookie),
                        params[i].u.batch.dev_ptr_base,
                        params[i].u.batch.dev_ptr_offset,
                        params[i].u.batch.file_offset,
                        params[i].u.batch.size_,
                    )
                    for i in range(count)
                ]
            )
        elif name == "batch_io_get_status":
            if self.entered is not None:
                self.entered.set()
                assert self.release.wait(5)
            _, _, count_ptr, events_ptr, _ = args
            count = ct.c_uint.from_address(count_ptr)
            events = native.IOEvents.from_ptr(events_ptr, size=count.value)
            returned = min(count.value, len(self.ready))
            for i in range(returned):
                token, status, value = self.ready.popleft()
                events[i].cookie = token
                events[i].status = status
                events[i].ret = ct.c_size_t(value).value
            count.value = returned if self.return_count is None else self.return_count
        elif name == "batch_io_cancel":
            pass
        elif name != "batch_io_destroy":
            raise AssertionError(name)

    def complete(self, *indices, result=BLOCK, status=Status.COMPLETE):
        self.ready.extend((index + 1, status, result) for index in indices)


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_native_multifile_mixed_buffers_offsets_reuse(tmp_path, registered):
    cp = pytest.importorskip("cupy")
    expected = np.arange(BLOCK, dtype=np.uint8)
    # Pool suballocations can share one CUDA allocation. Use independent native
    # allocations when testing registration of two distinct buffers.
    source = cp.ndarray(BLOCK * 3, dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(cp.cuda.Memory(BLOCK * 3), 0))
    target = cp.ndarray(BLOCK * 3, dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(cp.cuda.Memory(BLOCK * 3), 0))
    source.set(np.tile(expected, 3))
    target.fill(99)
    host = bytearray(b"!" * (BLOCK * 3))
    cp.cuda.get_current_stream().synchronize()
    registrations = []
    try:
        if registered:
            for buf in (source, target):
                api.memory_register(buf)
                registrations.append(buf)
        a, b = tmp_path / "a", tmp_path / "b"
        with api.CuFile(a, "w+") as fa, api.CuFile(b, "w+") as fb, api.Batch(2) as batch:
            ops = [
                api.BatchOp(fa, source, Opcode.WRITE, BLOCK, file_offset=BLOCK, dev_offset=BLOCK, cookie="gpu"),
                api.BatchOp(fb, expected, "write", cookie="host"),
            ]
            batch.submit(ops)
            written = batch.wait_all()
            assert [r.get() for r in written] == [BLOCK, BLOCK]
            assert all(r.status is Status.COMPLETE for r in written)
            ops = [
                api.BatchOp(fa, target, "read", BLOCK, file_offset=BLOCK, dev_offset=BLOCK, cookie=3),
                api.BatchOp(fb, host, "read", BLOCK, dev_offset=BLOCK, cookie={"host": True}),
            ]
            batch.submit(ops)
            if native.get_version() == 1190:
                with pytest.raises(NotImplementedError, match="Host-buffer"):
                    batch.poll(max_complete=1)
                with pytest.raises(NotImplementedError, match="Host-buffer"):
                    batch.wait_all(timeout=10)
            results = batch.wait_all()
            assert sorted(r.index for r in results) == [0, 1]
            assert [r.get() for r in results] == [BLOCK, BLOCK]
            assert batch.pending == 0
            assert batch.poll() == batch.wait_all() == ()
            ops[0] = replace(ops[0], dev_offset=0)
            batch.submit(ops[:1])
            first = batch.poll(max_complete=1)
            rest = batch.wait_all(timeout=10)
            assert len(first + rest) == 1
            assert (first + rest)[0].get() == BLOCK
        assert batch.closed
        assert fa.closed
        assert fb.closed
        assert written[0].get() == BLOCK
        assert host[:BLOCK] == host[-BLOCK:] == b"!" * BLOCK
        assert host[BLOCK : 2 * BLOCK] == expected.tobytes()
        np.testing.assert_array_equal(cp.asnumpy(target[: 2 * BLOCK]), np.tile(expected, 2))
        assert bool(cp.all(target[2 * BLOCK :] == 99))
        assert a.read_bytes() == bytes(BLOCK) + expected.tobytes()
        assert b.read_bytes() == expected.tobytes()
    finally:
        for buf in reversed(registrations):
            api.memory_deregister(buf)


@pytest.mark.parametrize("close_file", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_native_close_drains_work_without_context_switch(tmp_path, monkeypatch, close_file):
    cp = pytest.importorskip("cupy")
    path = tmp_path / "data"
    path.write_bytes(bytes(range(256)) * 16)
    data = cp.zeros(BLOCK, dtype=cp.uint8)
    cp.cuda.get_current_stream().synchronize()
    current = _runtime.cuda(_runtime.driver.cuCtxGetCurrent())

    def forbidden(*_args):
        raise AssertionError("Batch must use caller context without Python lookup/push/pop")

    file = api.CuFile(path)
    batch = api.Batch(1)
    for name in ("cuPointerGetAttribute", "cuCtxPushCurrent", "cuCtxPopCurrent"):
        monkeypatch.setattr(_runtime.driver, name, forbidden)
    try:
        batch.submit([api.BatchOp(file, data, "read")])
        if close_file:
            file.close()
            assert not batch.closed
            assert batch.wait_all() == ()
        else:
            batch.close()
            assert not file.closed
        np.testing.assert_array_equal(cp.asnumpy(data), np.arange(BLOCK, dtype=np.uint8))
        assert _runtime.cuda(_runtime.driver.cuCtxGetCurrent()) == current
    finally:
        batch.close()
        file.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_poll_timeout_results_are_not_replayed_and_release_exports(tmp_path, monkeypatch):
    driver = ControlledBatch(monkeypatch)
    first, second = bytearray(BLOCK), bytearray(BLOCK)
    with api.CuFile(tmp_path / "data", "w+") as file, api.Batch(2) as batch:
        batch.submit([api.BatchOp(file, buf, "write", cookie=i) for i, buf in enumerate((first, second))])
        with pytest.raises(BufferError):
            first.extend(b"x")
        driver.complete(1)
        with pytest.raises(TimeoutError) as caught:
            batch.wait_all(timeout=0)
        assert [r.cookie for r in caught.value.completions] == [1]
        assert caught.value.completions[0].get() == BLOCK
        second.extend(b"x")
        assert batch.pending == 1
        with pytest.raises(TimeoutError):
            batch.close(timeout=0)
        assert not batch.closed
        assert not file.closed
        driver.complete(0)
        assert [r.cookie for r in batch.poll()] == [0]
        first.extend(b"x")
        assert batch.wait_all() == ()
        batch.submit([api.BatchOp(file, first, "write", BLOCK)])
        driver.complete(0)
    assert driver.calls.count("batch_io_set_up") == driver.calls.count("batch_io_destroy") == 1
    assert not _runtime._pending_batches


@pytest.mark.parametrize(
    ("status", "value"),
    [
        (Status.FAILED, -5),
        (Status.CANCELED, 0),
        (Status.TIMEOUT, 0),
        (Status.INVALID, 0),
        (Status.COMPLETE, -5008),
        (Status.COMPLETE, 73),
    ],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_entry_status_and_error_delivery(tmp_path, monkeypatch, status, value):
    driver = ControlledBatch(monkeypatch)
    with api.CuFile(tmp_path / "data", "w+") as file, api.Batch(1) as batch:
        batch.submit([api.BatchOp(file, bytes(BLOCK), "write", cookie="tag")])
        driver.complete(0, status=status, result=value)
        (result,) = batch.wait_all()
        assert result.status is status
        assert result.native_result == value
        assert result.cookie == "tag"
        if status is Status.COMPLETE and value >= 0:
            assert result.get() == value
        else:
            with pytest.raises(RuntimeError) as caught:
                result.get()
            assert caught.value.status is status
            assert caught.value.native_result == value
        assert batch.pending == 0


@pytest.mark.parametrize("close_file", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_close_reports_uncollected_entry_error_after_releasing_resources(tmp_path, monkeypatch, close_file):
    driver = ControlledBatch(monkeypatch)
    file, batch = api.CuFile(tmp_path / "data", "w+"), api.Batch(1)
    batch.submit([api.BatchOp(file, bytes(BLOCK), "write")])
    driver.complete(0, status=Status.FAILED, result=-5)
    try:
        with pytest.raises(RuntimeError) as caught:
            (file.close if close_file else batch.close)()
        assert caught.value.completions[0].native_result == -5
        assert file.closed if close_file else batch.closed
        assert batch.pending == 0
    finally:
        batch.close()
        file.close()


@pytest.mark.parametrize("failure", ["batch_io_submit", "batch_io_get_status"])
@pytest.mark.agent_authored(model="gpt-6")
def test_native_failure_keeps_owners_and_blocks_file_close(tmp_path, monkeypatch, failure):
    driver = ControlledBatch(monkeypatch)
    file, batch = api.CuFile(tmp_path / "data", "w+"), api.Batch(1)
    buf = bytearray(BLOCK)
    try:
        if failure == "batch_io_submit":
            driver.fail = failure

            def action():
                batch.submit([api.BatchOp(file, buf, "write")])
        else:
            batch.submit([api.BatchOp(file, buf, "write")])
            driver.fail = failure
            action = batch.poll
        with pytest.raises(RuntimeError, match="injected"):
            action()
        assert batch.pending == 1
        with pytest.raises(BufferError):
            buf.extend(b"x")
        with pytest.raises(RuntimeError, match="uncertain"):
            file.close()
        assert not file.closed
        assert "batch_io_destroy" not in driver.calls
    finally:
        # This test's fake native backend never performed I/O. Release its owners
        # explicitly; production code must not guess completion after a failure.
        batch._handle._pending.clear()
        batch._handle._retire()
        batch._handle._broken = False
        driver.fail = None
        batch.close()
        file.close()


@pytest.mark.parametrize("bad", ["cookie", "duplicate", "status", "count", "size"])
@pytest.mark.agent_authored(model="gpt-6")
def test_corrupt_completion_is_rejected_without_freeing_owners(tmp_path, monkeypatch, bad):
    driver = ControlledBatch(monkeypatch)
    file, batch = api.CuFile(tmp_path / "data", "w+"), api.Batch(2)
    try:
        batch.submit([api.BatchOp(file, bytearray(BLOCK), "write") for _ in range(2)])
        if bad == "cookie":
            driver.complete(99)
        elif bad == "duplicate":
            driver.complete(0, 0)
        elif bad == "status":
            driver.complete(0, status=Status.PENDING)
        elif bad == "count":
            driver.return_count = 3
        else:
            driver.complete(0, result=BLOCK + 1)
        with pytest.raises(RuntimeError, match=r"Native|native"):
            batch.poll()
        assert batch.pending == 2
    finally:
        batch._handle._pending.clear()
        batch._handle._retire()
        batch._handle._broken = False
        batch.close()
        file.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_validation_and_one_outstanding_submission(tmp_path, monkeypatch):
    driver = ControlledBatch(monkeypatch)
    with api.CuFile(tmp_path / "data", "w+") as file, api.Batch(2) as batch:
        invalid = [
            [],
            [object()],
            [api.BatchOp(file, bytes(BLOCK), "bad")],
            [api.BatchOp(file, bytes(BLOCK), "read")],
            [api.BatchOp(file, bytearray(BLOCK), "read", BLOCK + 1)],
            [api.BatchOp(file, bytearray(BLOCK), "read", 0)],
            [api.BatchOp(file, memoryview(bytearray(BLOCK))[::2], "read")],
            [api.BatchOp(file, bytes(BLOCK), "write", file_offset=-1)],
            [api.BatchOp(file, bytes(BLOCK), "write", file_offset=2**63)],
            [api.BatchOp(file, bytes(BLOCK), "write")] * 3,
        ]
        for operations in invalid:
            with pytest.raises((TypeError, ValueError)):
                batch.submit(operations)
        assert not driver.submissions
        assert not file._handle._batches
        buf = bytearray(BLOCK)
        with pytest.raises(ValueError, match="overlap"):
            batch.submit([api.BatchOp(file, buf, "read"), api.BatchOp(file, buf, "write")])
        batch.submit([api.BatchOp(file, bytes(BLOCK), "write")])
        with pytest.raises(RuntimeError, match="Collect all"):
            batch.submit([api.BatchOp(file, bytes(BLOCK), "write")])
        driver.complete(0)
        batch.wait_all()
    for operation in (batch.submit, batch.poll, batch.wait_all):
        with pytest.raises(RuntimeError, match="closed"):
            operation([]) if operation == batch.submit else operation()
    batch.close()


@pytest.mark.parametrize("capacity", [0, -1, 2**32, 1.5])
@pytest.mark.agent_authored(model="gpt-6")
def test_invalid_capacity_never_sets_up_native_queue(monkeypatch, capacity):
    driver = ControlledBatch(monkeypatch)
    with pytest.raises((ValueError, TypeError)):
        api.Batch(capacity)
    assert not driver.calls


@pytest.mark.agent_authored(model="gpt-6")
def test_dropped_batch_and_buffers_are_retained_until_file_close(tmp_path, monkeypatch):
    driver = ControlledBatch(monkeypatch)
    file = api.CuFile(tmp_path / "data", "w+")
    batch = api.Batch(1)
    buf = np.zeros(BLOCK, dtype=np.uint8)
    owner, memory = weakref.ref(batch._handle), weakref.ref(buf)
    batch.submit([api.BatchOp(file, buf, "write")])
    del batch, buf
    gc.collect()
    assert owner() is not None
    assert memory() is not None
    driver.complete(0)
    file.close()
    gc.collect()
    assert memory() is None
    assert owner() is None
    assert driver.calls.count("batch_io_destroy") == 1


@pytest.mark.agent_authored(model="gpt-6")
def test_scopes_close_on_exception_and_reject_nested_entry(tmp_path, monkeypatch):
    driver = ControlledBatch(monkeypatch)
    with api.CuFile(tmp_path / "data", "w+") as file:
        batch = api.Batch(1)

        def body():
            with batch:
                with pytest.raises(RuntimeError, match="nest"), batch:
                    pass
                assert not batch.closed
                batch.submit([api.BatchOp(file, bytes(BLOCK), "write")])
                driver.complete(0)
                raise ValueError("body")

        with pytest.raises(ValueError, match="body"):
            body()
        assert batch.closed
        with pytest.raises(RuntimeError, match="closed"), batch:
            pass


@pytest.mark.parametrize("version", [1190, 1200])
@pytest.mark.agent_authored(model="gpt-6")
def test_cancel_native_contract_and_version_guard(tmp_path, monkeypatch, version):
    driver = ControlledBatch(monkeypatch, version)
    with api.CuFile(tmp_path / "data", "w+") as file, api.Batch(1) as batch:
        batch.cancel()
        assert "batch_io_cancel" not in driver.calls
        batch.submit([api.BatchOp(file, bytes(BLOCK), "write")])
        if version == 1190:
            with pytest.raises(NotImplementedError, match=r"1\.19"):
                batch.cancel()
            assert "batch_io_cancel" not in driver.calls
            driver.complete(0)
        else:
            batch.cancel()
            assert batch.pending == 1
            driver.complete(0, status=Status.CANCELED, result=0)
        batch.wait_all()


@pytest.mark.agent_authored(model="gpt-6")
def test_host_polling_guard_precedes_native_collection(tmp_path, monkeypatch):
    driver = ControlledBatch(monkeypatch, version=1190)
    with api.CuFile(tmp_path / "data", "w+") as file, api.Batch(1) as batch:
        batch.submit([api.BatchOp(file, bytes(BLOCK), "write")])
        for call in (batch.poll, lambda: batch.wait_all(1), lambda: batch.close(timeout=1)):
            with pytest.raises(NotImplementedError, match="Host-buffer"):
                call()
            assert "batch_io_get_status" not in driver.calls
            assert batch.pending == 1
            assert not batch.closed
        driver.complete(0)
        assert batch.wait_all()[0].get() == BLOCK


@pytest.mark.agent_authored(model="gpt-6")
def test_concurrent_batch_mutation_is_rejected(tmp_path, monkeypatch):
    driver = ControlledBatch(monkeypatch)
    driver.entered, driver.release = threading.Event(), threading.Event()
    with api.CuFile(tmp_path / "data", "w+") as file, api.Batch(1) as batch:
        batch.submit([api.BatchOp(file, bytes(BLOCK), "write")])
        driver.complete(0)
        results = []
        worker = threading.Thread(target=lambda: results.extend(batch.poll()))
        worker.start()
        try:
            assert driver.entered.wait(5)
            with pytest.raises(RuntimeError, match="Concurrent"):
                batch.close()
        finally:
            driver.release.set()
            worker.join(5)
        assert not worker.is_alive()
        assert results[0].get() == BLOCK


@pytest.mark.agent_authored(model="gpt-6")
def test_native_eof_returns_zero_and_file_access_is_validated(tmp_path):
    pytest.importorskip("cupy").cuda.runtime.free(0)
    path = tmp_path / "data"
    path.write_bytes(bytes(BLOCK))
    with api.CuFile(path) as file, api.Batch(1) as batch:
        with pytest.raises(RuntimeError, match="not permitted"):
            batch.submit([api.BatchOp(file, bytes(BLOCK), "write")])
        batch.submit([api.BatchOp(file, bytearray(BLOCK), "read", file_offset=BLOCK)])
        (result,) = batch.wait_all()
        assert result.get() == 0
        assert result.status is Status.COMPLETE


@pytest.mark.agent_authored(model="gpt-6")
def test_setup_failure_preserves_bindings_exception(monkeypatch):
    error = native.cuFileError(5022)

    def failure(_capacity):
        raise error

    monkeypatch.setattr(native, "batch_io_set_up", failure)
    with pytest.raises(RuntimeError) as caught:
        api.Batch(1)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf")])
@pytest.mark.agent_authored(model="gpt-6")
def test_bad_timeouts_rejected_without_native_poll(monkeypatch, timeout):
    driver = ControlledBatch(monkeypatch)
    with api.Batch(1) as batch:
        with pytest.raises(ValueError, match="timeout"):
            batch.wait_all(timeout)
        assert "batch_io_get_status" not in driver.calls

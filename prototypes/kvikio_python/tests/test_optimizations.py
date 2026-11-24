# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import threading

import numpy as np
import pytest

import cufile_python as kvikio
import cufile_python.defaults
from cufile_python._lib import _runtime, file_handle
from cufile_python._lib import statistics as stats


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_single_task_observation_finishes_before_result(tmp_path, monkeypatch, fail):
    path = tmp_path / "data"
    path.write_bytes(b"x" * 4096)
    entered, release = threading.Event(), threading.Event()
    original_observe = stats.observe

    def observe(*args):
        record = original_observe(*args)
        assert record is not None

        def finish(count, error):
            entered.set()
            assert release.wait(10)
            record(count, error)

        return finish

    if fail:

        def failed_transfer(*_args):
            raise RuntimeError("injected I/O failure")

        monkeypatch.setattr(file_handle.CuFile, "_transfer", failed_transfer)
    monkeypatch.setattr(stats, "observe", observe)
    with kvikio.SummaryMonitor() as monitor, kvikio.CuFile(path) as file:
        pending = file.pread(bytearray(4096))
        try:
            assert entered.wait(10)
            assert not pending.done()
        finally:
            release.set()
        if fail:
            with pytest.raises(RuntimeError, match="injected I/O failure"):
                pending.get()
        else:
            assert pending.get() == 4096
        summary = monitor.get()
        assert summary.num_reads == 1
        assert summary.num_errors == int(fail)
        assert summary.bytes_transferred == (0 if fail else 4096)
        with pytest.raises(RuntimeError, match="No associated state"):
            pending.get()


@pytest.mark.agent_authored(model="gpt-6")
def test_monitor_does_not_count_work_submitted_before_it_started(tmp_path, monkeypatch):
    path = tmp_path / "data"
    path.write_bytes(b"x" * 4096)
    entered, release = threading.Event(), threading.Event()
    original = file_handle.CuFile._transfer

    def delayed(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)

    monkeypatch.setattr(file_handle.CuFile, "_transfer", delayed)
    with kvikio.CuFile(path) as file:
        before = file.pread(bytearray(4096))
        try:
            assert entered.wait(10)
            with kvikio.SummaryMonitor() as monitor:
                release.set()
                assert before.get() == 4096
                assert monitor.get().num_ops == 0
                assert file.pread(bytearray(4096)).get() == 4096
                assert monitor.get().num_ops == 1
        finally:
            release.set()


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_shared_context_restores_each_worker_after_parallel_io(tmp_path, monkeypatch, fail):
    cp = pytest.importorskip("cupy")
    expected = np.arange(16384, dtype=np.uint8)
    path = tmp_path / "data"
    path.write_bytes(expected.tobytes())
    destination = cp.empty(expected.size, dtype=cp.uint8)
    cp.cuda.get_current_stream().synchronize()
    caller_context = _runtime.cuda(_runtime.driver.cuCtxGetCurrent())
    barrier = threading.Barrier(4)
    restored = set()
    lock = threading.Lock()
    original_transfer = file_handle.CuFile._transfer
    original_native = file_handle.native

    def native(name, *args):
        if fail and name == "read" and args[3] == 4096:
            raise RuntimeError("injected native failure")
        return original_native(name, *args)

    def transfer(*args):
        previous = _runtime.cuda(_runtime.driver.cuCtxGetCurrent())
        barrier.wait(10)
        try:
            return original_transfer(*args)
        finally:
            assert _runtime.cuda(_runtime.driver.cuCtxGetCurrent()) == previous
            with lock:
                restored.add(threading.get_ident())

    monkeypatch.setattr(file_handle, "native", native)
    monkeypatch.setattr(file_handle.CuFile, "_transfer", transfer)
    with (
        kvikio.defaults.set({"compat_mode": kvikio.CompatMode.OFF, "num_threads": 4, "gds_threshold": 0}),
        kvikio.CuFile(path) as file,
    ):
        pending = file.pread(destination, task_size=4096)
        if fail:
            with pytest.raises(RuntimeError, match="injected native failure"):
                pending.get()
        else:
            assert pending.get() == expected.size
            np.testing.assert_array_equal(cp.asnumpy(destination), expected)
        assert len(restored) == 4
        assert _runtime.cuda(_runtime.driver.cuCtxGetCurrent()) == caller_context

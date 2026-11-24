# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import gc
import os
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path

import numpy as np
import pytest

import cufile_python as kvikio
import cufile_python.defaults
from cufile_python._lib import _runtime, file_handle


def _require_caller_context(monkeypatch):
    def unexpected(*_args):
        raise AssertionError("Raw I/O must not resolve or switch the caller's CUDA context")

    monkeypatch.setattr(file_handle, "Context", unexpected)
    for name in ("cuPointerGetAttribute", "cuCtxPushCurrent", "cuCtxPopCurrent"):
        monkeypatch.setattr(_runtime.driver, name, unexpected)


@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_raw_io_uses_caller_context_on_success_and_failure(tmp_path, monkeypatch, write, fail):
    cp = pytest.importorskip("cupy")
    expected = np.arange(4096, dtype=np.uint8)
    path = tmp_path / "raw"
    original_contents = b"!" * 4096 if write else expected.tobytes()
    path.write_bytes(original_contents)
    buf = cp.asarray(expected) if write else cp.zeros(4096, dtype=cp.uint8)
    cp.cuda.get_current_stream().synchronize()
    current = _runtime.cuda(_runtime.driver.cuCtxGetCurrent())
    original = file_handle.native

    def native(name, *args):
        if fail and name in ("read", "write"):
            raise RuntimeError("injected native failure")
        return original(name, *args)

    with kvikio.CuFile(path, "r+") as file:
        _require_caller_context(monkeypatch)
        monkeypatch.setattr(file_handle, "native", native)
        operation = file.raw_write if write else file.raw_read
        if fail:
            with pytest.raises(RuntimeError, match="injected native failure"):
                operation(buf)
        else:
            assert operation(buf) == 4096
            np.testing.assert_array_equal(cp.asnumpy(buf), expected)
        assert _runtime.cuda(_runtime.driver.cuCtxGetCurrent()) == current
    assert path.read_bytes() == (original_contents if fail else expected.tobytes())


@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("finish", ["result", "file_close"])
@pytest.mark.agent_authored(model="gpt-6")
def test_stream_io_and_completion_use_caller_context(tmp_path, monkeypatch, write, finish):
    cp = pytest.importorskip("cupy")
    expected = np.arange(4096, dtype=np.uint8)
    path = tmp_path / "stream"
    path.write_bytes(b"!" * 4096 if write else expected.tobytes())
    buf = cp.asarray(expected) if write else cp.zeros(4096, dtype=cp.uint8)
    stream = cp.cuda.Stream()
    cp.cuda.get_current_stream().synchronize()
    current = _runtime.cuda(_runtime.driver.cuCtxGetCurrent())
    with kvikio.CuFile(path, "r+") as file:
        _require_caller_context(monkeypatch)
        operation = file.raw_write_async if write else file.raw_read_async
        pending = operation(buf, stream.ptr)
        if finish == "file_close":
            file.close()
        assert pending.check_bytes_done() == 4096
        assert pending.check_bytes_done() == 4096
        assert not file._handle._streams
        assert _runtime.cuda(_runtime.driver.cuCtxGetCurrent()) == current
    np.testing.assert_array_equal(cp.asnumpy(buf), expected)
    assert path.read_bytes() == expected.tobytes()


@pytest.mark.agent_authored(model="gpt-6-astra")
def test_reset_during_overlapping_io_preserves_new_busy_interval(tmp_path, monkeypatch):
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    original = file_handle.CuFile._transfer

    def delayed(self, arr, size, file_offset, *args):
        index = int(file_offset != 0)
        entered[index].set()
        assert release[index].wait(10)
        return original(self, arr, size, file_offset, *args)

    monkeypatch.setattr(file_handle.CuFile, "_transfer", delayed)
    with (
        kvikio.defaults.set("num_threads", 2),
        kvikio.SummaryMonitor() as monitor,
        kvikio.CuFile(tmp_path / "data", "w") as file,
    ):
        try:
            old = file.pwrite(bytearray(128))
            assert entered[0].wait(10)
            monitor.reset()
            new = file.pwrite(bytearray(128), file_offset=128)
            assert entered[1].wait(10)
            release[0].set()
            assert old.get() == 128
            before = monitor.get().busy_ns
            time.sleep(0.03)
            assert monitor.get().busy_ns - before > 10_000_000
            release[1].set()
            assert new.get() == 128
            assert monitor.get().num_ops == 1
        finally:
            for gate in release:
                gate.set()


@pytest.mark.agent_authored(model="gpt-6-astra")
def test_no_cpp_kvikio_is_loaded():
    assert "prototypes/kvikio_python" in kvikio.__file__
    assert "libkvikio.so" not in Path("/proc/self/maps").read_text()
    assert all(
        not getattr(value, "__file__", "").endswith(".so")
        for name, value in sys.modules.items()
        if name.startswith("cufile_python.") and getattr(value, "__file__", None)
    )


@pytest.mark.agent_authored(model="gpt-6-astra")
def test_host_export_retained_until_completion(tmp_path, monkeypatch):
    path = tmp_path / "data"
    path.write_bytes(b"x" * 4096)
    entered, release = threading.Event(), threading.Event()
    original = file_handle.CuFile._transfer

    def blocked(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)

    monkeypatch.setattr(file_handle.CuFile, "_transfer", blocked)
    buf = bytearray(4096)
    with kvikio.CuFile(path) as f:
        result = f.pread(buf)
        try:
            assert entered.wait(10)
            with pytest.raises(BufferError):
                buf.extend(b"y")
        finally:
            release.set()
        assert result.get() == 4096
    gc.collect()
    buf.extend(b"y")
    assert buf == b"x" * 4096 + b"y"


@pytest.mark.agent_authored(model="gpt-6-astra")
def test_close_drains_pending_worker(tmp_path, monkeypatch):
    path = tmp_path / "data"
    path.write_bytes(b"x" * 4096)
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    original = file_handle.CuFile._transfer

    def blocked(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)

    monkeypatch.setattr(file_handle.CuFile, "_transfer", blocked)
    file = kvikio.CuFile(path)
    buf = bytearray(4096)
    future = file.pread(buf)

    def close():
        file.close()
        closed.set()

    thread = threading.Thread(target=close)
    assert entered.wait(10)
    thread.start()
    try:
        assert not closed.wait(0.05)
    finally:
        release.set()
        thread.join(10)
    assert closed.is_set()
    assert file.closed
    assert future.get() == 4096
    assert buf == b"x" * 4096


@pytest.mark.parametrize("mode", [kvikio.CompatMode.ON, kvikio.CompatMode.OFF])
@pytest.mark.agent_authored(model="gpt-6-astra")
def test_gpu_sync_stream_and_future_lifetimes(tmp_path, mode):
    cp = pytest.importorskip("cupy")
    source = cp.arange(1024**2, dtype=cp.uint8)
    cp.cuda.get_current_stream().synchronize()
    destination = cp.empty_like(source)
    stream = cp.cuda.Stream()
    with kvikio.defaults.set({"compat_mode": mode, "gds_threshold": 0}), kvikio.CuFile(tmp_path / "data", "w+") as file:
        assert file.pwrite(source).get() == source.nbytes
        future = file.raw_read_async(destination, stream.ptr)
        ref = weakref.ref(destination)
        del destination
        gc.collect()
        assert ref() is not None
        assert future.check_bytes_done() == source.nbytes
        assert future.check_bytes_done() == source.nbytes
        gc.collect()
        assert ref() is None
        destination = cp.empty_like(source)
        assert file.read(destination) == source.nbytes
        cp.testing.assert_array_equal(source, destination)


@pytest.mark.parametrize("mode", list(kvikio.CompatMode))
@pytest.mark.agent_authored(model="gpt-6")
def test_native_submission_failure_releases_retained_arguments(tmp_path, monkeypatch, mode):
    cp = pytest.importorskip("cupy")
    arr = cp.zeros(4096, dtype=cp.uint8)
    stream = cp.cuda.Stream()
    original = file_handle.native

    def fail(name, *args):
        if name == "read_async":
            raise RuntimeError("submission failed")
        return original(name, *args)

    with kvikio.defaults.set("compat_mode", mode), kvikio.CuFile(tmp_path / "data", "w+") as file:
        _require_caller_context(monkeypatch)
        monkeypatch.setattr(file_handle, "native", fail)
        before = len(_runtime._pending_streams)
        with pytest.raises(RuntimeError, match="submission failed"):
            file.raw_read_async(arr, stream.ptr)
        assert len(_runtime._pending_streams) == before
        assert not file._handle._streams


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.agent_authored(model="gpt-6-astra")
def test_unaligned_io_with_legacy_settings(tmp_path, direct):
    cp = pytest.importorskip("cupy")
    path = tmp_path / "data"
    path.write_bytes(b"z" * 20000)
    source = cp.arange(9999, dtype=cp.uint8)
    target = cp.empty_like(source)
    with (
        kvikio.defaults.set(
            {
                "compat_mode": kvikio.CompatMode.ON,
                "auto_direct_io_read": direct,
                "auto_direct_io_write": direct,
                "gds_threshold": 0,
            }
        ),
        kvikio.CuFile(path, "r+") as f,
    ):
        assert f.write(source, file_offset=123) == source.nbytes
        assert f.read(target, file_offset=123) == source.nbytes
    cp.testing.assert_array_equal(source, target)
    contents = path.read_bytes()
    assert contents[:123] == b"z" * 123
    assert contents[10122:] == b"z" * (20000 - 10122)


@pytest.mark.agent_authored(model="gpt-6-astra")
def test_registered_base_is_used_for_device_slice(tmp_path):
    cp = pytest.importorskip("cupy")
    data = cp.arange(2**20, dtype=cp.uint8)
    with kvikio.defaults.set({"compat_mode": kvikio.CompatMode.OFF, "gds_threshold": 0}):
        kvikio.memory_register(data)
        try:
            with pytest.raises(RuntimeError):
                kvikio.memory_register(data)
            with kvikio.CuFile(tmp_path / "data", "w+") as f:
                assert f.write(data[4096:8192]) == 4096
                assert (tmp_path / "data").read_bytes() == cp.asnumpy(data[4096:8192]).tobytes()
        finally:
            kvikio.memory_deregister(data)


@pytest.mark.agent_authored(model="gpt-6-astra")
def test_summary_binary_exchange_with_reference(tmp_path):
    path = tmp_path / "data"
    payload = np.arange(128, dtype=np.uint8)
    monitor = kvikio.SummaryMonitor()
    with kvikio.CuFile(path, "w") as f:
        f.write(payload)
    monitor.stop()
    summary = monitor.get()
    encoded = tmp_path / "summary"
    encoded.write_bytes(summary.serialize())
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    script = """import pathlib, sys, kvikio
p=pathlib.Path(sys.argv[1]);s=kvikio.Summary.deserialize(p.read_bytes())
assert s.bytes_transferred==128 and s.num_writes==1
p.write_bytes(s.serialize())
"""
    subprocess.run([sys.executable, "-c", script, str(encoded)], env=env, check=True, cwd="/tmp")
    assert kvikio.Summary.deserialize(encoded.read_bytes()).to_json() == summary.to_json()

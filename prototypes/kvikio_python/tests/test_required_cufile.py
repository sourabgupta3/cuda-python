# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import errno
import os
import subprocess
import sys

import pytest

import cufile_python as cufile
from cufile_python import buffer, defaults
from cufile_python._lib import buffer as registration
from cufile_python._lib import file_handle


@pytest.mark.parametrize("mode", list(cufile.CompatMode))
@pytest.mark.parametrize("device", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_legacy_settings_never_select_python_io(tmp_path, monkeypatch, mode, device):
    cp = pytest.importorskip("cupy")
    expected = bytes(range(256)) * 16
    source = cp.asarray(bytearray(expected)) if device else expected
    target = cp.empty(len(expected), dtype=cp.uint8) if device else bytearray(len(expected))
    calls = []
    original = file_handle.native

    def native(name, *args):
        calls.append(name)
        return original(name, *args)

    def forbidden(*_args):
        raise AssertionError("Python POSIX fallback must never run")

    monkeypatch.setattr(file_handle, "native", native)
    monkeypatch.setattr(os, "preadv", forbidden)
    monkeypatch.setattr(os, "pwrite", forbidden)
    settings = {
        "compat_mode": mode,
        "gds_threshold": 2**30,
        "bounce_buffer_size": 1,
        "auto_direct_io_read": False,
        "auto_direct_io_write": False,
    }
    with defaults.set(settings), cufile.CuFile(tmp_path / "data", "w+") as file:
        assert defaults.get("compat_mode") == mode
        assert not defaults.is_compat_mode_preferred()
        assert file.pwrite(source).get() == len(expected)
        assert file.pread(target).get() == len(expected)
    actual = cp.asnumpy(target).tobytes() if device else bytes(target)
    assert actual == expected
    assert calls == ["handle_register", "write", "read", "handle_deregister"]
    assert buffer.bounce_buffer_free() == 0


@pytest.mark.parametrize("mode", list(cufile.CompatMode))
@pytest.mark.agent_authored(model="gpt-6")
def test_registration_failure_is_not_a_file_fallback(tmp_path, monkeypatch, mode):
    path = tmp_path / "data"
    path.write_bytes(b"keep existing contents")
    descriptors = []
    original = os.open

    def opened(*args, **kwargs):
        fd = original(*args, **kwargs)
        descriptors.append(fd)
        return fd

    def unavailable(name, *_args):
        assert name == "handle_register"
        raise RuntimeError("injected cuFile unavailable")

    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(file_handle, "native", unavailable)
    with defaults.set("compat_mode", mode), pytest.raises(RuntimeError, match="cuFile unavailable"):
        cufile.CuFile(path)
    assert len(descriptors) == 2
    for fd in descriptors:
        with pytest.raises(OSError, match="Bad file descriptor") as error:
            os.fstat(fd)
        assert error.value.errno == errno.EBADF
    assert path.read_bytes() == b"keep existing contents"


@pytest.mark.parametrize("mode", list(cufile.CompatMode))
@pytest.mark.agent_authored(model="gpt-6")
def test_memory_registration_always_calls_native(monkeypatch, mode):
    cp = pytest.importorskip("cupy")
    data = cp.arange(8192, dtype=cp.uint8)
    calls = []
    original = registration.native

    def native(name, *args):
        calls.append(name)
        return original(name, *args)

    monkeypatch.setattr(registration, "native", native)
    with defaults.set("compat_mode", mode):
        buffer.memory_register(data[4096:])
        try:
            with pytest.raises(RuntimeError):
                buffer.memory_register(data)
        finally:
            buffer.memory_deregister(data[4096:])
    assert calls == ["buf_register", "buf_register", "buf_deregister"]
    cp.testing.assert_array_equal(data, cp.arange(data.size, dtype=cp.uint8))


@pytest.mark.parametrize("mode", ["ON", "OFF", "AUTO"])
@pytest.mark.agent_authored(model="gpt-6")
def test_environment_setting_cannot_bypass_missing_cufile(tmp_path, mode):
    script = """
import sys
import cufile_python as api
from cufile_python._lib import file_handle
assert api.defaults.get('compat_mode') == getattr(api.CompatMode, sys.argv[2])
assert not api.defaults.is_compat_mode_preferred()
def unavailable(*args):
    raise RuntimeError('injected missing libcufile')
file_handle.native = unavailable
try:
    api.CuFile(sys.argv[1], 'w+')
except RuntimeError as error:
    assert 'missing libcufile' in str(error)
else:
    raise AssertionError('cuFile must be required')
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "data"), mode],
        env=dict(os.environ, KVIKIO_COMPAT_MODE=mode),
        check=True,
        timeout=30,
    )

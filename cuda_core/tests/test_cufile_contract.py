# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation of native file I/O parameters and error contracts."""

import ctypes
import os
import pickle
import sys

import numpy as np
import pytest

if sys.platform != "linux":
    pytest.skip("cuFile is Linux-only", allow_module_level=True)

from cuda.core import CUDAError, cufile
from cuda.core._cufile_native import BufferLease

pytestmark = pytest.mark.thread_unsafe(reason="injects native binding function lookup")


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.agent_authored(model="gpt-6")
def test_host_buffer_accepts_both_contiguous_orders(order):
    array = np.zeros((8, 4), dtype=np.uint32, order=order)
    lease = BufferLease(array, writable=True)
    assert lease.ptr == array.ctypes.data
    assert lease.nbytes == array.nbytes
    assert not lease.is_device


@pytest.mark.agent_authored(model="gpt-6")
def test_host_buffer_rejects_noncontiguous_and_readonly_destinations():
    with pytest.raises(ValueError):
        BufferLease(np.zeros(16)[::2], writable=True)
    with pytest.raises(ValueError):
        BufferLease(b"immutable", writable=True)
    assert BufferLease(b"immutable").nbytes == 9


@pytest.mark.parametrize("capacity", [0, -1, 1.5, 1 << 32])
@pytest.mark.agent_authored(model="gpt-6")
def test_batch_capacity_is_validated_before_native_setup(capacity):
    with pytest.raises((TypeError, ValueError)):
        cufile.Batch(capacity)


@pytest.mark.parametrize("mode", ["a", "a+", "rb", "invalid"])
@pytest.mark.agent_authored(model="gpt-6")
def test_unsupported_modes_are_rejected_before_open(tmp_path, mode):
    path = tmp_path / "unchanged.bin"
    path.write_bytes(b"original")
    with pytest.raises(ValueError):
        cufile.File(path, mode)
    assert path.read_bytes() == b"original"


@pytest.mark.parametrize(
    "flags",
    [
        os.O_WRONLY,
        os.O_RDWR,
        os.O_CREAT,
        os.O_TRUNC,
        os.O_APPEND,
        os.O_NOCTTY,
        os.O_NONBLOCK,
        os.O_DIRECTORY,
        os.O_NOFOLLOW,
        os.O_NOATIME,
        os.O_TMPFILE,
        1 << 30,
        -1,
        1 << 31,
        1.5,
        "O_DIRECT",
        None,
    ],
)
@pytest.mark.agent_authored(model="gpt-6")
def test_invalid_open_flags_do_not_create_or_truncate(tmp_path, flags):
    existing = tmp_path / "unchanged.bin"
    existing.write_bytes(b"original")
    missing = tmp_path / "missing.bin"
    for path in (existing, missing):
        with pytest.raises((TypeError, ValueError)):
            cufile.File(path, "w+", flags=flags)
    assert existing.read_bytes() == b"original"
    assert not missing.exists()


@pytest.mark.parametrize("mode", ["r", "r+"])
@pytest.mark.agent_authored(model="gpt-6")
def test_exclusive_flag_requires_a_creating_mode(tmp_path, mode):
    path = tmp_path / "unchanged.bin"
    path.write_bytes(b"original")
    with pytest.raises(ValueError):
        cufile.File(path, mode, flags=os.O_EXCL)
    assert path.read_bytes() == b"original"


@pytest.mark.parametrize("returned,errno_value", [(0, 0), (13, 0), (-5030, 0), (-9999, 0), (-1, 9), (-1, 28)])
@pytest.mark.agent_authored(model="gpt-6")
def test_native_return_counts_and_errno_are_preserved(monkeypatch, returned, errno_value):
    from cuda.bindings._internal import cufile as loader
    from cuda.core._cufile_native import NativeTransfers

    signature = ctypes.CFUNCTYPE(
        ctypes.c_ssize_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int64,
        ctypes.c_int64,
        use_errno=True,
    )

    @signature
    def operation(handle, pointer, size, file_offset, buffer_offset):
        ctypes.set_errno(errno_value)
        return returned

    monkeypatch.setattr(
        loader, "_inspect_function_pointer", lambda _name: ctypes.cast(operation, ctypes.c_void_p).value
    )
    table = NativeTransfers()
    if returned >= 0:
        assert table.transfer(0, 0, 4096, 0, 0, False) == returned
    else:
        with pytest.raises(cufile.CuFileError) as error:
            table.transfer(0, 0, 4096, 0, 0, False)
        assert error.value.operation == "read"
        assert error.value.native_result == returned
        if returned == -1:
            assert type(error.value) is cufile.CuFileError
            assert not isinstance(error.value, OSError)
            assert error.value.errno == errno_value
            assert error.value.strerror == os.strerror(errno_value)
            assert error.value.status is None
            assert isinstance(error.value.__cause__, OSError)
            assert error.value.__cause__.errno == errno_value
            assert error.value.__cause__.strerror == error.value.strerror
        else:
            assert error.value.native_code == -returned


@pytest.mark.agent_authored(model="gpt-6")
def test_native_error_retains_core_type_and_status():
    error = cufile.CuFileError("transfer failed", 5030)
    assert isinstance(error, CUDAError)
    assert error.native_code == 5030


@pytest.mark.parametrize("status,name", [(None, None), (5030, "INTERNAL_ERROR"), (9999, None)])
@pytest.mark.agent_authored(model="gpt-6")
def test_cufile_status_metadata_preserves_unknown_codes(status, name):
    error = cufile.CuFileError("failure", status, operation="read", native_result=-5030)
    assert error.status == error.native_code == status
    assert error.status_name == name
    assert error.cuda_error is None
    assert error.errno is None
    assert error.filename is None
    assert error.strerror is None
    assert error.operation == "read"
    assert error.native_result == -5030


@pytest.mark.agent_authored(model="gpt-6")
def test_binding_error_is_chained_and_preserves_cuda_status(monkeypatch):
    from cuda.bindings import cufile as bindings
    from cuda.bindings import driver
    from cuda.core.cufile._runtime import call

    cause = bindings.cuFileError(bindings.OpError.CUDA_DRIVER_ERROR, driver.CUresult.CUDA_ERROR_INVALID_CONTEXT)

    def fail():
        raise cause

    monkeypatch.setattr(bindings, "driver_open", fail)
    with pytest.raises(cufile.CuFileError) as caught:
        call("driver_open")
    error = caught.value
    assert error.__cause__ is cause
    assert error.status == int(bindings.OpError.CUDA_DRIVER_ERROR)
    assert error.status_name == "CUDA_DRIVER_ERROR"
    assert error.cuda_error == driver.CUresult.CUDA_ERROR_INVALID_CONTEXT
    assert error.operation == "driver_open"
    assert error.native_result is None
    assert error.errno is None


@pytest.mark.parametrize("os_failure", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_error_metadata_survives_pickling(os_failure):
    if os_failure:
        error = cufile.CuFileError(
            "missing", errno=2, filename="missing.bin", strerror="missing", operation="open", native_result=-1
        )
    else:
        error = cufile.CuFileError("failed", 5030, cuda_error=201, operation="read", native_result=-5030)
    assert type(error) is cufile.CuFileError
    assert isinstance(error, CUDAError)
    assert not isinstance(error, OSError)
    copied = pickle.loads(pickle.dumps(error))  # noqa: S301 - round-trip a locally constructed test exception
    assert type(copied) is type(error)
    assert copied.args == error.args
    for attribute in (
        "status",
        "status_name",
        "native_code",
        "cuda_error",
        "errno",
        "filename",
        "strerror",
        "operation",
        "native_result",
    ):
        assert getattr(copied, attribute) == getattr(error, attribute)
    if os_failure:
        assert copied.filename == "missing.bin"
        assert copied.strerror == "missing"


@pytest.mark.agent_authored(model="gpt-6")
def test_cufile_exposes_one_error_class():
    assert "CuFileError" in cufile.__all__
    assert not hasattr(cufile, "CuFileOSError")


@pytest.mark.parametrize("kind,flags", [("buffer", -1), ("buffer", 1), ("stream", -1), ("stream", 16)])
@pytest.mark.agent_authored(model="gpt-6")
def test_registration_flags_rejected_before_resource_access(kind, flags):
    resource = cufile.RegisteredBuffer if kind == "buffer" else cufile.RegisteredStream
    with pytest.raises(ValueError):
        resource(object(), flags=flags)


@pytest.mark.parametrize("kind", ["buffer", "stream"])
@pytest.mark.parametrize("flags", [None, 1.5, "0"])
@pytest.mark.agent_authored(model="gpt-6")
def test_registration_flags_require_integers(kind, flags):
    resource = cufile.RegisteredBuffer if kind == "buffer" else cufile.RegisteredStream
    with pytest.raises(TypeError):
        resource(object(), flags=flags)


@pytest.mark.parametrize("size", [0, -1, 17, 1 << 64, 1.5, "16"])
@pytest.mark.agent_authored(model="gpt-6")
def test_registration_size_rejected_before_native_setup(monkeypatch, size):
    from cuda.core.cufile import _buffers

    def unexpected_registration(*args):
        pytest.fail("Invalid registration size reached native setup")

    monkeypatch.setattr(_buffers, "_register_memory", unexpected_registration)
    with pytest.raises((TypeError, ValueError)):
        cufile.RegisteredBuffer(bytearray(16), size)


@pytest.mark.agent_authored(model="gpt-6")
def test_empty_buffer_cannot_be_registered():
    with pytest.raises(ValueError, match="size"):
        cufile.RegisteredBuffer(bytearray())

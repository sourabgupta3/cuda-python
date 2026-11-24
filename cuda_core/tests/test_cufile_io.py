# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real libcufile integration tests on an explicitly selected storage mount.

CUDA_CORE_CUFILE_TEST_DIR defaults to /mnt/nvme0. Every file is contained in a
new temporary directory; existing files on the mount are never modified.
"""

import ctypes
import errno
import fcntl
import gc
import json
import mmap
import os
import pathlib
import runpy
import subprocess
import sys
import tempfile
import textwrap
import threading
import weakref
from contextlib import ExitStack

import numpy as np
import pytest

if sys.platform != "linux":
    pytest.skip("cuFile is Linux-only", allow_module_level=True)

import cupy as cp
from helpers.constants import POOL_SIZE
from helpers.memory import skip_if_pinned_memory_unsupported

from cuda.bindings import driver
from cuda.core import (
    Buffer,
    DeviceMemoryResource,
    PinnedMemoryResource,
    PinnedMemoryResourceOptions,
    VirtualMemoryResource,
    VirtualMemoryResourceOptions,
    cufile,
)
from cuda.core._utils.cuda_utils import handle_return

BLOCK = 4096

pytestmark = pytest.mark.thread_unsafe(reason="injects cuFile binding failures")


def _submit_stream(operation, buf, stream, size=None, **offsets):
    params = cufile.IOParams(buf, size, **offsets)
    operation(params, stream)
    return params


def _stream_bytes(operation, buf, stream, size=None, *, synchronize, **offsets):
    params = _submit_stream(operation, buf, stream, size, **offsets)
    synchronize()
    return params.bytes_done


@pytest.fixture(scope="module")
def storage():
    root = pathlib.Path(os.environ.get("CUDA_CORE_CUFILE_TEST_DIR", "/mnt/nvme0"))
    if not root.is_dir():
        pytest.skip(f"cuFile test mount does not exist: {root}")
    with tempfile.TemporaryDirectory(prefix="cuda-core-cufile-", dir=root) as directory:
        yield pathlib.Path(directory)


@pytest.fixture
def file_path(storage, request):
    return storage / f"{request.node.name}.bin"


@pytest.fixture
def data(init_cuda):
    expected = np.arange(BLOCK * 4, dtype=np.uint8)
    source = cp.asarray(expected)
    destination = cp.asarray(np.zeros_like(expected))
    cp.cuda.get_current_stream().synchronize()
    yield expected, source, destination
    cp.cuda.get_current_stream().synchronize()


@pytest.mark.parametrize("memory", ["host", "device"])
@pytest.mark.agent_authored(model="gpt-6")
def test_sync_params_reuse_returns_counts_directly(file_path, init_cuda, data, memory):
    expected, source, destination = data
    if memory == "host":
        source, destination = expected.copy(), np.zeros_like(expected)
    write_params = cufile.IOParams(source, BLOCK)
    read_params = cufile.IOParams(destination, BLOCK)
    with cufile.File(file_path, "w+") as file:
        for offset in range(0, expected.nbytes, BLOCK):
            for params in (write_params, read_params):
                params.file_offset = params.buffer_offset = offset
            assert file.write(write_params) == BLOCK
            assert file.read(read_params) == BLOCK
            for params in (write_params, read_params):
                with pytest.raises(RuntimeError):
                    _ = params.bytes_done
                assert not params.is_closed
        read_params.file_offset = expected.nbytes
        read_params.buffer_offset = 0
        assert file.read(read_params) == 0
        read_params.size = 0
        assert file.read(read_params) == 0
    actual = cp.asnumpy(destination) if memory == "device" else destination
    np.testing.assert_array_equal(actual, expected)
    assert file_path.read_bytes() == expected.tobytes()


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_params_reuse_between_sync_and_stream_io(file_path, init_cuda, data, registered):
    expected, source, _ = data
    stream = init_cuda.create_stream()
    try:
        with ExitStack() as resources:
            file = resources.enter_context(cufile.File(file_path, "w+"))
            buffer = resources.enter_context(cufile.RegisteredBuffer(source)) if registered else source
            params = resources.enter_context(cufile.IOParams(buffer, BLOCK, buffer_offset=BLOCK))
            file.write_async(params, stream)
            stream.sync()
            # Reuse after completion without fetching the previous stream result.
            params.file_offset = BLOCK
            assert file.write(params) == BLOCK
            with pytest.raises(RuntimeError):
                _ = params.bytes_done
            params.buffer_offset = 2 * BLOCK
            file.write_async(params, stream)
            stream.sync()
            assert params.bytes_done == BLOCK
        assert file_path.read_bytes() == expected[BLOCK : 3 * BLOCK].tobytes()
    finally:
        stream.sync()
        stream.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_file_methods_require_open_params(file_path, init_cuda):
    with cufile.File(file_path, "w+") as file:
        for method in (file.read, file.write):
            with pytest.raises(TypeError):
                method(bytearray(BLOCK))
        for method in (file.read_async, file.write_async):
            with pytest.raises(TypeError):
                method(bytearray(BLOCK), 0)
        params = cufile.IOParams(bytearray(BLOCK))
        params.close()
        for method in (file.read, file.write):
            with pytest.raises(RuntimeError):
                method(params)
        for method in (file.read_async, file.write_async):
            with pytest.raises(RuntimeError):
                method(params, 0)
    assert file_path.read_bytes() == b""


@pytest.mark.parametrize("flags", [None, 0, os.O_DIRECT, os.O_SYNC, os.O_DSYNC, os.O_DIRECT | os.O_SYNC, os.O_CLOEXEC])
@pytest.mark.agent_authored(model="gpt-6")
def test_file_open_flags_and_access_modes(file_path, init_cuda, flags):
    options = {} if flags is None else {"flags": flags}
    expected = os.O_DIRECT if flags is None else flags
    for mode, access in (("w", os.O_WRONLY), ("r", os.O_RDONLY), ("r+", os.O_RDWR), ("w+", os.O_RDWR)):
        with cufile.File(file_path, mode, **options) as file:
            actual = fcntl.fcntl(file.fileno(), fcntl.F_GETFL)
            assert actual & os.O_ACCMODE == access
            assert actual & (os.O_DIRECT | os.O_SYNC | os.O_DSYNC) == expected & (os.O_DIRECT | os.O_SYNC | os.O_DSYNC)
            assert not os.get_inheritable(file.fileno())


@pytest.mark.parametrize("mode", ["w", "w+"])
@pytest.mark.agent_authored(model="gpt-6")
def test_exclusive_creation_preserves_existing_file(file_path, init_cuda, mode):
    with cufile.File(file_path, mode, flags=os.O_DIRECT | os.O_EXCL):
        pass
    file_path.write_bytes(b"original")
    with pytest.raises(cufile.CuFileError) as error:
        cufile.File(file_path, mode, flags=os.O_DIRECT | os.O_EXCL)
    assert error.value.errno == errno.EEXIST
    assert file_path.read_bytes() == b"original"


@pytest.mark.parametrize("mode", ["sync", "stream", "batch"])
@pytest.mark.parametrize("flags", [0, os.O_DIRECT | os.O_SYNC])
@pytest.mark.agent_authored(model="gpt-6")
def test_open_flags_across_io_paths(file_path, init_cuda, data, mode, flags):
    expected, source, destination = data
    stream = init_cuda.create_stream()
    try:
        with cufile.File(file_path, "w+", flags=flags) as file:
            if mode == "sync":
                assert file.write(cufile.IOParams(source)) == source.nbytes
                assert file.read(cufile.IOParams(destination)) == destination.nbytes
            elif mode == "stream":
                with cufile.IOParams(source) as write_params, cufile.IOParams(destination) as read_params:
                    file.write_async(write_params, stream)
                    file.read_async(read_params, stream)
                    stream.sync()
                    assert write_params.bytes_done == source.nbytes
                    assert read_params.bytes_done == destination.nbytes
            else:
                with cufile.Batch(1) as batch:
                    for buffer, opcode in ((source, "write"), (destination, "read")):
                        batch.submit([cufile.BatchIOParams(file, buffer, opcode, buffer.nbytes)])
                        assert [item.result() for item in batch.wait_all(timeout=10)] == [buffer.nbytes]
        np.testing.assert_array_equal(cp.asnumpy(destination), expected)
        assert file_path.read_bytes() == expected.tobytes()
    finally:
        stream.close()


@pytest.mark.parametrize("mode", ["sync", "stream_int", "stream_object"])
@pytest.mark.agent_authored(model="gpt-6")
def test_native_transfer_families(file_path, init_cuda, data, mode):
    expected, source, destination = data
    stream = init_cuda.create_stream()
    try:
        with cufile.File(file_path, "w+") as file:
            if mode == "sync":
                assert file.write(cufile.IOParams(source)) == source.nbytes
                assert file.read(cufile.IOParams(destination)) == destination.nbytes
            else:
                handle = int(stream.handle) if mode == "stream_int" else stream
                assert _stream_bytes(file.write_async, source, handle, synchronize=stream.sync) == source.nbytes
                request = _submit_stream(file.read_async, destination, handle)
                stream.sync()
                assert request.bytes_done == destination.nbytes
                stream.sync()
                assert request.bytes_done == destination.nbytes
                assert not hasattr(request, "wait")
        np.testing.assert_array_equal(cp.asnumpy(destination), expected)
        assert file.is_closed
        file.close()
    finally:
        stream.close()


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.agent_authored(model="gpt-6")
def test_host_buffers_and_file_offsets(file_path, init_cuda, order):
    source = np.array(np.arange(BLOCK, dtype=np.uint8).reshape(64, 64), order=order)
    destination = np.zeros_like(source, order=order)
    with cufile.File(file_path, "w+") as file:
        assert file.write(cufile.IOParams(source, file_offset=BLOCK)) == BLOCK
        assert file.read(cufile.IOParams(destination, file_offset=BLOCK)) == BLOCK
        assert file.fileno() >= 0
        os.fsync(file.fileno())
    np.testing.assert_array_equal(source, destination)
    assert file_path.read_bytes()[:BLOCK] == bytes(BLOCK)


@pytest.mark.parametrize("kind", ["bytearray", "numpy_c", "numpy_f", "mmap", "core_pinned"])
@pytest.mark.parametrize("mode", ["sync", "stream", "stream_int", "batch"])
@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_host_memory_native_paths(file_path, init_cuda, kind, mode, registered):
    expected = bytes(range(256)) * 16
    initial_source = b"s" * BLOCK + expected + b"s" * BLOCK
    initial_destination = b"d" * (3 * BLOCK)
    stream = init_cuda.create_stream()
    try:
        with ExitStack() as resources:
            if kind == "core_pinned":
                skip_if_pinned_memory_unsupported(init_cuda)
                memory = PinnedMemoryResource(PinnedMemoryResourceOptions(max_size=POOL_SIZE))
                resources.callback(memory.close)
                source = resources.enter_context(memory.allocate(3 * BLOCK, stream=stream))
                destination = resources.enter_context(memory.allocate(3 * BLOCK, stream=stream))
                stream.sync()
            elif kind == "mmap":
                source = resources.enter_context(mmap.mmap(-1, 3 * BLOCK))
                destination = resources.enter_context(mmap.mmap(-1, 3 * BLOCK))
            elif kind.startswith("numpy"):
                order = "F" if kind == "numpy_f" else "C"
                source = np.empty((64, 192), dtype=np.uint8, order=order)
                destination = np.empty_like(source, order=order)
            else:
                source, destination = bytearray(3 * BLOCK), bytearray(3 * BLOCK)
            for buffer, content in ((source, initial_source), (destination, initial_destination)):
                if isinstance(buffer, Buffer):
                    ctypes.memmove(int(buffer.handle), content, len(content))
                elif isinstance(buffer, np.ndarray):
                    buffer.ravel(order="K")[:] = np.frombuffer(content, dtype=np.uint8)
                else:
                    buffer[:] = content
            source_arg, destination_arg = source, destination
            if registered:
                source_arg = resources.enter_context(cufile.RegisteredBuffer(source))
                destination_arg = resources.enter_context(cufile.RegisteredBuffer(destination))
            file = resources.enter_context(cufile.File(file_path, "w+"))
            if mode == "sync":
                assert file.write(cufile.IOParams(source_arg, BLOCK, file_offset=BLOCK, buffer_offset=BLOCK)) == BLOCK
                assert (
                    file.read(cufile.IOParams(destination_arg, BLOCK, file_offset=BLOCK, buffer_offset=BLOCK)) == BLOCK
                )
            elif mode == "batch":
                batch = resources.enter_context(cufile.Batch(1))
                for buffer, opcode in ((source_arg, "write"), (destination_arg, "read")):
                    batch.submit([cufile.BatchIOParams(file, buffer, opcode, BLOCK, BLOCK, BLOCK)])
                    assert [item.result() for item in batch.wait_all(timeout=10)] == [BLOCK]
            else:
                selected = int(stream.handle) if mode == "stream_int" else stream
                with _submit_stream(
                    file.write_async, source_arg, selected, BLOCK, file_offset=BLOCK, buffer_offset=BLOCK
                ) as request:
                    stream.sync()
                    assert request.bytes_done == BLOCK
                with _submit_stream(
                    file.read_async, destination_arg, selected, BLOCK, file_offset=BLOCK, buffer_offset=BLOCK
                ) as request:
                    stream.sync()
                    assert request.bytes_done == BLOCK
            if isinstance(destination, Buffer):
                result = ctypes.string_at(int(destination.handle), 3 * BLOCK)
            elif isinstance(destination, np.ndarray):
                result = destination.tobytes(order="A")
            else:
                result = bytes(destination)
            assert result == b"d" * BLOCK + expected + b"d" * BLOCK
            assert file_path.read_bytes() == bytes(BLOCK) + expected
    finally:
        stream.sync()
        stream.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_host_stream_request_retains_buffer_exports(file_path, init_cuda):
    expected = bytes(range(256)) * 16
    source = np.frombuffer(expected, dtype=np.uint8).copy()
    source_reference = weakref.ref(source)
    stream = init_cuda.create_stream()
    try:
        with cufile.File(file_path, "w+") as file:
            request = _submit_stream(file.write_async, source, stream)
            del source
            gc.collect()
            assert source_reference() is not None
            stream.sync()
            assert request.bytes_done == BLOCK
            request.close()
            gc.collect()
            assert source_reference() is None
            destination = bytearray(BLOCK)
            request = _submit_stream(file.read_async, destination, stream)
            with pytest.raises(BufferError):
                destination.extend(b"x")
            stream.sync()
            assert request.bytes_done == BLOCK
            assert destination == expected
            destination.extend(b"x")
            assert destination == expected + b"x"
    finally:
        stream.sync()
        stream.close()


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_batch_mixes_host_and_device_memory(storage, init_cuda, data, registered):
    expected, device_source, device_destination = data
    host_source = bytearray(expected[:BLOCK].tobytes())
    host_destination = bytearray(BLOCK)
    with ExitStack() as resources:
        source_arg, destination_arg = host_source, host_destination
        if registered:
            source_arg = resources.enter_context(cufile.RegisteredBuffer(host_source))
            destination_arg = resources.enter_context(cufile.RegisteredBuffer(host_destination))
        host_file = resources.enter_context(cufile.File(storage / f"host-mixed-{registered}.bin", "w+"))
        device_file = resources.enter_context(cufile.File(storage / f"device-mixed-{registered}.bin", "w+"))
        batch = resources.enter_context(cufile.Batch(2))
        for host_buffer, device_buffer, opcode in (
            (source_arg, device_source, "write"),
            (destination_arg, device_destination, "read"),
        ):
            batch.submit(
                [
                    cufile.BatchIOParams(host_file, host_buffer, opcode, BLOCK, cookie="host"),
                    cufile.BatchIOParams(device_file, device_buffer, opcode, BLOCK, cookie="device"),
                ]
            )
            completed = batch.wait_all(timeout=10)
            assert [item.cookie for item in completed] == ["host", "device"]
            assert [item.result() for item in completed] == [BLOCK, BLOCK]
        assert host_destination == expected[:BLOCK].tobytes()
        np.testing.assert_array_equal(cp.asnumpy(device_destination[:BLOCK]), expected[:BLOCK])


@pytest.mark.agent_authored(model="gpt-6")
def test_pending_host_batch_retains_buffer_exports(file_path, init_cuda):
    expected = bytes(range(256)) * 16
    source = bytearray(expected)
    with cufile.File(file_path, "w") as file, cufile.Batch(1) as batch:
        batch.submit([cufile.BatchIOParams(file, source, "write", BLOCK)])
        with pytest.raises(BufferError):
            source.extend(b"x")
        assert [item.result() for item in batch.wait_all(timeout=10)] == [BLOCK]
        source.extend(b"x")
    assert file_path.read_bytes() == expected


@pytest.mark.parametrize("kind", ["bytearray", "mmap"])
@pytest.mark.parametrize("phase", ["registration", "stream", "batch", "batch_registered"])
@pytest.mark.agent_authored(model="gpt-6")
def test_host_exports_block_pointer_invalidation_until_native_use_ends(file_path, init_cuda, kind, phase):
    expected = bytes(range(256)) * 16
    with ExitStack() as resources:
        if kind == "mmap":
            buffer = resources.enter_context(mmap.mmap(-1, BLOCK))
            buffer[:] = expected
            invalidate = buffer.close
        else:
            buffer = bytearray(expected)

            def invalidate():
                buffer.extend(b"x")

        stream = init_cuda.create_stream()
        resources.callback(stream.close)
        file = resources.enter_context(cufile.File(file_path, "w+"))
        if phase == "registration":
            registration = resources.enter_context(cufile.RegisteredBuffer(buffer))
        elif phase == "stream":
            params = resources.enter_context(cufile.IOParams(buffer, BLOCK))
            file.write_async(params, stream)
        else:
            argument = buffer
            if phase == "batch_registered":
                registration = resources.enter_context(cufile.RegisteredBuffer(buffer))
                argument = registration
            batch = resources.enter_context(cufile.Batch(1))
            batch.submit([cufile.BatchIOParams(file, argument, "write", BLOCK)])
        with pytest.raises(BufferError):
            invalidate()
        if phase == "registration":
            registration.close()
        elif phase == "stream":
            stream.sync()
            assert params.bytes_done == BLOCK
            params.close()
        else:
            assert batch.wait_all(timeout=10)[0].result() == BLOCK
            if phase == "batch_registered":
                with pytest.raises(BufferError):
                    invalidate()
                registration.close()
        invalidate()
        if phase != "registration":
            assert file_path.read_bytes() == expected


@pytest.mark.agent_authored(model="gpt-6")
def test_short_reads_and_zero_length(file_path, init_cuda, data):
    expected, _, destination = data
    file_path.write_bytes(expected[:BLOCK].tobytes())
    with cufile.File(file_path) as file:
        assert file.read(cufile.IOParams(destination)) == BLOCK
        assert file.read(cufile.IOParams(destination, size=0)) == 0
        stream = init_cuda.create_stream()
        try:
            request = _submit_stream(file.read_async, destination, stream, size=0)
            stream.sync()
            assert request.bytes_done == 0
            with request:
                stream.sync()
                assert request.bytes_done == 0
        finally:
            stream.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_buffer_offsets(file_path, init_cuda, data):
    expected, source, destination = data
    with cufile.File(file_path, "w+") as file:
        assert file.write(cufile.IOParams(source, size=BLOCK, file_offset=BLOCK, buffer_offset=BLOCK * 2)) == BLOCK
        assert file.read(cufile.IOParams(destination, size=BLOCK, file_offset=BLOCK, buffer_offset=BLOCK)) == BLOCK
    np.testing.assert_array_equal(cp.asnumpy(destination[BLOCK : BLOCK * 2]), expected[BLOCK * 2 : BLOCK * 3])


@pytest.mark.parametrize(
    "kind", ["size", "offset", "file_offset", "host_async_readonly", "host_async_strided", "readonly", "strided"]
)
@pytest.mark.agent_authored(model="gpt-6")
def test_invalid_transfers_leave_file_usable(file_path, init_cuda, data, kind):
    _, source, destination = data
    with cufile.File(file_path, "w+") as file:
        assert file.write(cufile.IOParams(source)) == source.nbytes
        with pytest.raises((TypeError, ValueError)):
            if kind == "size":
                file.read(cufile.IOParams(destination, size=destination.nbytes + 1))
            elif kind == "offset":
                file.read(cufile.IOParams(destination, size=BLOCK, buffer_offset=destination.nbytes))
            elif kind == "file_offset":
                file.read(cufile.IOParams(destination, file_offset=-1))
            elif kind == "host_async_readonly":
                _submit_stream(file.read_async, bytes(BLOCK), init_cuda.default_stream)
            elif kind == "host_async_strided":
                _submit_stream(file.read_async, np.zeros(BLOCK, dtype=np.uint8)[::2], init_cuda.default_stream)
            elif kind == "readonly":
                file.read(cufile.IOParams(bytes(BLOCK)))
            else:
                file.read(cufile.IOParams(destination[::2]))
        assert file.read(cufile.IOParams(destination)) == destination.nbytes


@pytest.mark.agent_authored(model="gpt-6")
def test_driver_scope_keeps_files_valid_while_another_driver_closes(file_path, init_cuda, data):
    _, source, destination = data
    with cufile.Driver():
        driver_scope = cufile.Driver()
        with cufile.File(file_path, "w+") as file:
            driver_scope.close()
            driver_scope.close()
            assert file.write(cufile.IOParams(source)) == source.nbytes
            with cufile.Driver(), cufile.Driver():
                assert file.read(cufile.IOParams(destination)) == destination.nbytes
            assert file.read(cufile.IOParams(destination)) == destination.nbytes


@pytest.mark.parametrize("flags", [0, 1, 2, 4, 8, 15])
@pytest.mark.agent_authored(model="gpt-6")
def test_scoped_stream_registration(file_path, init_cuda, data, flags):
    expected, source, destination = data
    stream = init_cuda.create_stream()
    try:
        with cufile.File(file_path, "w+") as file, cufile.RegisteredStream(stream, flags) as registered:
            assert registered.flags == flags
            assert _stream_bytes(file.write_async, source, registered, synchronize=stream.sync) == source.nbytes
            assert (
                _stream_bytes(file.read_async, destination, registered, synchronize=stream.sync) == destination.nbytes
            )
        np.testing.assert_array_equal(cp.asnumpy(destination), expected)
    finally:
        stream.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_scoped_buffer_registration(file_path, init_cuda, data):
    expected, source, destination = data
    with (
        cufile.RegisteredBuffer(source) as registered_source,
        cufile.RegisteredBuffer(destination, flags=0) as registered_destination,
        cufile.File(file_path, "w+") as file,
    ):
        assert registered_source.flags == registered_destination.flags == 0
        assert file.write(cufile.IOParams(registered_source)) == source.nbytes
        assert file.read(cufile.IOParams(registered_destination)) == destination.nbytes
    np.testing.assert_array_equal(cp.asnumpy(destination), expected)


@pytest.mark.parametrize("resource_type", [DeviceMemoryResource, VirtualMemoryResource])
@pytest.mark.agent_authored(model="gpt-6")
def test_core_buffers_retain_native_ownership(file_path, init_cuda, data, resource_type):
    expected, source, _ = data
    with cufile.File(file_path, "w+") as file:
        file.write(cufile.IOParams(source))
    stream = init_cuda.create_stream()
    resource = (
        VirtualMemoryResource(init_cuda, VirtualMemoryResourceOptions(gpu_direct_rdma=True))
        if resource_type is VirtualMemoryResource
        else resource_type(init_cuda)
    )
    buffer = resource.allocate(source.nbytes, stream=stream)
    try:
        with cufile.File(file_path) as file:
            future = _submit_stream(file.read_async, buffer, stream, size=source.nbytes)
            completion = stream.record()
            buffer.close()
            stream.close()
            try:
                completion.sync()
                assert future.bytes_done == expected.nbytes
            finally:
                completion.close()
    finally:
        buffer.close()
        stream.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_rdma_capable_core_vmm_registration(file_path, init_cuda, data):
    expected, source, _ = data
    stream = init_cuda.create_stream()
    resource = VirtualMemoryResource(init_cuda, VirtualMemoryResourceOptions(gpu_direct_rdma=True))
    buffer = resource.allocate(source.nbytes, stream=stream)
    try:
        handle_return(driver.cuMemcpyDtoD(int(buffer.handle), source.data.ptr, source.nbytes))
        with cufile.RegisteredBuffer(buffer) as registered, cufile.File(file_path, "w+") as file:
            future = _submit_stream(file.write_async, registered, stream, size=source.nbytes)
            buffer.close()
            stream.sync()
            assert future.bytes_done == source.nbytes
        assert file_path.read_bytes() == expected.tobytes()
    finally:
        buffer.close()
        stream.close()


@pytest.mark.parametrize("mode", ["allowed", "forced", "disabled"])
@pytest.mark.agent_authored(model="gpt-6")
def test_registration_capability_policy_in_all_modes(file_path, mode):
    program = textwrap.dedent("""
        import json
        import sys
        from contextlib import ExitStack

        import cupy as cp
        import numpy as np
        from cuda.bindings import cufile as native, driver
        from cuda.core import (
            Device, DeviceMemoryResource, VirtualMemoryResource,
            VirtualMemoryResourceOptions, cufile,
        )
        from cuda.core._cufile_native import BufferLease
        from cuda.core._utils.cuda_utils import handle_return

        mode, path = sys.argv[1:]
        native.set_parameter_bool(native.BoolConfigParameter.FORCE_COMPAT_MODE, mode == "forced")
        native.set_parameter_bool(native.BoolConfigParameter.PROPERTIES_ALLOW_COMPAT_MODE, mode != "disabled")
        native.set_parameter_bool(native.BoolConfigParameter.PROPERTIES_ALLOW_SYSTEM_MEMORY, True)
        device = Device(0)
        device.set_current()
        stream = device.create_stream()
        expected = np.arange(4096, dtype=np.uint8)
        rejected = 0
        ineligible = []
        transfers = 0
        try:
            with ExitStack() as cleanup:
                allocations = []
                for name, resource in [
                    ("vmm", VirtualMemoryResource(device)),
                    ("pool", DeviceMemoryResource(device)),
                ]:
                    if hasattr(resource, "close"):
                        cleanup.callback(resource.close)
                    allocation = cleanup.enter_context(resource.allocate(8192, stream=stream))
                    allocations.append((name, allocation))
                allocations.append(("cupy_async", cp.ndarray((8192,), dtype=cp.uint8,
                    memptr=cp.cuda.malloc_async(8192))))
                stream.sync()
                cp.cuda.get_current_stream().synchronize()

                original_open, original_register = native.driver_open, native.buf_register
                def unexpected_native_mutation(*args):
                    raise AssertionError("Unsupported registration reached a native cuFile mutation")
                native.driver_open = native.buf_register = unexpected_native_mutation
                try:
                    for name, allocation in allocations:
                        lease = BufferLease(allocation)
                        capable = handle_return(driver.cuPointerGetAttribute(
                            driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE,
                            lease.ptr))
                        if capable:
                            continue
                        ineligible.append(name)
                        view = cp.ndarray((4096,), dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(
                            cp.cuda.UnownedMemory(lease.ptr + 4096, 4096, allocation), 0))
                        for argument in (allocation, view):
                            for size in (None, 4096):
                                context = handle_return(driver.cuCtxGetCurrent())
                                try:
                                    cufile.RegisteredBuffer(argument, size)
                                except NotImplementedError:
                                    rejected += 1
                                else:
                                    raise AssertionError("Unsupported registration was accepted")
                                assert handle_return(driver.cuCtxGetCurrent()) == context
                finally:
                    native.driver_open, native.buf_register = original_open, original_register

                for name, allocation in allocations:
                    lease = BufferLease(allocation)
                    with cufile.File(path, "w+") as file:
                        for model in ("sync", "stream", "batch"):
                            for opcode in ("write", "read"):
                                if opcode == "write":
                                    handle_return(driver.cuMemcpyHtoD(lease.ptr, expected.ctypes.data, 4096))
                                else:
                                    handle_return(driver.cuMemsetD8(lease.ptr, 0, 4096))
                                if model == "batch":
                                    with cufile.Batch(1) as batch:
                                        batch.submit([cufile.BatchIOParams(file, allocation, opcode, 4096)])
                                        assert batch.wait_all()[0].result() == 4096
                                else:
                                    with cufile.IOParams(allocation, 4096) as params:
                                        if model == "sync":
                                            assert getattr(file, opcode)(params) == 4096
                                        else:
                                            getattr(file, opcode + "_async")(params, stream)
                                            stream.sync()
                                            assert params.bytes_done == 4096
                                if opcode == "read":
                                    actual = np.empty_like(expected)
                                    handle_return(driver.cuMemcpyDtoH(actual.ctypes.data, lease.ptr, 4096))
                                    np.testing.assert_array_equal(actual, expected)
                                transfers += 1
                    assert open(path, "rb").read() == expected.tobytes()

                resource = VirtualMemoryResource(device, VirtualMemoryResourceOptions(gpu_direct_rdma=True))
                if hasattr(resource, "close"):
                    cleanup.callback(resource.close)
                capable_buffer = cleanup.enter_context(resource.allocate(4096, stream=stream))
                conventional = cp.ndarray((4096,), dtype=cp.uint8,
                    memptr=cp.cuda.MemoryPointer(cp.cuda.Memory(4096), 0))
                stream.sync()
                for allocation in (capable_buffer, conventional, bytearray(4096)):
                    with cufile.RegisteredBuffer(allocation, size=4096) as registered:
                        assert registered.size == 4096
                stream.sync()
        finally:
            stream.close()
        print(json.dumps({"ineligible": ineligible, "rejected": rejected, "transfers": transfers}))
    """)
    result = subprocess.run(  # noqa: S603 -- fixed test program and private test path
        [sys.executable, "-c", program, mode, str(file_path)], capture_output=True, text=True, timeout=25
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["transfers"] == 18
    assert report["rejected"] == 4 * len(report["ineligible"])
    if not report["ineligible"]:
        pytest.skip("Available allocations all report GPUDirect capability")


@pytest.mark.parametrize("attribute", ["MEMORY_TYPE", "IS_GPU_DIRECT_RDMA_CAPABLE"])
@pytest.mark.agent_authored(model="gpt-6")
def test_registration_pointer_query_failure_preserves_cuda_error(init_cuda, data, monkeypatch, attribute):
    from cuda.core import CUDAError

    _, source, _ = data
    target = getattr(driver.CUpointer_attribute, "CU_POINTER_ATTRIBUTE_" + attribute)
    original = driver.cuPointerGetAttribute

    def fail_query(queried, pointer):
        if queried == target:
            return driver.CUresult.CUDA_ERROR_INVALID_CONTEXT, 0
        return original(queried, pointer)

    context = handle_return(driver.cuCtxGetCurrent())
    monkeypatch.setattr(driver, "cuPointerGetAttribute", fail_query)
    with pytest.raises(CUDAError):
        cufile.RegisteredBuffer(source)
    assert handle_return(driver.cuCtxGetCurrent()) == context


@pytest.mark.agent_authored(model="gpt-6")
def test_early_file_close_drains_pending_work(file_path, init_cuda, data):
    expected, source, destination = data
    stream = init_cuda.create_stream()
    try:
        file = cufile.File(file_path, "w+")
        request = _submit_stream(file.write_async, source, stream)
        file.close()
        stream.sync()
        assert request.bytes_done == source.nbytes
        with cufile.File(file_path) as file:
            assert file.read(cufile.IOParams(destination)) == destination.nbytes
        np.testing.assert_array_equal(cp.asnumpy(destination), expected)
    finally:
        stream.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_dropped_request_finishes_before_releasing_arguments(file_path, init_cuda, data):
    _, source, _ = data
    stream = init_cuda.create_stream()
    try:
        with cufile.File(file_path, "w+") as file:
            future = _submit_stream(file.write_async, source, stream)
            del future
            gc.collect()
        assert file_path.read_bytes() == cp.asnumpy(source).tobytes()
    finally:
        stream.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_no_current_context_is_preserved(file_path, init_cuda, data):
    _, source, destination = data
    previous = handle_return(driver.cuCtxGetCurrent())
    try:
        handle_return(driver.cuCtxSetCurrent(0))
        with cufile.File(file_path, "w+") as file:
            assert file.write(cufile.IOParams(source)) == source.nbytes
            assert file.read(cufile.IOParams(destination)) == destination.nbytes
        assert int(handle_return(driver.cuCtxGetCurrent())) == 0
    finally:
        handle_return(driver.cuCtxSetCurrent(previous))


@pytest.mark.agent_authored(model="gpt-6")
def test_shared_file_reads_across_host_threads(file_path, init_cuda, data):
    expected, source, _ = data
    file_path.write_bytes(expected.tobytes())
    destinations = [cp.asarray(np.zeros_like(expected)) for _ in range(4)]
    counts = [None] * len(destinations)
    with cufile.File(file_path) as file:

        def read_one(index):
            counts[index] = file.read(cufile.IOParams(destinations[index]))

        readers = [threading.Thread(target=read_one, args=(index,)) for index in range(len(destinations))]
        for reader in readers:
            reader.start()
        for reader in readers:
            reader.join()
        assert counts == [source.nbytes] * 4
    for destination in destinations:
        np.testing.assert_array_equal(cp.asnumpy(destination), expected)


@pytest.mark.agent_authored(model="gpt-6")
def test_native_sync_failure_leaves_file_usable(file_path, init_cuda, data, monkeypatch):
    from cuda.core.cufile import _file

    expected, _, destination = data
    file_path.write_bytes(expected.tobytes())

    def fail_transfer(*args):
        raise cufile.CuFileError("injected synchronous failure")

    params = cufile.IOParams(destination)
    with cufile.File(file_path) as file:
        with monkeypatch.context() as patch:
            patch.setattr(_file, "_native_transfer", fail_transfer)
            with pytest.raises(cufile.CuFileError):
                file.read(params)
        with pytest.raises(RuntimeError):
            _ = params.bytes_done
        assert file.read(params) == destination.nbytes
    np.testing.assert_array_equal(cp.asnumpy(destination), expected)


@pytest.mark.agent_authored(model="gpt-6")
def test_batch_multiple_files_reuse_and_completion_snapshots(storage, init_cuda, data):
    expected, source, destination = data
    first, second = storage / "batch-first.bin", storage / "batch-second.bin"
    with cufile.File(first, "w+") as a, cufile.File(second, "w+") as b, cufile.Batch(4) as batch:
        operations = [
            cufile.BatchIOParams(a, source, "write", BLOCK, cookie="first"),
            cufile.BatchIOParams(
                b, source, "write", BLOCK, file_offset=BLOCK, buffer_offset=BLOCK, cookie={"second": 2}
            ),
        ]
        for _ in range(3):
            batch.submit(operations)
            completions = batch.wait_all(timeout=10)
            assert [result.result() for result in completions] == [BLOCK, BLOCK]
            assert [result.cookie for result in completions] == ["first", {"second": 2}]
            assert batch.wait_all() == ()
        old = completions
        batch.submit(
            [
                cufile.BatchIOParams(a, destination, "read", BLOCK),
                cufile.BatchIOParams(b, destination, "read", BLOCK, file_offset=BLOCK, buffer_offset=BLOCK),
            ]
        )
        assert sum(item.result() for item in batch.wait_all(timeout=10)) == BLOCK * 2
        assert old[0].cookie == "first"
    np.testing.assert_array_equal(cp.asnumpy(destination[: BLOCK * 2]), expected[: BLOCK * 2])


@pytest.mark.agent_authored(model="gpt-6")
def test_pending_batch_retains_core_allocation_until_completion(file_path, init_cuda, data):
    expected, source, _ = data
    stream = init_cuda.create_stream()
    resource = VirtualMemoryResource(init_cuda)
    buffer = resource.allocate(source.nbytes, stream=stream)
    handle_return(driver.cuMemcpyDtoD(int(buffer.handle), source.data.ptr, source.nbytes))
    try:
        with cufile.File(file_path, "w+") as file, cufile.Batch(1) as batch:
            batch.submit([cufile.BatchIOParams(file, buffer, "write", source.nbytes)])
            buffer.close()
            assert batch.wait_all(timeout=10)[0].result() == source.nbytes
        assert file_path.read_bytes() == expected.tobytes()
    finally:
        buffer.close()
        stream.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_batch_active_state_and_polling(file_path, init_cuda, data):
    _, source, _ = data
    with cufile.File(file_path, "w+") as file, cufile.Batch(2) as batch:
        operation = cufile.BatchIOParams(file, source, "write", BLOCK)
        batch.submit([operation, operation])
        with pytest.raises(ValueError, match="capacity"):
            batch.submit([operation])
        received = batch.poll(min_complete=1, max_complete=1, timeout=10)
        assert len(received) == 1
        assert batch.is_active
        from cuda.bindings import cufile as backend

        if backend.get_version() == 1190:
            with pytest.raises(NotImplementedError, match="Rolling batch"):
                batch.submit([operation])
            remaining = batch.wait_all(timeout=10)
            assert len(remaining) == 1
            batch.submit([operation])
            remaining += batch.wait_all(timeout=10)
        else:
            batch.submit([operation])
            remaining = batch.wait_all(timeout=10)
        assert len(remaining) == 2
        assert {item.index for item in received}.isdisjoint(item.index for item in remaining)
        assert not batch.is_active
        assert batch.poll() == batch.wait_all() == ()


@pytest.mark.agent_authored(model="gpt-6")
def test_driver_lifetime_and_standalone_queries(init_cuda, monkeypatch):
    from cuda.bindings import cufile as backend

    calls = []
    native_open, native_close = backend.driver_open, backend.driver_close

    def open_driver():
        native_open()
        calls.append("open")

    def close_driver():
        native_close()
        calls.append("close")

    monkeypatch.setattr(backend, "driver_open", open_driver)
    monkeypatch.setattr(backend, "driver_close", close_driver)
    first = cufile.Driver()
    try:
        assert calls == ["open"]
        assert not first.is_closed
        with cufile.Driver() as second:
            assert calls == ["open", "open"]
            first.close()
            first.close()
            assert first.is_closed and not second.is_closed
            assert isinstance(backend.get_version(), int)
            assert isinstance(backend.get_parameter_bool(backend.BoolConfigParameter.PROPERTIES_USE_POLL_MODE), bool)
        assert second.is_closed
        assert calls == ["open", "open", "close", "close"]
    finally:
        first.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_failed_file_construction_closes_descriptors(file_path, init_cuda, monkeypatch):
    from cuda.bindings import cufile as backend

    descriptors = []
    original = os.open

    def tracking_open(*args, **kwargs):
        descriptor = original(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    def fail_register(*args):
        raise cufile.CuFileError("injected handle registration failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", tracking_open)
        patch.setattr(backend, "handle_register", fail_register)
        with pytest.raises(cufile.CuFileError):
            cufile.File(file_path, "w+")
    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    with cufile.File(file_path, "w+"):
        pass


@pytest.mark.agent_authored(model="gpt-6")
def test_registration_failure_and_cleanup_failure_can_be_retried(init_cuda, data, monkeypatch):
    from cuda.bindings import cufile as backend

    _, source, _ = data

    def fail(*args):
        raise cufile.CuFileError("injected registration failure")

    with monkeypatch.context() as patch:
        patch.setattr(backend, "buf_register", fail)
        with pytest.raises(cufile.CuFileError):
            cufile.RegisteredBuffer(source)
    registration = cufile.RegisteredBuffer(source)
    with monkeypatch.context() as patch:
        patch.setattr(backend, "buf_deregister", fail)
        with pytest.raises(cufile.CuFileError):
            registration.close()
        assert not registration.is_closed
    registration.close()
    assert registration.is_closed
    registration.close()
    with cufile.RegisteredBuffer(source):
        pass


@pytest.mark.parametrize("failure", ["size", "duplicate", "native", "runtime"])
@pytest.mark.agent_authored(model="gpt-6")
def test_failed_registration_releases_host_export_with_retained_exception(init_cuda, monkeypatch, failure):
    from cuda.bindings import cufile as backend
    from cuda.core.cufile._runtime import runtime

    def fail_register(*args):
        raise cufile.CuFileError("injected registration failure")

    with mmap.mmap(-1, BLOCK) as buffer:
        first = cufile.RegisteredBuffer(buffer) if failure == "duplicate" else None
        if failure == "native":
            monkeypatch.setattr(backend, "buf_register", fail_register)
        elif failure == "runtime":
            monkeypatch.setattr(runtime, "check", fail_register)
        try:
            with pytest.raises((ValueError, cufile.CuFileError)) as retained_error:
                cufile.RegisteredBuffer(buffer, size=BLOCK + 1 if failure == "size" else BLOCK)
        finally:
            if first is not None:
                first.close()
    assert retained_error.value.__traceback__ is not None


@pytest.mark.agent_authored(model="gpt-6")
def test_registration_aliases_preserve_view_bounds(file_path, init_cuda, data):
    expected, source, _ = data
    with cufile.RegisteredBuffer(source[BLOCK : BLOCK * 2]) as view:
        assert view.size == BLOCK
        with cufile.File(file_path, "w+") as file:
            assert file.write(cufile.IOParams(view)) == BLOCK
    assert file_path.read_bytes() == expected[BLOCK : BLOCK * 2].tobytes()


@pytest.mark.parametrize("kind", ["stream", "device_buffer", "host_buffer"])
@pytest.mark.agent_authored(model="gpt-6")
def test_registration_objects_reject_duplicates(file_path, init_cuda, data, monkeypatch, kind):
    from cuda.bindings import cufile as backend

    expected, source, destination = data
    counts = {"register": 0, "deregister": 0}
    prefix = "stream" if kind == "stream" else "buf"
    native_register = getattr(backend, prefix + "_register")
    native_deregister = getattr(backend, prefix + "_deregister")

    def register(*args):
        counts["register"] += 1
        return native_register(*args)

    def deregister(*args):
        counts["deregister"] += 1
        return native_deregister(*args)

    monkeypatch.setattr(backend, prefix + "_register", register)
    monkeypatch.setattr(backend, prefix + "_deregister", deregister)
    with ExitStack() as cleanup:
        stream = init_cuda.create_stream()
        cleanup.callback(stream.close)
        factory = cufile.RegisteredStream if kind == "stream" else cufile.RegisteredBuffer
        resource = stream if kind == "stream" else destination
        if kind == "host_buffer":
            resource = np.zeros_like(expected)
        first = cleanup.enter_context(factory(resource))
        with pytest.raises(cufile.CuFileError, match="already registered") as error:
            factory(resource)
        assert error.value.native_code == (5022 if kind == "stream" else 5023)
        if kind == "stream":
            with pytest.raises(cufile.CuFileError, match="already registered"):
                cufile.RegisteredStream(stream, flags=15)
        assert counts == {"register": 1, "deregister": 0}
        assert not first.is_closed
        with cufile.File(file_path, "w+") as file:
            assert file.write(cufile.IOParams(source)) == source.nbytes
            if kind == "stream":
                assert _stream_bytes(file.read_async, destination, first, synchronize=stream.sync) == source.nbytes
            else:
                assert file.read(cufile.IOParams(first)) == source.nbytes
        actual = resource if kind == "host_buffer" else cp.asnumpy(destination)
        np.testing.assert_array_equal(actual, expected)
        first.close()
        first.close()
        assert counts == {"register": 1, "deregister": 1}
        with factory(resource):
            assert counts == {"register": 2, "deregister": 1}
        assert counts == {"register": 2, "deregister": 2}


@pytest.mark.parametrize("memory", ["host", "device"])
@pytest.mark.parametrize("second_size", [BLOCK, 2 * BLOCK, 3 * BLOCK])
@pytest.mark.parametrize("offset", [0, BLOCK])
@pytest.mark.agent_authored(model="gpt-6")
def test_registration_rejects_different_sizes_and_overlapping_views(init_cuda, data, memory, second_size, offset):
    expected, source, _ = data
    if memory == "host":
        source = expected.copy()
    with cufile.RegisteredBuffer(source, 2 * BLOCK) as first:
        with pytest.raises(cufile.CuFileError, match="already registered") as error:
            cufile.RegisteredBuffer(source[offset:], size=second_size)
        assert error.value.status_name == "MEMORY_ALREADY_REGISTERED"
        assert first.size == 2 * BLOCK
        assert not first.is_closed
    with cufile.RegisteredBuffer(source, second_size) as replacement:
        assert replacement.size == second_size


@pytest.mark.parametrize("memory", ["host", "device"])
@pytest.mark.parametrize("mode", ["sync", "stream", "batch"])
@pytest.mark.parametrize("size", [None, 2 * BLOCK])
@pytest.mark.agent_authored(model="gpt-6")
def test_registration_optional_size_and_view_pointer(file_path, init_cuda, data, monkeypatch, memory, mode, size):
    from cuda.bindings import cufile as backend

    expected, source, _ = data
    if memory == "host":
        source = expected.copy()
    view = source[BLOCK:]
    length = view.nbytes if size is None else size
    pointer = view.ctypes.data if memory == "host" else view.data.ptr
    calls = []
    native_register = backend.buf_register

    def register(*args):
        calls.append(args)
        return native_register(*args)

    monkeypatch.setattr(backend, "buf_register", register)
    with ExitStack() as cleanup:
        stream = init_cuda.create_stream()
        cleanup.callback(stream.close)
        registered = cleanup.enter_context(cufile.RegisteredBuffer(view, size))
        assert registered.size == length
        assert calls == [(pointer, length, 0)]
        file = cleanup.enter_context(cufile.File(file_path, "w+"))
        batch = cleanup.enter_context(cufile.Batch(1))
        if mode == "sync":
            assert file.write(cufile.IOParams(registered)) == length
        elif mode == "stream":
            assert _stream_bytes(file.write_async, registered, stream, synchronize=stream.sync) == length
        else:
            batch.submit([cufile.BatchIOParams(file, registered, "write", length)])
            assert [event.result() for event in batch.wait_all(timeout=10)] == [length]
        with pytest.raises(ValueError):
            file.write(cufile.IOParams(registered, length + 1))
        with pytest.raises(ValueError):
            file.write_async(cufile.IOParams(registered, length + 1), stream)
        with pytest.raises(ValueError):
            batch.submit([cufile.BatchIOParams(file, registered, "write", length + 1)])
    assert registered.size == length
    assert file_path.read_bytes() == expected[BLOCK : BLOCK + length].tobytes()


@pytest.mark.parametrize("memory", ["host", "device"])
@pytest.mark.agent_authored(model="gpt-6")
def test_raw_buffer_transfer_respects_partial_registration(file_path, init_cuda, data, memory):
    expected, source, _ = data
    if memory == "host":
        source = expected.copy()
    with cufile.RegisteredBuffer(source, 2 * BLOCK), cufile.File(file_path, "w+") as file:
        assert file.write(cufile.IOParams(source, BLOCK, buffer_offset=BLOCK)) == BLOCK
        with pytest.raises(ValueError, match="registered buffer range"):
            file.write(cufile.IOParams(source, 2 * BLOCK, buffer_offset=BLOCK))
        assert file.write(cufile.IOParams(source, BLOCK, file_offset=BLOCK, buffer_offset=3 * BLOCK)) == BLOCK
    assert file_path.read_bytes() == expected[BLOCK : 2 * BLOCK].tobytes() + expected[3 * BLOCK :].tobytes()


@pytest.mark.parametrize("registered_argument", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_deferred_offsets_cannot_leave_partial_registration(file_path, init_cuda, data, registered_argument):
    expected, source, _ = data
    stream = init_cuda.create_stream()
    try:
        with (
            cufile.RegisteredBuffer(source[BLOCK:], BLOCK) as registered,
            cufile.File(file_path, "w+") as file,
        ):
            buf = registered if registered_argument else source
            initial_offset = 0 if registered_argument else BLOCK
            with cufile.IOParams(buf, BLOCK, buffer_offset=initial_offset) as params:
                file.write_async(params, stream)
                with pytest.raises(ValueError):
                    params.buffer_offset = initial_offset + BLOCK
                if not registered_argument:
                    with pytest.raises(ValueError, match="registered buffer range"):
                        params.buffer_offset = 0
                stream.sync()
                assert params.bytes_done == BLOCK
        assert file_path.read_bytes() == expected[BLOCK : 2 * BLOCK].tobytes()
    finally:
        stream.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_registration_close_drains_batch_and_releases_native_registration(file_path, init_cuda, data):
    expected, source, _ = data
    with cufile.File(file_path, "w+") as file, cufile.Batch(1) as batch:
        registered = cufile.RegisteredBuffer(source, BLOCK)
        batch.submit([cufile.BatchIOParams(file, registered, "write", BLOCK)])
        with pytest.raises(cufile.CuFileError, match="already registered"):
            cufile.RegisteredBuffer(source, BLOCK)
        registered.close()
        assert registered.is_closed
        assert not batch.is_active
        assert batch.wait_all() == ()
        with cufile.RegisteredBuffer(source, 2 * BLOCK):
            pass
    assert file_path.read_bytes() == expected[:BLOCK].tobytes()


@pytest.mark.parametrize("kind", ["stream", "device_buffer", "host_buffer"])
@pytest.mark.agent_authored(model="gpt-6")
def test_native_duplicate_registration_is_rejected(file_path, init_cuda, data, kind):
    from cuda.bindings import cufile as backend

    expected, source, destination = data
    with ExitStack() as cleanup:
        cleanup.enter_context(cufile.Driver())
        stream = init_cuda.create_stream()
        cleanup.callback(stream.close)
        if kind == "stream":
            pointer = int(stream.handle)
            arguments = (pointer, 0)
            register, deregister = backend.stream_register, backend.stream_deregister
        else:
            if kind == "host_buffer":
                destination = cleanup.enter_context(mmap.mmap(-1, source.nbytes))
                pointer = ctypes.addressof(ctypes.c_char.from_buffer(destination))
            else:
                pointer = destination.data.ptr
            arguments = (pointer, source.nbytes, 0)
            register, deregister = backend.buf_register, backend.buf_deregister
        register(*arguments)
        try:
            with pytest.raises(backend.cuFileError) as error:
                register(*arguments)
            assert int(error.value.status) == (5022 if kind == "stream" else 5023)
            with pytest.raises(cufile.CuFileError) as wrapper_error:
                if kind == "stream":
                    cufile.RegisteredStream(stream)
                else:
                    cufile.RegisteredBuffer(destination)
            assert wrapper_error.value.native_code == (5022 if kind == "stream" else 5023)
            assert isinstance(wrapper_error.value.__cause__, backend.cuFileError)
            with cufile.File(file_path, "w+") as file:
                assert file.write(cufile.IOParams(source)) == source.nbytes
                if kind == "stream":
                    assert _stream_bytes(file.read_async, destination, stream, synchronize=stream.sync) == source.nbytes
                else:
                    assert file.read(cufile.IOParams(destination)) == source.nbytes
            if kind == "host_buffer":
                assert destination[:] == expected.tobytes()
            else:
                np.testing.assert_array_equal(cp.asnumpy(destination), expected)
        finally:
            deregister(pointer)


@pytest.mark.parametrize(
    "mode", ["native_implicit", "core_implicit", "core_explicit", "multiple_drivers", "resources", "shutdown"]
)
@pytest.mark.agent_authored(model="gpt-6")
def test_native_driver_reference_counts(file_path, mode):
    file_path.write_bytes(bytes(BLOCK))
    code = textwrap.dedent("""
        import ctypes
        import json
        import os
        import sys
        from cuda.bindings import cufile as native
        from cuda.bindings._internal import cufile as loader
        from cuda.core import Device, cufile
        from cuda.core.cufile._runtime import runtime

        Device(0).set_current()
        # The current Python use_count binding discards the native long result.
        count = ctypes.CFUNCTYPE(ctypes.c_long)(loader._inspect_function_pointer("__cuFileUseCount"))
        counts = [count()]
        if sys.argv[1] == "native_implicit":
            fd = os.open(sys.argv[2], os.O_RDONLY | os.O_DIRECT)
            try:
                descriptor = native.Descr()
                descriptor.type = native.FileHandleType.OPAQUE_FD
                descriptor.handle.fd = fd
                descriptor.fs_ops = 0
                handle = native.handle_register(descriptor.ptr)
                counts.append(count())
                native.handle_deregister(handle)
                counts.append(count())
            finally:
                os.close(fd)
            native.driver_close()
            counts.append(count())
        elif sys.argv[1] == "core_implicit":
            original_open, original_close = native.driver_open, native.driver_close
            def unexpected_driver_call():
                raise AssertionError("File/runtime made an explicit native driver open/close call")
            native.driver_open = native.driver_close = unexpected_driver_call
            try:
                with cufile.File(sys.argv[2]) as file:
                    counts.append(count())
                    destination = bytearray(4096)
                    assert file.read(cufile.IOParams(destination)) == 4096
                    assert destination == bytes(4096)
                counts.append(count())
                runtime.shutdown()
                counts.append(count())
            finally:
                native.driver_open, native.driver_close = original_open, original_close
                original_close()
            counts.append(count())
        elif sys.argv[1] == "multiple_drivers":
            first = cufile.Driver()
            counts.append(count())
            second = cufile.Driver()
            counts.append(count())
            with cufile.File(sys.argv[2]) as file:
                counts.append(count())
                first.close()
                counts.append(count())
                assert file.read(cufile.IOParams(bytearray(4096))) == 4096
            counts.append(count())
            second.close()
            counts.append(count())
            second.close()
            counts.append(count())
        elif sys.argv[1] == "resources":
            native.set_parameter_bool(native.BoolConfigParameter.PROPERTIES_ALLOW_SYSTEM_MEMORY, True)
            scope = cufile.Driver()
            counts.append(count())
            original_open, original_close = native.driver_open, native.driver_close
            def unexpected_driver_call():
                raise AssertionError("A cuFile resource made an explicit driver open/close call")
            native.driver_open = native.driver_close = unexpected_driver_call
            stream = Device(0).create_stream()
            try:
                with (
                    cufile.File(sys.argv[2]) as file,
                    cufile.RegisteredBuffer(bytearray(4096)) as buffer,
                    cufile.RegisteredStream(stream) as registered_stream,
                    cufile.Batch(1) as batch,
                ):
                    counts.append(count())
                    assert file.read(cufile.IOParams(buffer)) == 4096
                    with cufile.IOParams(buffer) as params:
                        file.read_async(params, registered_stream)
                        stream.sync()
                        assert params.bytes_done == 4096
                    batch.submit([cufile.BatchIOParams(file, buffer, "read", 4096)])
                    assert batch.wait_all(timeout=5)[0].result() == 4096
                counts.append(count())
            finally:
                native.driver_open, native.driver_close = original_open, original_close
                scope.close()
                stream.close()
            counts.append(count())
        elif sys.argv[1] == "shutdown":
            scope = cufile.Driver()
            counts.append(count())
            file = cufile.File(sys.argv[2])
            counts.append(count())
            runtime.shutdown()
            assert scope.is_closed and file.is_closed
            counts.append(count())
        else:
            scope = cufile.Driver()
            try:
                counts.append(count())
                with cufile.File(sys.argv[2]):
                    counts.append(count())
                counts.append(count())
            finally:
                scope.close()
            counts.append(count())
            runtime.shutdown()
            counts.append(count())
        print(json.dumps(counts))
    """)
    completed = subprocess.run(  # noqa: S603 -- fixed interpreter, test source, and private test path
        [sys.executable, "-c", code, mode, str(file_path)], capture_output=True, text=True, check=True, timeout=20
    )
    expected = {
        "native_implicit": [0, 1, 1, 0],
        "core_implicit": [0, 1, 1, 1, 0],
        "core_explicit": [0, 1, 1, 1, 0, 0],
        "multiple_drivers": [0, 1, 2, 2, 1, 1, 0, 0],
        "resources": [0, 1, 1, 1, 0],
        "shutdown": [0, 1, 1, 0],
    }[mode]
    assert json.loads(completed.stdout) == expected


@pytest.mark.parametrize("stage", ["submission", "completion"])
@pytest.mark.agent_authored(model="gpt-6")
def test_async_failures_release_ownership(file_path, init_cuda, data, monkeypatch, stage):
    from cuda.bindings import cufile as backend

    expected, source, destination = data
    stream = init_cuda.create_stream()

    def fail_async(handle, ptr, size, file_offset, buffer_offset, result, raw_stream):
        if stage == "submission":
            raise cufile.CuFileError("injected submission failure")
        ctypes.c_ssize_t.from_address(result).value = -5030

    try:
        with cufile.File(file_path, "w+") as file:
            file.write(cufile.IOParams(source))
            with monkeypatch.context() as patch:
                patch.setattr(backend, "read_async", fail_async)
                with pytest.raises(cufile.CuFileError):
                    _stream_bytes(file.read_async, destination, stream, synchronize=stream.sync)
            assert _stream_bytes(file.read_async, destination, stream, synchronize=stream.sync) == destination.nbytes
        np.testing.assert_array_equal(cp.asnumpy(destination), expected)
    finally:
        stream.close()


@pytest.mark.parametrize("token", [0, 1, 2])
@pytest.mark.agent_authored(model="gpt-6")
def test_default_stream_tokens(file_path, init_cuda, data, token):
    expected, source, destination = data
    with cufile.File(file_path, "w+") as file:
        assert (
            _stream_bytes(
                file.write_async, source, token, synchronize=(lambda: handle_return(driver.cuStreamSynchronize(token)))
            )
            == source.nbytes
        )
        assert (
            _stream_bytes(
                file.read_async,
                destination,
                token,
                synchronize=(lambda: handle_return(driver.cuStreamSynchronize(token))),
            )
            == destination.nbytes
        )
    np.testing.assert_array_equal(cp.asnumpy(destination), expected)


@pytest.mark.agent_authored(model="gpt-6")
def test_foreign_stream_protocol(file_path, init_cuda, data):
    _, source, _ = data
    stream = init_cuda.create_stream()

    class ForeignStream:
        def __cuda_stream__(self):
            return 0, int(stream.handle)

    try:
        with cufile.File(file_path, "w+") as file:
            with pytest.warns(DeprecationWarning, match="stream protocol"):
                future = _submit_stream(file.write_async, source, ForeignStream())
            completion = stream.record()
            stream.close()
            try:
                completion.sync()
                assert future.bytes_done == source.nbytes
            finally:
                completion.close()
    finally:
        stream.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_readonly_cuda_interface_remains_readonly_when_registered(file_path, init_cuda, data):
    _, source, _ = data

    class ReadonlyBuffer:
        @property
        def __cuda_array_interface__(self):
            interface = source.__cuda_array_interface__.copy()
            interface["data"] = (source.data.ptr, True)
            return interface

    with cufile.File(file_path, "w+") as file, cufile.RegisteredBuffer(ReadonlyBuffer()) as registered:
        assert file.write(cufile.IOParams(registered)) == source.nbytes
        with pytest.raises(ValueError):
            file.read(cufile.IOParams(registered))


class _SimulatedBatchDriver:
    """Inject completions at the binding boundary using its generated dtypes."""

    def __init__(self, monkeypatch):
        from cuda.bindings import cufile as backend

        self.backend = backend
        self.tokens = []
        self.ready = []
        monkeypatch.setattr(backend, "get_version", lambda: 1180)
        monkeypatch.setattr(backend, "batch_io_submit", self.submit)
        monkeypatch.setattr(backend, "batch_io_get_status", self.poll)

    @staticmethod
    def array(ptr, dtype, count):
        storage = (ctypes.c_ubyte * (dtype.itemsize * count)).from_address(ptr)
        return np.ndarray((count,), dtype=dtype, buffer=storage)

    def submit(self, handle, count, parameters, flags):
        array = self.array(parameters, self.backend.io_params_dtype, count)
        self.tokens.extend(int(value) for value in array["cookie"])

    def poll(self, handle, minimum, count_ptr, events_ptr, timeout):
        count = ctypes.c_uint.from_address(count_ptr)
        returned = self.ready[: count.value]
        del self.ready[: count.value]
        events = self.array(events_ptr, self.backend.io_events_dtype, count.value)
        for index, (operation, status, result) in enumerate(returned):
            events[index]["cookie"] = self.tokens[operation]
            events[index]["status"] = status
            events[index]["ret"] = ctypes.c_size_t(result).value
        count.value = len(returned)


@pytest.mark.parametrize("status_name", ["COMPLETE", "CANCELED", "FAILED", "INVALID", "TIMEOUT"])
@pytest.mark.agent_authored(model="gpt-6")
def test_batch_completion_preserves_native_bindings_status(file_path, init_cuda, data, monkeypatch, status_name):
    _, source, _ = data
    simulator = _SimulatedBatchDriver(monkeypatch)
    native_status = getattr(simulator.backend.Status, status_name)
    with cufile.File(file_path, "w+") as file, cufile.Batch(1) as batch:
        batch.submit([cufile.BatchIOParams(file, source, "write", BLOCK)])
        simulator.ready = [(0, int(native_status), BLOCK if status_name == "COMPLETE" else 0)]
        completion = batch.poll()[0]
        assert completion.status is native_status
        assert int(completion.status) == int(native_status)
        assert not isinstance(completion.status, str)
        if native_status == simulator.backend.Status.COMPLETE:
            assert completion.result() == BLOCK
        else:
            with pytest.raises(cufile.CuFileError) as caught:
                completion.result()
            assert caught.value is completion.error
            assert native_status.name in str(caught.value)

        assert batch.wait_all() == ()
        batch.submit([cufile.BatchIOParams(file, source, "write", BLOCK)])
        simulator.ready = [(1, int(simulator.backend.Status.COMPLETE), BLOCK)]
        assert batch.wait_all()[0].status is simulator.backend.Status.COMPLETE
        assert completion.status is native_status


@pytest.mark.agent_authored(model="gpt-6")
def test_batch_timeout_retention_out_of_order_polling_and_reuse(file_path, init_cuda, data, monkeypatch):
    _, source, _ = data
    simulator = _SimulatedBatchDriver(monkeypatch)
    with cufile.File(file_path, "w+") as file, cufile.Batch(3) as batch:
        batch.submit(
            [cufile.BatchIOParams(file, source, "write", BLOCK, file_offset=i * BLOCK, cookie=i) for i in range(3)]
        )
        with pytest.raises(TimeoutError):
            batch.wait_all(timeout=0)
        assert batch.is_active
        simulator.ready = [(2, 16, BLOCK), (0, 16, BLOCK), (1, 16, BLOCK)]
        first = batch.poll(max_complete=1)
        assert first[0].cookie == 2
        completions = batch.wait_all(timeout=1)
        assert [item.cookie for item in completions] == [0, 1]
        batch.submit([cufile.BatchIOParams(file, source, "write", BLOCK) for _ in range(3)])
        simulator.ready = [(4, 16, BLOCK), (5, 16, BLOCK), (3, 16, BLOCK)]
        assert [item.result() for item in batch.wait_all(timeout=1)] == [BLOCK] * 3
        assert first[0].cookie == 2


@pytest.mark.parametrize("observe", [True, False])
@pytest.mark.parametrize("native_result", [-1, -5030])
@pytest.mark.agent_authored(model="gpt-6")
def test_batch_signed_errors_are_reported_and_cleanup_completes(
    file_path, init_cuda, data, monkeypatch, observe, native_result
):
    _, source, _ = data
    simulator = _SimulatedBatchDriver(monkeypatch)
    with cufile.File(file_path, "w+") as file:
        batch = cufile.Batch(1)
        try:
            batch.submit([cufile.BatchIOParams(file, source, "write", BLOCK)])
            simulator.ready = [(0, 64, native_result)]
            if observe:
                with pytest.raises(cufile.CuFileError) as error:
                    batch.wait_all()
                assert error.value.native_code == (None if native_result == -1 else 5030)
                assert error.value.native_result == native_result
                assert error.value.operation == "batch[0]"
                if native_result == -1:
                    assert type(error.value) is cufile.CuFileError
                    assert not isinstance(error.value, OSError)
                    assert error.value.errno is None
                    assert error.value.strerror is None
                batch.close()
            else:
                completion = batch.poll()[0]
                assert completion.error.native_code == (None if native_result == -1 else 5030)
                assert completion.native_result == native_result
                assert completion.error.operation == "batch[0]"
                with pytest.raises(cufile.CuFileError):
                    completion.result()
                assert batch.wait_all() == ()
                batch.close()
            assert batch.is_closed
        finally:
            batch.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_native_batch_cancellation_is_drained(file_path, init_cuda, data):
    _, source, _ = data
    with cufile.File(file_path, "w+") as file, cufile.Batch(8) as batch:
        batch.submit([cufile.BatchIOParams(file, source, "write", BLOCK, file_offset=i * BLOCK) for i in range(8)])
        from cuda.bindings import cufile as backend

        if backend.get_version() == 1190:
            with pytest.raises(NotImplementedError):
                batch.cancel()
        else:
            batch.cancel()
        try:
            completions = batch.wait_all(timeout=10)
        except cufile.CuFileError:
            assert not batch.is_active
        else:
            assert len(completions) == 8
            assert all(item.result() == BLOCK for item in completions)


@pytest.mark.agent_authored(model="gpt-6")
def test_cancel_and_close_timeout_retain_owners(file_path, init_cuda, data, monkeypatch):
    _, source, _ = data
    simulator = _SimulatedBatchDriver(monkeypatch)
    monkeypatch.setattr(simulator.backend, "batch_io_cancel", lambda _handle: None)
    monkeypatch.setattr(simulator.backend, "get_version", lambda: 1180)
    with cufile.File(file_path, "w+") as file, cufile.Batch(1) as batch:
        batch.submit([cufile.BatchIOParams(file, source, "write", BLOCK, cookie="cancel")])
        batch.cancel()
        with pytest.raises(TimeoutError):
            batch.close(timeout=0)
        assert batch.is_active and not batch.is_closed
        simulator.ready = [(0, 8, 0)]
        completion = batch.poll()[0]
        assert completion.status is simulator.backend.Status.CANCELED
        with pytest.raises(cufile.CuFileError):
            completion.result()
        assert batch.wait_all() == ()
        assert not batch.is_active


@pytest.mark.agent_authored(model="gpt-6")
def test_batch_destroy_failure_keeps_queue_available_for_retry(file_path, init_cuda, data, monkeypatch):
    from cuda.bindings import cufile as backend

    expected, source, _ = data
    with cufile.File(file_path, "w+") as file, cufile.Batch(1) as batch:
        batch.submit([cufile.BatchIOParams(file, source, "write", source.nbytes)])
        assert batch.wait_all(timeout=10)[0].result() == source.nbytes

        def fail_destroy(_handle):
            raise cufile.CuFileError("injected destroy failure")

        with monkeypatch.context() as patch:
            patch.setattr(backend, "batch_io_destroy", fail_destroy)
            with pytest.raises(cufile.CuFileError):
                batch.close()
        assert not batch.is_closed
        batch.submit([cufile.BatchIOParams(file, source, "write", source.nbytes)])
        assert batch.wait_all(timeout=10)[0].result() == source.nbytes
    assert file_path.read_bytes() == expected.tobytes()


@pytest.mark.agent_authored(model="gpt-6")
def test_batch_consumed_completions_survive_context_restore_error(file_path, init_cuda, data, monkeypatch):
    from cuda.core import CUDAError
    from cuda.core._rt import _set_context_restore_fault_for_testing

    _, source, _ = data
    simulator = _SimulatedBatchDriver(monkeypatch)
    with cufile.File(file_path, "w+") as file, cufile.Batch(1) as batch:
        batch.submit([cufile.BatchIOParams(file, source, "write", BLOCK, cookie="kept")])
        simulator.ready = [(0, 16, BLOCK)]

        def inject_after_poll(*args):
            simulator.poll(*args)
            _set_context_restore_fault_for_testing(int(driver.CUresult.CUDA_ERROR_INVALID_CONTEXT))

        previous = handle_return(driver.cuCtxPopCurrent())
        try:
            with monkeypatch.context() as patch:
                patch.setattr(simulator.backend, "batch_io_get_status", inject_after_poll)
                with pytest.raises(CUDAError) as caught:
                    batch.poll()
        finally:
            _set_context_restore_fault_for_testing(0)
            handle_return(driver.cuCtxSetCurrent(0))
            handle_return(driver.cuCtxPushCurrent(previous))
        assert not batch.is_active
        assert caught.value.completions[0].cookie == "kept"
        assert batch.wait_all() == ()


@pytest.mark.agent_authored(model="gpt-6")
def test_other_current_device_is_restored(file_path, init_cuda, data):
    from cuda.core import Device

    if handle_return(driver.cuDeviceGetCount()) < 2:
        pytest.skip("Requires two CUDA devices")
    _, source, destination = data
    original = handle_return(driver.cuCtxGetCurrent())
    try:
        Device(1).set_current()
        other_context = handle_return(driver.cuCtxGetCurrent())
        with cufile.File(file_path, "w+") as file:
            assert file.write(cufile.IOParams(source)) == source.nbytes
            assert file.read(cufile.IOParams(destination)) == destination.nbytes
        assert handle_return(driver.cuCtxGetCurrent()) == other_context
    finally:
        handle_return(driver.cuCtxSetCurrent(original))


@pytest.mark.thread_unsafe(reason="injects a thread-local context restoration failure")
@pytest.mark.parametrize("operation", ["register", "async_write"])
@pytest.mark.agent_authored(model="gpt-6")
def test_context_restore_failure_rolls_back_or_drains(file_path, init_cuda, data, monkeypatch, operation):
    from cuda.bindings import cufile as backend
    from cuda.core import CUDAError
    from cuda.core._rt import _set_context_restore_fault_for_testing

    expected, source, _ = data
    stream = init_cuda.create_stream()
    file = cufile.File(file_path, "w+")
    original = backend.buf_register if operation == "register" else backend.write_async

    def inject_after_success(*args):
        original(*args)
        _set_context_restore_fault_for_testing(int(driver.CUresult.CUDA_ERROR_INVALID_CONTEXT))

    previous = handle_return(driver.cuCtxPopCurrent())
    try:
        with monkeypatch.context() as patch:
            patch.setattr(backend, "buf_register" if operation == "register" else "write_async", inject_after_success)
            with pytest.raises(CUDAError):
                if operation == "register":
                    cufile.RegisteredBuffer(source)
                else:
                    _submit_stream(file.write_async, source, stream)
    finally:
        _set_context_restore_fault_for_testing(0)
        handle_return(driver.cuCtxSetCurrent(0))
        handle_return(driver.cuCtxPushCurrent(previous))
        file.close()
        stream.close()
    if operation == "register":
        with cufile.RegisteredBuffer(source):
            pass
    else:
        assert file_path.read_bytes() == expected.tobytes()


@pytest.mark.agent_authored(model="gpt-6")
def test_body_exception_survives_file_cleanup_failure(file_path, init_cuda, monkeypatch):
    from cuda.bindings import cufile as backend

    file = cufile.File(file_path, "w+")

    def fail_deregister(*args):
        raise cufile.CuFileError("injected cleanup failure")

    with monkeypatch.context() as patch:
        patch.setattr(backend, "handle_deregister", fail_deregister)
        with pytest.raises(LookupError) as error, file:
            raise LookupError("application failure")
    assert error.value.__notes__
    file.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_dropped_file_closes_descriptors(file_path, init_cuda, data):
    _, source, _ = data
    stream = init_cuda.create_stream()
    try:
        file = cufile.File(file_path, "w+")
        descriptor = file.fileno()
        request = _submit_stream(file.write_async, source, stream)
        del file
        gc.collect()
        stream.sync()
        assert request.bytes_done == source.nbytes
        with pytest.raises(OSError):
            os.fstat(descriptor)
    finally:
        stream.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_unaligned_sync_transfers(file_path, init_cuda, data):
    expected, source, destination = data
    with cufile.File(file_path, "w+") as file:
        assert file.write(cufile.IOParams(source, size=137, file_offset=19)) == 137
        assert file.read(cufile.IOParams(destination, size=137, file_offset=19)) == 137
    np.testing.assert_array_equal(cp.asnumpy(destination[:137]), expected[:137])


@pytest.mark.agent_authored(model="gpt-6")
def test_filesystem_failure_retains_errno(storage, init_cuda):
    missing = storage / "missing-parent" / "missing-file.bin"
    with pytest.raises(cufile.CuFileError) as error:
        cufile.File(missing)
    assert type(error.value) is cufile.CuFileError
    assert not isinstance(error.value, OSError)
    assert error.value.errno == errno.ENOENT
    assert error.value.filename == str(missing)
    assert error.value.strerror == os.strerror(errno.ENOENT)
    assert error.value.operation == "open"
    assert error.value.native_code is None
    assert isinstance(error.value.__cause__, FileNotFoundError)


@pytest.mark.agent_authored(model="gpt-6")
def test_core_batch_application(storage, init_cuda):
    application = runpy.run_path(str(pathlib.Path(__file__).parents[1] / "examples" / "cufile_batch.py"))
    assert application["run_application"](storage) == {
        "verified": True,
        "files": 2,
        "bytes_read": BLOCK * 2,
        "bytes_written": BLOCK * 2,
    }


@pytest.mark.parametrize("mode", ["r", "r+", "w", "w+"])
@pytest.mark.agent_authored(model="gpt-6")
def test_file_modes(file_path, init_cuda, mode):
    payload = bytes(range(256)) * 16
    file_path.write_bytes(payload)
    with cufile.File(file_path, mode=mode) as file:
        assert os.fstat(file.fileno()).st_size == (0 if mode.startswith("w") else BLOCK)
        if "w" in mode or "+" in mode:
            assert file.write(cufile.IOParams(payload)) == BLOCK
        if mode.startswith("r") or "+" in mode:
            destination = bytearray(BLOCK)
            assert file.read(cufile.IOParams(destination)) == BLOCK
            assert destination == payload


@pytest.mark.parametrize("mode", ["stream", "batch"])
@pytest.mark.parametrize("explicit_driver", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_process_shutdown_drains_unclosed_work(file_path, mode, explicit_driver):
    program = textwrap.dedent("""
        import sys
        import cupy as cp
        from cuda.core import Device, cufile

        Device(0).set_current()
        if sys.argv[3] == "explicit":
            driver = cufile.Driver()
        source = cp.arange(4096, dtype=cp.uint8)
        cp.cuda.Stream.null.synchronize()
        file = cufile.File(sys.argv[1], "w+")
        if sys.argv[2] == "stream":
            stream = Device(0).create_stream()
            request = cufile.IOParams(source)
            file.write_async(request, stream)
        else:
            request = cufile.Batch(1)
            request.submit([cufile.BatchIOParams(file, source, "write", 4096)])
    """)
    result = subprocess.run(  # noqa: S603 -- fixed test program and local temporary file
        [sys.executable, "-c", program, str(file_path), mode, "explicit" if explicit_driver else "implicit"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "CUDAWarning" not in result.stderr
    assert file_path.read_bytes() == bytes(range(256)) * 16


@pytest.mark.agent_authored(model="gpt-6")
def test_forked_child_rejected_before_native_access(file_path, init_cuda):
    program = textwrap.dedent("""
        import os
        import sys
        from cuda.core import Device, cufile

        Device(0).set_current()
        with cufile.File(sys.argv[1], "w+"):
            child = os.fork()
            if child == 0:
                try:
                    cufile.Driver()
                except RuntimeError:
                    os._exit(0)
                os._exit(1)
            _, status = os.waitpid(child, 0)
            assert os.waitstatus_to_exitcode(status) == 0
    """)
    result = subprocess.run(  # noqa: S603 -- fixed test program and local temporary file
        [sys.executable, "-c", program, str(file_path)], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_default_size_uses_remaining_buffer(file_path, init_cuda, data, asynchronous):
    expected, source, destination = data
    stream = init_cuda.create_stream()
    try:
        with cufile.File(file_path, "w+") as file:
            if asynchronous:
                assert (
                    _stream_bytes(file.write_async, source, stream, synchronize=stream.sync, buffer_offset=BLOCK)
                    == source.nbytes - BLOCK
                )
                assert (
                    _stream_bytes(file.read_async, destination, stream, synchronize=stream.sync, buffer_offset=BLOCK)
                    == destination.nbytes - BLOCK
                )
            else:
                assert file.write(cufile.IOParams(source, buffer_offset=BLOCK)) == source.nbytes - BLOCK
                assert file.read(cufile.IOParams(destination, buffer_offset=BLOCK)) == destination.nbytes - BLOCK
        np.testing.assert_array_equal(cp.asnumpy(destination[BLOCK:]), expected[BLOCK:])
    finally:
        stream.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_recording_completion_failure_drains_accepted_io(file_path, init_cuda, data, monkeypatch):
    from cuda.core.cufile import _file

    expected, source, _ = data
    stream = init_cuda.create_stream()

    def fail_record(_stream):
        raise LookupError("injected event recording failure")

    try:
        with cufile.File(file_path, "w+") as file, monkeypatch.context() as patch:
            patch.setattr(_file, "_record_completion", fail_record)
            with pytest.raises(LookupError):
                _submit_stream(file.write_async, source, stream)
        assert file_path.read_bytes() == expected.tobytes()
    finally:
        stream.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_stream_result_access_excludes_later_stream_work(file_path, init_cuda, data):
    _, source, _ = data
    stream = init_cuda.create_stream()
    release, entered, finished = threading.Event(), threading.Event(), threading.Event()
    outcome = []

    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def later_work(_argument):
        entered.set()
        release.wait(timeout=10)

    def read_result(request):
        try:
            outcome.append(request.bytes_done)
        finally:
            finished.set()

    waiter = None
    try:
        with cufile.File(file_path, "w+") as file:
            request = _submit_stream(file.write_async, source, stream)
            completion = stream.record()
            handle_return(
                driver.cuLaunchHostFunc(int(stream.handle), ctypes.cast(later_work, ctypes.c_void_p).value, 0)
            )
            try:
                assert entered.wait(timeout=5)
                completion.sync()
                waiter = threading.Thread(target=read_result, args=(request,))
                waiter.start()
                assert finished.wait(timeout=2)
                assert outcome == [source.nbytes]
            finally:
                release.set()
                if waiter is not None:
                    waiter.join(timeout=5)
                stream.sync()
                completion.close()
    finally:
        release.set()
        stream.sync()
        stream.close()


@pytest.mark.parametrize("operation", ["open", "close"])
@pytest.mark.agent_authored(model="gpt-6")
def test_driver_failure_preserves_reference_balance(init_cuda, monkeypatch, operation):
    from cuda.bindings import cufile as backend
    from cuda.bindings._internal import cufile as loader

    count = ctypes.CFUNCTYPE(ctypes.c_long)(loader._inspect_function_pointer("__cuFileUseCount"))
    before = count()

    def fail():
        raise cufile.CuFileError("injected driver failure")

    if operation == "open":
        with monkeypatch.context() as patch:
            patch.setattr(backend, "driver_open", fail)
            with pytest.raises(cufile.CuFileError, match="injected driver failure"):
                cufile.Driver()
            gc.collect()
            assert count() == before
    else:
        scope = cufile.Driver()
        try:
            with monkeypatch.context() as patch:
                patch.setattr(backend, "driver_close", fail)
                with pytest.raises(cufile.CuFileError, match="injected driver failure"):
                    scope.close()
                assert not scope.is_closed
                assert count() == before + 1
            scope.close()
            assert scope.is_closed
            assert count() == before
        finally:
            scope.close()


@pytest.mark.parametrize("invalid", ["size", "file_offset", "buffer_offset"])
@pytest.mark.agent_authored(model="gpt-6")
def test_stream_alignment_promise_checked_before_submission(file_path, init_cuda, data, invalid):
    _, source, _ = data
    stream = init_cuda.create_stream()
    arguments = {"size": BLOCK, "file_offset": 0, "buffer_offset": 0}
    arguments[invalid] += 1
    try:
        with cufile.File(file_path, "w+") as file, cufile.RegisteredStream(stream, 8) as registered:
            with pytest.raises(ValueError):
                _submit_stream(file.write_async, source, registered, **arguments)
            assert _stream_bytes(file.write_async, source, registered, BLOCK, synchronize=stream.sync) == BLOCK
    finally:
        stream.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_native_file_application(storage, init_cuda):
    application = runpy.run_path(str(pathlib.Path(__file__).parents[1] / "examples" / "cufile_io.py"))
    result = application["run_application"](storage, 1024 * 1024, 1)
    assert set(result) == {"sync", "stream"}
    assert all(value["verified"] for value in result.values())
    assert len({value["sha256"] for value in result.values()}) == 1


@pytest.mark.agent_authored(model="gpt-6")
def test_stream_params_return_results_without_waiting(file_path, init_cuda, data, monkeypatch):
    from cuda.core.cufile import _file

    expected, source, destination = data
    stream = init_cuda.create_stream()
    try:
        with (
            cufile.File(file_path, "w+") as file,
            cufile.IOParams(source) as write_params,
            cufile.IOParams(destination) as read_params,
        ):
            assert not hasattr(write_params, "wait")
            with pytest.raises(RuntimeError):
                _ = write_params.bytes_done
            assert file.write_async(write_params, stream) is None
            assert file.read_async(read_params, stream) is None
            stream.sync()

            def unexpected_wait(_work):
                raise AssertionError("Result access must not wait")

            with monkeypatch.context() as patch:
                patch.setattr(_file._Work, "wait", unexpected_wait)
                assert write_params.bytes_done == source.nbytes
                assert read_params.bytes_done == destination.nbytes
                assert read_params.bytes_done == destination.nbytes
        np.testing.assert_array_equal(cp.asnumpy(destination), expected)
        assert read_params.bytes_done == destination.nbytes
    finally:
        stream.sync()
        stream.close()


@pytest.mark.parametrize("operation", ["read", "write"])
@pytest.mark.parametrize("native_result", [-1, -5030])
@pytest.mark.agent_authored(model="gpt-6")
def test_stream_errors_preserve_metadata_after_completion_and_close(
    file_path, init_cuda, data, monkeypatch, operation, native_result
):
    from cuda.bindings import cufile as bindings

    _, source, _ = data
    stream = init_cuda.create_stream()

    def deferred_failure(handle, buffer, size, file_offset, buffer_offset, result, raw_stream):
        ctypes.c_ssize_t.from_address(result).value = native_result

    try:
        with cufile.File(file_path, "w+") as file:
            params = cufile.IOParams(source, BLOCK)
            with monkeypatch.context() as patch:
                patch.setattr(bindings, f"{operation}_async", deferred_failure)
                getattr(file, f"{operation}_async")(params, stream)
            stream.sync()
            with pytest.raises(cufile.CuFileError) as caught:
                _ = params.bytes_done
            error = caught.value
            assert error.operation == f"{operation}_async"
            assert error.native_result == native_result
            if native_result == -1:
                assert type(error) is cufile.CuFileError
                assert not isinstance(error, OSError)
                assert error.errno is None
                assert error.strerror is None
                assert error.status is None
            else:
                assert error.status_name == "INTERNAL_ERROR"
                assert error.native_code == 5030
            params.close()
            with pytest.raises(cufile.CuFileError) as repeated:
                _ = params.bytes_done
            assert repeated.value is error
    finally:
        stream.close()


@pytest.mark.parametrize("memory", ["host", "device"])
@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_stream_params_deferred_values_and_pending_result(file_path, init_cuda, memory, registered):
    payload = b"a" * BLOCK + b"b" * BLOCK + b"c" * BLOCK
    source = bytearray(payload) if memory == "host" else cp.asarray(np.frombuffer(payload, dtype=np.uint8))
    cp.cuda.get_current_stream().synchronize()
    stream = init_cuda.create_stream()
    entered, release = threading.Event(), threading.Event()

    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def prior_work(_argument):
        entered.set()
        release.wait(timeout=10)

    try:
        with ExitStack() as resources:
            file = resources.enter_context(cufile.File(file_path, "w+"))
            io_stream = resources.enter_context(cufile.RegisteredStream(stream))
            buffer = resources.enter_context(cufile.RegisteredBuffer(source)) if registered else source
            params = resources.enter_context(cufile.IOParams(buffer, BLOCK, buffer_offset=0 if registered else BLOCK))
            handle_return(
                driver.cuLaunchHostFunc(int(stream.handle), ctypes.cast(prior_work, ctypes.c_void_p).value, 0)
            )
            try:
                assert entered.wait(timeout=5)
                file.write_async(params, io_stream)
                with pytest.raises(RuntimeError):
                    _ = params.bytes_done
                with pytest.raises(RuntimeError):
                    file.write_async(params, io_stream)
                with pytest.raises(RuntimeError):
                    file.write(params)
                with pytest.raises(RuntimeError):
                    file.read(params)
                params.size = BLOCK // 2
                params.file_offset = BLOCK
                if registered:
                    params.buffer_offset = BLOCK
                else:
                    with pytest.raises(ValueError):
                        params.buffer_offset = 2 * BLOCK
                with pytest.raises(ValueError):
                    params.size = BLOCK + 1
                with pytest.raises(ValueError):
                    params.size = None
                with pytest.raises(ValueError):
                    params.buffer_offset = len(payload)
            finally:
                release.set()
            stream.sync()
            assert params.bytes_done == BLOCK // 2
        assert file_path.read_bytes() == bytes(BLOCK) + b"b" * (BLOCK // 2)
    finally:
        release.set()
        stream.sync()
        stream.close()


@pytest.mark.parametrize("flag,field", [(1, "buffer_offset"), (2, "file_offset"), (4, "size")])
@pytest.mark.agent_authored(model="gpt-6")
def test_stream_params_respect_fixed_registration_arguments(file_path, init_cuda, flag, field):
    stream = init_cuda.create_stream()
    release = threading.Event()

    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def prior_work(_argument):
        release.wait(timeout=10)

    try:
        with cufile.File(file_path, "w+") as file, cufile.RegisteredStream(stream, flag) as registered:
            params = cufile.IOParams(bytearray(BLOCK), BLOCK)
            handle_return(
                driver.cuLaunchHostFunc(int(stream.handle), ctypes.cast(prior_work, ctypes.c_void_p).value, 0)
            )
            try:
                file.write_async(params, registered)
                with pytest.raises(ValueError):
                    setattr(params, field, getattr(params, field))
            finally:
                release.set()
            stream.sync()
            assert params.bytes_done == BLOCK
    finally:
        release.set()
        stream.sync()
        stream.close()


@pytest.mark.agent_authored(model="gpt-6")
def test_stream_params_reuse_and_short_result(file_path, init_cuda):
    source = bytearray(b"x" * BLOCK)
    destination = bytearray(2 * BLOCK)
    stream = init_cuda.create_stream()
    try:
        with cufile.File(file_path, "w+") as file, cufile.IOParams(source) as params:
            file.write_async(params, stream)
            stream.sync()
            assert params.bytes_done == BLOCK
            params.size = None
            params.file_offset = BLOCK
            file.write_async(params, stream)
            stream.sync()
            assert params.bytes_done == BLOCK
            with cufile.IOParams(destination, file_offset=BLOCK) as read:
                file.read_async(read, stream)
                stream.sync()
                assert read.bytes_done == BLOCK
                assert destination == b"x" * BLOCK + bytes(BLOCK)
        assert file_path.read_bytes() == b"x" * (2 * BLOCK)
        with pytest.raises(RuntimeError):
            params.size = 0
    finally:
        stream.sync()
        stream.close()


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_rolling_batch_reuses_completed_host_and_device_slots(storage, init_cuda, registered):
    from dataclasses import replace

    from cuda.bindings import cufile as backend

    if backend.get_version() == 1190:
        pytest.skip("Native 1.19 partial-resubmission corruption; wrapper rejection tested separately")
    block, count = BLOCK, 16
    expected = [np.full(block, index, dtype=np.uint8) for index in range(count)]
    paths = [storage / "rolling-a.bin", storage / "rolling-b.bin"]
    for path in paths:
        path.write_bytes(b"".join(item.tobytes() for item in expected))
    buffers = [np.empty(block, dtype=np.uint8), cp.empty(block, dtype=cp.uint8)]
    with ExitStack() as resources:
        files = [resources.enter_context(cufile.File(path)) for path in paths]
        arguments = (
            [resources.enter_context(cufile.RegisteredBuffer(buf)) for buf in buffers] if registered else buffers
        )
        batch = resources.enter_context(cufile.Batch(2))
        entries = [
            cufile.BatchIOParams(file, buf, "read", block, file_offset=index * block, cookie=(index, index))
            for index, (file, buf) in enumerate(zip(files, arguments))
        ]
        batch.submit(entries)
        next_job, received, refills = 2, set(), 0
        while batch.is_active:
            completions = batch.poll(min_complete=1, max_complete=1, timeout=10)
            assert len(completions) == 1
            completion = completions[0]
            slot, job = completion.cookie
            assert completion.index not in received
            received.add(completion.index)
            assert completion.result() == block
            actual = buffers[slot] if slot == 0 else cp.asnumpy(buffers[slot])
            np.testing.assert_array_equal(actual, expected[job])
            if next_job < count:
                assert batch.is_active
                refills += 1
                entries[slot] = replace(entries[slot], file_offset=next_job * block, cookie=(slot, next_job))
                batch.submit([entries[slot]])
                next_job += 1
        assert refills == count - 2
        assert received == set(range(count))
        assert batch.wait_all() == ()


@pytest.mark.agent_authored(model="gpt-6")
def test_rolling_batch_validation_preserves_existing_requests(file_path, init_cuda, monkeypatch):
    simulator = _SimulatedBatchDriver(monkeypatch)
    first, second, rollback = bytearray(BLOCK), bytearray(BLOCK), bytearray(BLOCK)
    with cufile.File(file_path, "w+") as file, cufile.Batch(3) as batch:
        batch.submit([cufile.BatchIOParams(file, first, "write", BLOCK, cookie="first")])
        with pytest.raises(TypeError, match="BatchIOParams"):
            batch.submit([cufile.BatchIOParams(file, rollback, "write", BLOCK), object()])
        rollback.extend(b"x")
        with pytest.raises(BufferError):
            first.extend(b"x")
        with pytest.raises(ValueError, match="capacity"):
            batch.submit([])
        with pytest.raises(ValueError, match="capacity"):
            batch.submit([cufile.BatchIOParams(file, second, "write", BLOCK)] * 3)
        with pytest.raises(ValueError):
            cufile.BatchIOParams(file, first, "write", 0)
        batch.submit([cufile.BatchIOParams(file, second, "write", BLOCK, cookie="second")])
        simulator.ready = [(0, 16, BLOCK)]
        assert batch.poll()[0].cookie == "first"
        first.extend(b"x")
        with pytest.raises(BufferError):
            second.extend(b"x")
        simulator.ready = [(1, 16, BLOCK)]
        assert [item.cookie for item in batch.wait_all()] == ["second"]
        second.extend(b"x")
        assert not any(hasattr(batch, name) for name in ("prepare", "clear", "wait"))


@pytest.mark.agent_authored(model="gpt-6")
def test_batch_partial_timeout_exposes_results_without_replay(file_path, init_cuda, monkeypatch):
    simulator = _SimulatedBatchDriver(monkeypatch)
    buffers = [bytearray(BLOCK), bytearray(BLOCK)]
    with cufile.File(file_path, "w+") as file, cufile.Batch(2) as batch:
        batch.submit([cufile.BatchIOParams(file, buf, "write", BLOCK, cookie=i) for i, buf in enumerate(buffers)])
        simulator.ready = [(1, 16, BLOCK)]
        with pytest.raises(TimeoutError) as caught:
            batch.wait_all(timeout=0)
        assert [item.cookie for item in caught.value.completions] == [1]
        assert caught.value.completions[0].result() == BLOCK
        assert batch.is_active
        simulator.ready = [(0, 16, BLOCK)]
        assert [item.cookie for item in batch.wait_all()] == [0]
        assert batch.wait_all() == ()


@pytest.mark.agent_authored(model="gpt-6")
def test_batch_wait_failure_exposes_all_new_results(file_path, init_cuda, monkeypatch):
    simulator = _SimulatedBatchDriver(monkeypatch)
    with cufile.File(file_path, "w+") as file, cufile.Batch(3) as batch:
        batch.submit([cufile.BatchIOParams(file, bytearray(BLOCK), "write", BLOCK, cookie=i) for i in range(3)])
        simulator.ready = [(2, 16, BLOCK)]
        earlier = batch.poll()
        simulator.ready = [(1, 64, -5030), (0, 16, BLOCK)]
        with pytest.raises(cufile.CuFileError) as caught:
            batch.wait_all()
        assert [item.cookie for item in caught.value.completions] == [0, 1]
        assert caught.value.completions[1].error is caught.value
        assert earlier[0].cookie == 2
        assert batch.wait_all() == ()


@pytest.mark.agent_authored(model="gpt-6")
def test_core_rolling_batch_application(storage, init_cuda):
    application = runpy.run_path(str(pathlib.Path(__file__).parents[1] / "examples" / "cufile_rolling_batch.py"))
    from cuda.bindings import cufile as backend

    if backend.get_version() == 1190:
        with pytest.raises(NotImplementedError, match="Rolling batch"):
            application["run_application"](storage)
        return
    assert application["run_application"](storage) == {
        "verified": True,
        "transfers": 12,
        "bytes_read": 12 * BLOCK,
        "rolling_refills": 10,
    }


@pytest.mark.agent_authored(model="gpt-6")
def test_rolling_batch_simulated_replacements_and_capacity(file_path, init_cuda, monkeypatch):
    from dataclasses import replace

    simulator = _SimulatedBatchDriver(monkeypatch)
    with cufile.File(file_path, "w+") as file, cufile.Batch(2) as batch:
        buffers = [bytearray(BLOCK), bytearray(BLOCK)]
        entries = [cufile.BatchIOParams(file, buf, "write", BLOCK, cookie=i) for i, buf in enumerate(buffers)]
        batch.submit(entries)
        received = []
        for job in range(1, 12):
            simulator.ready = [(job, 16, BLOCK)]
            current = batch.poll(max_complete=1)
            received.extend(current)
            assert batch.is_active
            assert current[0].cookie == job
            with pytest.raises(ValueError, match="capacity"):
                batch.submit(entries)
            entries[1] = replace(entries[1], file_offset=(job + 1) * BLOCK, cookie=job + 1)
            batch.submit([entries[1]])
        simulator.ready = [(12, 16, BLOCK), (0, 16, BLOCK)]
        remaining = batch.wait_all()
        assert [item.cookie for item in remaining] == [0, 12]
        assert {item.index for item in received + list(remaining)} == set(range(13))
        assert batch.wait_all() == ()


@pytest.mark.agent_authored(model="gpt-6")
def test_native_1190_rolling_guard_precedes_submission(file_path, init_cuda, monkeypatch):
    simulator = _SimulatedBatchDriver(monkeypatch)
    monkeypatch.setattr(simulator.backend, "get_version", lambda: 1190)
    with cufile.File(file_path, "w+") as file, cufile.Batch(2) as batch:
        first, second = bytearray(BLOCK), bytearray(BLOCK)
        batch.submit([cufile.BatchIOParams(file, first, "write", BLOCK)])
        with pytest.raises(NotImplementedError, match="partial resubmission"):
            batch.submit([cufile.BatchIOParams(file, second, "write", BLOCK)])
        assert len(simulator.tokens) == 1
        second.extend(b"x")
        simulator.ready = [(0, 16, BLOCK)]
        assert batch.wait_all()[0].result() == BLOCK
        batch.submit([cufile.BatchIOParams(file, second, "write", BLOCK)])
        simulator.ready = [(1, 16, BLOCK)]
        assert batch.wait_all()[0].result() == BLOCK

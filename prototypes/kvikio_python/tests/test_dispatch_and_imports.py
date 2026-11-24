# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
import threading

import pytest

import cufile_python as kvikio
from cufile_python._lib import file_handle


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.agent_authored(model="gpt-6")
def test_grouped_chunks_drain_all_work_and_preserve_bounds(tmp_path, monkeypatch, workers, fail):
    size, task_size = 7 * 4096 + 53, 4096
    source = bytes(i % 251 for i in range(size))
    path = tmp_path / "data"
    path.write_bytes(b"prefix" + source)
    destination = bytearray(b"!" * (size + 26))
    entered, release = threading.Event(), threading.Event()
    calls, lock = [], threading.Lock()
    original = file_handle.CuFile._transfer

    def transfer(self, arr, count, file_offset, offset, *args):
        with lock:
            calls.append((offset, count, file_offset))
        if offset == 7 * task_size:
            entered.set()
            assert release.wait(10)
        if fail and offset == 0:
            raise RuntimeError("injected first-chunk failure")
        return original(self, arr, count, file_offset, offset, *args)

    monkeypatch.setattr(file_handle.CuFile, "_transfer", transfer)
    with (
        kvikio.defaults.set({"num_threads": workers, "compat_mode": kvikio.CompatMode.ON}),
        kvikio.SummaryMonitor() as monitor,
        kvikio.CuFile(path) as file,
    ):
        pending = file.pread(memoryview(destination)[13:-13], file_offset=6, task_size=task_size)
        try:
            assert entered.wait(10)
            assert not pending.done()
        finally:
            release.set()
        if fail:
            with pytest.raises(RuntimeError, match="injected first-chunk failure"):
                pending.get()
        else:
            assert pending.get() == size
            assert destination[13:-13] == source
        assert monitor.get().num_ops == 1
        assert monitor.get().num_errors == int(fail)
    assert sorted(calls) == [
        (offset, min(task_size, size - offset), 6 + offset) for offset in range(0, size, task_size)
    ]
    assert destination[:13] == destination[-13:] == b"!" * 13


@pytest.mark.agent_authored(model="gpt-6")
def test_kvikio_compatibility_aliases_share_one_backend(tmp_path):
    code = """
import importlib
import sys
import cufile_python
from cufile_python.compat import enable_kvikio_compat
assert 'kvikio' not in sys.modules
enable_kvikio_compat()
enable_kvikio_compat()
import kvikio
import kvikio.defaults
from kvikio.cufile import CuFile
assert kvikio is cufile_python
assert CuFile is cufile_python.CuFile
for suffix in ('defaults', 'statistics', 'batch', '_lib.batch', '_lib.defaults', '_lib._runtime', '_lib.file_handle'):
    assert importlib.import_module('kvikio.' + suffix) is importlib.import_module('cufile_python.' + suffix)
with kvikio.defaults.set('num_threads', 3):
    assert cufile_python.defaults.get('num_threads') == 3
with CuFile(sys.argv[1], 'w+') as file:
    assert file.pwrite(b'alias').get() == 5
    target = bytearray(5)
    assert file.pread(target).get() == 5
    assert target == b'alias'
assert not hasattr(kvikio, 'RemoteFile')
assert cufile_python.__spec__.name == 'cufile_python'
assert kvikio.Batch is cufile_python.Batch
"""
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "data")],
        env=dict(os.environ, KVIKIO_COMPAT_MODE="ON"),
        check=True,
        timeout=30,
    )


@pytest.mark.agent_authored(model="gpt-6")
def test_canonical_import_coexists_and_alias_refuses_loaded_kvikio():
    code = """
import sys
import cufile_python
assert 'kvikio' not in sys.modules
import kvikio
assert kvikio is not cufile_python
assert kvikio.CuFile is not cufile_python.CuFile
from cufile_python.compat import enable_kvikio_compat
try:
    enable_kvikio_compat()
except RuntimeError as error:
    assert 'already imported' in str(error)
else:
    raise AssertionError('Existing KvikIO must not be replaced')
assert sys.modules['kvikio'] is kvikio
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys

import pytest


@pytest.mark.agent_authored(model="gpt-6")
def test_local_file_io_without_remote_dependencies(tmp_path):
    # A clean interpreter must not depend on optional network/storage packages.
    code = """
import importlib.abc
import importlib.util
import sys

class NoRemoteImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {
            'aiohttp', 'botocore', 'boto3', 'RangeHTTPServer', 'zarr', 'numcodecs',
            'http', 'urllib',
        }:
            raise AssertionError('Unexpected dependency: ' + fullname)

sys.meta_path.insert(0, NoRemoteImports())
import cufile_python as kvikio
import cufile_python.defaults

assert not hasattr(kvikio, 'RemoteFile')
assert not hasattr(kvikio, 'RemoteIOBackend')
assert importlib.util.find_spec('cufile_python.remote_file') is None
assert importlib.util.find_spec('cufile_python.zarr') is None
assert importlib.util.find_spec('cufile_python.benchmarks') is None
with kvikio.CuFile(sys.argv[1], 'w+') as file:
    payload = b'CuFile local I/O'
    assert file.pwrite(payload).get() == len(payload)
    destination = bytearray(len(payload))
    assert file.pread(destination).get() == len(destination)
    assert destination == payload
try:
    kvikio.defaults.get('remote_io_backend')
except KeyError:
    pass
else:
    raise AssertionError('Remote settings must be outside this subset')
"""
    env = dict(os.environ, KVIKIO_COMPAT_MODE="ON")
    subprocess.run([sys.executable, "-c", code, str(tmp_path / "data")], env=env, check=True, timeout=30)

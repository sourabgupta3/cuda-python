# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit import aliases for applications migrating their local KvikIO I/O."""

import importlib
import sys
import threading

_lock = threading.RLock()
_modules = (
    "",
    "._version",
    ".compat",
    ".buffer",
    ".batch",
    ".cufile",
    ".cufile_driver",
    ".defaults",
    ".statistics",
    ".stream",
    ".utils",
    "._lib",
    "._lib._runtime",
    "._lib.arr",
    "._lib.buffer",
    "._lib.batch",
    "._lib.cufile_driver",
    "._lib.defaults",
    "._lib.file_handle",
    "._lib.future",
    "._lib.statistics",
    "._lib.stream",
)


def enable_kvikio_compat():
    """Alias the supported modules at application startup, before importing KvikIO.

    Repeated calls are harmless. Existing KvikIO modules are never replaced.
    """
    with _lock:
        existing = {
            name: module
            for name, module in tuple(sys.modules.items())
            if name == "kvikio" or name.startswith("kvikio.")
        }
        for name, module in existing.items():
            if module is not sys.modules.get("cufile_python" + name[len("kvikio") :]):
                raise RuntimeError("KvikIO is already imported; enable compatibility before importing it")
        aliases = {"kvikio" + suffix: importlib.import_module("cufile_python" + suffix) for suffix in _modules}
        for name, module in aliases.items():
            if name in sys.modules and sys.modules[name] is not module:
                raise RuntimeError("KvikIO was imported while enabling compatibility")
        sys.modules.update(aliases)

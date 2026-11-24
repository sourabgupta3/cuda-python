# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Import order and flushed progress messages are part of this crash reproducer.
# ruff: noqa: E402, T201
"""Reproduce cuCIM Device parsing after loading libcufile; use a subprocess.

The default order segfaulted with cuCIM 26.10.0 and libcufile reporting 1190.
--device-first succeeded. Neither case needs cufile_python or file I/O.
"""

import argparse
import faulthandler
import importlib

faulthandler.enable()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device-first", action="store_true")
parser.add_argument("--kvikio", action="store_true")
args = parser.parse_args()

if args.device_first:
    from cucim.clara.io import Device

    print(Device("cpu"), flush=True)

if args.kvikio:
    importlib.import_module("kvikio")
else:
    from cuda.bindings import cufile

    print("cuFile version:", cufile.get_version(), flush=True)

from cucim.clara.io import Device

print("Constructing Device after library load", flush=True)
print(Device("cpu"), flush=True)

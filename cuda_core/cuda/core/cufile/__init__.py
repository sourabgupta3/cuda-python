# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native GPUDirect Storage with CUDA core ownership and context management.

Linux only. libcufile is loaded lazily. CUDA stream and memory objects remain
interoperable with cuda.core; integer streams and external contiguous arrays
are also accepted. Native library routing and configuration are respected.
"""

import sys as _sys

if _sys.platform != "linux":
    raise ImportError("cuda.core.cufile requires Linux")

from ._batch import Batch, BatchIOParams, IOCompletion, IOOpcode
from ._buffers import RegisteredBuffer, RegisteredStream
from ._driver import Driver
from ._file import File, IOParams
from ._runtime import CuFileError

__all__ = [
    "Batch",
    "BatchIOParams",
    "CuFileError",
    "Driver",
    "File",
    "IOCompletion",
    "IOOpcode",
    "IOParams",
    "RegisteredBuffer",
    "RegisteredStream",
]

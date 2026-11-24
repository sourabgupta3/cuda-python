# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Python backend: no libkvikio shared library is imported.

from cufile_python import defaults  # noqa: F401
from cufile_python._lib.defaults import CompatMode  # noqa: F401
from cufile_python._version import __git_commit__, __version__
from cufile_python.batch import Batch, BatchCompletion, BatchOp
from cufile_python.buffer import bounce_buffer_free, memory_deregister, memory_register
from cufile_python.cufile import (
    CuFile,
    clear_page_cache,
    drop_file_page_cache,
    drop_system_page_cache,
    get_page_cache_info,
)
from cufile_python.statistics import Summary, SummaryMonitor
from cufile_python.stream import stream_deregister, stream_register
from cufile_python.utils import kvikio_deprecation_notice

__all__ = [
    "Batch",
    "BatchOp",
    "BatchCompletion",
    "__git_commit__",
    "__version__",
    "clear_page_cache",
    "CuFile",
    "drop_file_page_cache",
    "drop_system_page_cache",
    "get_page_cache_info",
    "kvikio_deprecation_notice",
    "Summary",
    "SummaryMonitor",
    "stream_register",
    "stream_deregister",
    "memory_register",
    "memory_deregister",
    "bounce_buffer_free",
]

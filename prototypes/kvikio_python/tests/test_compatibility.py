# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

import cufile_python as kvikio
import cufile_python.defaults


@pytest.mark.parametrize("mode", list(kvikio.CompatMode))
@pytest.mark.agent_authored(model="gpt-6")
def test_native_stream_statistics_are_independent_of_legacy_mode(tmp_path, mode):
    import cupy as cp

    source = cp.arange(4096, dtype=cp.uint8)
    stream = cp.cuda.Stream()
    cp.cuda.get_current_stream().synchronize()
    with kvikio.defaults.set("compat_mode", mode), kvikio.SummaryMonitor() as monitor:
        with kvikio.CuFile(tmp_path / "data", "w") as file:
            completion = file.raw_write_async(source, stream.ptr)
            # Native stream operations retain KvikIO's statistics coverage.
            assert monitor.get().num_ops == 0
            assert completion.check_bytes_done() == 4096
        assert monitor.get().num_ops == 0

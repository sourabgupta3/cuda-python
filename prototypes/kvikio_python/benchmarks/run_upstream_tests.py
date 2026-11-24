# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run selected upstream tests with explicit aliases, including spawned children."""

import sys

from cufile_python.compat import enable_kvikio_compat

enable_kvikio_compat()

if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main(sys.argv[1:]))

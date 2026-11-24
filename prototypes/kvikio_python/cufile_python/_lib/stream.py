# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from ._runtime import native


def stream_register(stream, flags):
    native("stream_register", stream, flags)


def stream_deregister(stream):
    native("stream_deregister", stream)

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cufile_python._lib import buffer  # type: ignore


def memory_register(buf) -> None:
    """Register a device memory allocation with cuFile for GPUDirect Storage access.

    This function automatically discovers the base address and size of the CUDA memory
    allocation containing ``buf``. The entire underlying allocation is registered,
    regardless of which portion ``buf`` points to.

    Registration pins the memory for GPU Direct DMA transfers, which can improve
    performance when the same buffer is reused across multiple cuFile I/O operations.

    This always calls cuFile, regardless of the legacy ``compat_mode`` setting.
    Unsupported allocations and native registration failures raise errors.

    Warnings
    --------
    This API is intended for streaming buffers reused across multiple cuFile I/O
    operations. For one-time transfers, the overhead of registration may outweigh the
    benefits.

    Parameters
    ----------
    buf: buffer-like or array-like
        Device buffer to register .
    """
    return buffer.memory_register(buf)


def memory_deregister(buf) -> None:
    """Deregister a device memory allocation from cuFile.

    This function automatically discovers the base address of the CUDA memory
    allocation containing ``buf``. The entire underlying allocation is deregistered,
    regardless of which portion ``buf`` points to.

    This always calls cuFile, regardless of the legacy ``compat_mode`` setting.

    Parameters
    ----------
    buf: buffer-like or array-like
        Device buffer to deregister.
    """
    buffer.memory_deregister(buf)


def bounce_buffer_free() -> int:
    """Retained no-op for source compatibility; this backend owns no bounce buffers.

    This does not release buffers cached inside libcufile.

    Returns
    -------
    int
        Always zero.
    """
    return buffer.bounce_buffer_free()

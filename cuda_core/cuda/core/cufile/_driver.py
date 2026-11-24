# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit ownership of a native cuFile driver reference."""

from ._runtime import _ScopedResource, call, runtime


class Driver(_ScopedResource):
    """Open a native cuFile driver reference immediately and close it explicitly.

    This object supports ``with`` and idempotent ``close()``. It owns a balanced
    reference to process-wide cuFile state, not an independent driver instance.
    Native cuFile manages driver reference counts. File and buffer registration
    use native implicit initialization; the wrapper holds no hidden reference.
    Complete I/O and close its resources before releasing the final Driver.
    ``is_closed`` reports only whether this object's close succeeded.

    Keep one Driver open around the workload rather than creating one per
    transfer. Repeated construction and close add native open/close calls.

    Use standalone ``cuda.bindings.cufile`` functions for version, property,
    and configuration APIs. Configuration is process-wide; setters that require
    an unopened driver must run before constructing Driver or any I/O resource.
    """

    def __init__(self) -> None:
        self._closed = True
        runtime.check()
        call("driver_open")
        self._closed = False
        runtime.track(self)

    @property
    def is_closed(self) -> bool:
        """Whether this object's explicit native driver reference was released."""
        return self._closed

    def close(self) -> None:
        """Balance this object's native driver open; repeated closes do nothing."""
        if not self._closed:
            runtime.check(allow_shutdown=True)
            call("driver_close")
            self._closed = True

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import enum
import operator
import os
import threading


class CompatMode(enum.IntEnum):
    OFF = 0
    ON = 1
    AUTO = 2


_values = dict(
    compat_mode=CompatMode.AUTO,
    thread_pool_nthreads=1,
    task_size=4 * 1024**2,
    gds_threshold=16 * 1024,
    bounce_buffer_size=16 * 1024**2,
    auto_direct_io_read=False,
    auto_direct_io_write=True,
)
_lock = threading.RLock()
_initializing = True


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw.upper() not in ("0", "1", "OFF", "ON", "FALSE", "TRUE", "NO", "YES"):
        raise ValueError(f"Invalid boolean setting {name}")
    return raw.upper() in ("1", "ON", "TRUE", "YES")


auto_direct_io_read_overread = _env_bool("KVIKIO_AUTO_DIRECT_IO_READ_OVERREAD", False)
thread_pool_per_block_device = _env_bool("KVIKIO_THREAD_POOL_PER_BLOCK_DEVICE", False)


def _set(name, value):
    old = _values[name]
    if isinstance(old, enum.IntEnum):
        value = type(old)(value)
    elif isinstance(old, bool):
        value = bool(value)
    else:
        value = operator.index(value)
        if value < 0:
            raise OverflowError("can't convert negative value to unsigned integer")
        if value == 0 and name != "gds_threshold":
            raise ValueError(f"{name} must be a positive integer")
    with _lock:
        _values[name] = value
    if name == "thread_pool_nthreads" and value != old and not _initializing:
        from ._runtime import reset_pool

        reset_pool()


def _getter(name):
    def get():
        with _lock:
            return _values[name]

    get.__name__ = name
    return get


def _setter(name):
    def set_value(value):
        _set(name, value)

    set_value.__name__ = "set_" + name
    return set_value


for _name in _values:
    globals()[_name] = _getter(_name)
    globals()["set_" + _name] = _setter(_name)
    env = "KVIKIO_" + ("NTHREADS" if _name == "thread_pool_nthreads" else _name.upper())
    raw = os.environ.get(env)
    if _name == "thread_pool_nthreads":
        raw = raw if raw is not None else os.environ.get("KVIKIO_NUM_THREADS")
    if raw is not None:
        value = _values[_name]
        if isinstance(value, enum.IntEnum):
            if isinstance(value, CompatMode):
                raw = {"TRUE": "ON", "YES": "ON", "FALSE": "OFF", "NO": "OFF"}.get(raw.upper(), raw)
            raw = type(value)[raw.upper()] if not raw.isdigit() else int(raw)
        elif isinstance(value, bool):
            if raw.lower() not in ("true", "false", "on", "off", "yes", "no", "0", "1"):
                raise ValueError(f"unknown config value {env}={raw}")
            raw = raw.lower() in ("true", "on", "yes", "1")
        else:
            raw = int(raw)
        # No thread pool exists during initial environment parsing.
        _set(_name, raw)

_initializing = False


def is_compat_mode_preferred():
    """The legacy preference is accepted, but I/O always requires cuFile."""
    return False

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json
import struct
import threading
import time
import weakref

# Preserve KvikIO's serialized Summary layout. Excluded backends stay at zero
# in locally collected summaries; deserialization can still read their fields.
_BACKENDS = ("POSIX", "GDS", "MMAP", "REMOTE_HTTP", "REMOTE_HDFS")
_TOTALS = (
    "num_ops",
    "num_reads",
    "num_writes",
    "bytes_requested",
    "bytes_transferred",
    "bytes_read",
    "bytes_written",
    "num_errors",
)
_COUNTERS = (
    "remote_size_probes",
    "remote_size_probing_ns",
    "http_connections",
    "http_dns_ns",
    "http_tcp_ns",
    "http_tls_ns",
    "http_retries",
    "http_retry_backoff_ns",
)
_BACKEND_TOTALS = ("num_ops", "bytes_transferred", "total_duration_ns", "num_errors")
_lock = threading.RLock()
_monitors = weakref.WeakSet()
_counters = dict.fromkeys(_COUNTERS, 0)


def counter(name, value=1):
    with _lock:
        _counters[name] += value


class Summary:
    def __init__(self, data, start, end, anchor):
        self.data, self.start, self.end, self.anchor = data, start, end, anchor

    def as_dict(self):
        data = copy.deepcopy(self.data)
        wall = max(self.end - self.start, 0)
        data.update(
            start_unix_ns=self.start - self.anchor[0] + self.anchor[1],
            end_unix_ns=self.end - self.anchor[0] + self.anchor[1],
            wall_ns=wall,
            busy_bytes_per_sec=data["bytes_transferred"] * 1e9 / data["busy_ns"] if data["busy_ns"] else 0.0,
            busy_fraction=data["busy_ns"] / wall if wall else 0.0,
            mean_duration_ns=data["total_duration_ns"] // data["num_ops"] if data["num_ops"] else 0,
        )
        return data

    def since(self, previous):
        if self.anchor != previous.anchor or self.start != previous.start or previous.end > self.end:
            raise ValueError("Summaries must be readings of the same monitor and span")
        data = copy.deepcopy(self.data)
        for k in (*_TOTALS, "busy_ns", "total_duration_ns"):
            data[k] -= previous.data[k]
        for k in _COUNTERS:
            data["counters"][k] -= previous.data["counters"][k]
        for backend in _BACKENDS:
            for k in _BACKEND_TOTALS:
                data["by_backend"][backend][k] -= previous.data["by_backend"][backend][k]
        return Summary(data, previous.end, self.end, self.anchor)

    def to_json(self):
        return json.dumps(self.as_dict(), sort_keys=True)

    def serialize(self):
        values = [self.start, self.end, *self.anchor, *[self.data[k] for k in _TOTALS]]
        for backend in _BACKENDS:
            values.extend(self.data["by_backend"][backend][k] for k in _BACKEND_TOTALS)
        values.extend(self.data["counters"][k] for k in _COUNTERS)
        values += [self.data["total_duration_ns"], self.data["busy_ns"]]
        return b"KVSM" + struct.pack("<III", 2, 336, 0x01020304) + struct.pack("=42Q", *values)

    @staticmethod
    def deserialize(data):
        if len(data) != 352 or data[:16] != b"KVSM" + struct.pack("<III", 2, 336, 0x01020304):
            raise ValueError("Invalid serialized KvikIO summary")
        values = struct.unpack("=42Q", data[16:])
        record = dict(zip(_TOTALS, values[4:12]))
        record["by_backend"] = {
            b: dict(zip(_BACKEND_TOTALS, values[12 + i * 4 : 16 + i * 4])) for i, b in enumerate(_BACKENDS)
        }
        record["counters"] = dict(zip(_COUNTERS, values[32:40]))
        record["total_duration_ns"], record["busy_ns"] = values[40:42]
        return Summary(record, values[0], values[1], values[2:4])

    def report(self, all_rows=False):
        d = self.as_dict()
        rows = [
            "KvikIO I/O summary",
            f"  wall time            {d['wall_ns']} ns",
            f"  busy time            {d['busy_ns']} ns",
            f"  operations           {d['num_ops']}",
            f"  bytes                {d['bytes_transferred']} of {d['bytes_requested']} requested",
            f"  errors               {d['num_errors']}",
        ]
        for name, b in d["by_backend"].items():
            if b["num_ops"] or all_rows:
                rows.append(f"  backend {name:12} {b['bytes_transferred']} bytes in {b['num_ops']} ops")
        if all_rows or any(d["counters"].values()):
            for name, value in d["counters"].items():
                if value or all_rows:
                    rows.append(f"  {name.replace('_', ' '):24} {value}")
        return "\n".join(rows) + "\n"


class SummaryMonitor:
    def __init__(self):
        self.stopped = False
        self.reset()
        with _lock:
            _monitors.add(self)

    def reset(self):
        with _lock:
            self.start = time.monotonic_ns()
            self.anchor = (self.start, time.time_ns())
            self.end = self.start
            self.data = dict.fromkeys((*_TOTALS, "busy_ns", "total_duration_ns"), 0)
            self.data["by_backend"] = {b: dict.fromkeys(_BACKEND_TOTALS, 0) for b in _BACKENDS}
            self.data["counters"] = dict.fromkeys(_COUNTERS, 0)
            self.base_counters = dict(_counters)
            self.active = 0
            self.busy_start = self.start

    def get(self):
        with _lock:
            end = self.end if self.stopped else time.monotonic_ns()
            data = copy.deepcopy(self.data)
            if self.active:
                data["busy_ns"] += end - self.busy_start
            data["busy_ns"] = min(data["busy_ns"], end - self.start)
            if not self.stopped:
                data["counters"] = {k: _counters[k] - self.base_counters[k] for k in _COUNTERS}
            return Summary(data, self.start, end, self.anchor)

    def since(self, previous):
        return self.get().since(previous)

    def stop(self):
        with _lock:
            if not self.stopped:
                self.end = time.monotonic_ns()
                self.data["counters"] = {k: _counters[k] - self.base_counters[k] for k in _COUNTERS}
                self.stopped = True
                _monitors.discard(self)


def observe(size, write, backend):
    if not _monitors:
        return None
    backend = {"posix_device": "POSIX", "posix_host": "POSIX", "gds": "GDS"}.get(backend, backend)
    start = time.monotonic_ns()
    with _lock:
        monitors = tuple(_monitors)
        for m in monitors:
            if m.active == 0:
                m.busy_start = start
            m.active += 1
    finished = False

    def finish(count, error):
        nonlocal finished
        end = time.monotonic_ns()
        with _lock:
            if finished:
                return
            finished = True
            for m in monitors:
                if m.stopped or start < m.start:
                    continue
                if m.active:
                    m.active -= 1
                    if m.active == 0:
                        m.data["busy_ns"] += end - m.busy_start
                d = m.data
                duration = end - start
                d["num_ops"] += 1
                d["num_writes" if write else "num_reads"] += 1
                d["bytes_requested"] += size
                d["bytes_transferred"] += count
                d["bytes_written" if write else "bytes_read"] += count
                d["num_errors"] += int(error is not None)
                d["total_duration_ns"] += duration
                b = d["by_backend"][backend]
                for k, v in zip(_BACKEND_TOTALS, (1, count, duration, int(error is not None))):
                    b[k] += v

    return finish

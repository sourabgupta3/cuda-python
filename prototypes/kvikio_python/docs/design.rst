cufile_python: Python API Design
================================

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: entry

1. Purpose and Release Goals
----------------------------

Provide cuFile's Python APIs through ``cufile_python``. One goal is to release
the package with the CUDA Toolkit, so its APIs and supported cuFile versions
stay in step. This is a release goal; the current implementation is a local
prototype.

Keep familiar KvikIO Python names and behavior where they fit, so applications
can move with few changes. Implement the backend in Python using CUDA bindings,
and add native cuFile batch I/O.

2. Scope and Compatibility
--------------------------

The shared APIs follow the `KvikIO 26.10 Python API reference
<https://docs.nvidia.com/kvikio/26.10/python/>`_: local file reads and writes,
futures, CUDA-stream I/O, driver controls, and buffer/stream registration.
That reference covers the shared names and arguments; this document focuses
on our implementation and differences.

Native batch I/O is an addition to that KvikIO API. Remote storage, Zarr,
and CUDA graph capture are outside this design. This is not a replacement
for every KvikIO feature.

3. Architecture and Call Path
-----------------------------

The call path is::

    Application
        -> cufile_python (Python APIs, buffers, futures, workers)
        -> cuda.bindings.cufile / cuda.bindings.driver
        -> native cuFile / CUDA

The shared Python frontends retain KvikIO's Apache-2.0 attribution. Their
Cython/C++ backend is replaced with Python code calling ``cuda.bindings``.
There is no dependency on KvikIO's C++ library and no new native extension.

Python checks arguments, keeps buffers and files alive during I/O, and manages
futures. Native cuFile performs the transfers and selects GDS or its own
compatibility path based on system support and configuration. ``libcufile``
is required, including for host-buffer I/O.

4. API Design
-------------

4.1 File I/O and Completion
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``read/write`` block until completion. ``pread/pwrite`` use Python workers
and return a future; ``get()`` waits and returns the byte count. Raw calls
submit directly, and stream calls use the supplied CUDA stream. The stream
future's ``check_bytes_done()`` waits for that stream and returns the result.

Contiguous host buffers and GPU buffers exposing the CUDA Array Interface are
supported for file I/O. Raw and stream methods keep KvikIO's GPU-only interface.
Native cuFile still determines which allocations and operations it supports.
For raw, stream, and batch calls, the caller must have the appropriate CUDA
context current; Python workers manage their own current context.

KvikIO compatibility-mode settings are accepted for porting, but do not change
I/O routing. There is no Python fallback when cuFile is missing. Changing a
Python import also cannot replace I/O compiled into a framework's C++ library.

4.2 Resource Lifetime and Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Files and batches support ``with`` or explicit ``close()``. Closing finishes
their pending I/O before releasing the file handle or batch queue. A successful
second close does nothing. Collect results before closing if they are needed.
Keep externally owned memory, registrations, and raw streams valid until I/O
finishes; removing a registration does not free the allocation or stream.

File operations use cuFile's implicit driver initialization. The wrapper adds
no automatic driver-open reference. Explicit ``driver_open/driver_close`` call
the native APIs; close the driver only after all users have finished. Driver
settings are process-wide, including changes made by other libraries.

Argument errors use Python exceptions such as ``ValueError``. Native cuFile
failures are reported as ``RuntimeError`` with the bindings exception as the
cause. Future or batch result access reports transfer failures; a batch result
also exposes the native status and result value.

4.3 Native Batch I/O
~~~~~~~~~~~~~~~~~~~~

``Batch(capacity)`` creates a native batch queue. Each ``BatchOp`` describes a
read or write with its own file, buffer, size, offsets, and optional ``cookie``
to identify the result. One queue can use several files and host/GPU buffers.

``submit(entries)`` packs the native parameters and starts the transfers.
There is no separate prepare step or Python worker pool. ``poll()`` returns
ready completions without waiting; ``wait_all()`` waits for the remaining
ones. Neither returns a completion already collected by the other.
``pending`` is the number still to collect. Each completion's ``get()`` returns
bytes transferred or raises an error.

Collect all results before reusing the queue. Entries can be reused or changed
with ``dataclasses.replace`` for the next submission. Refilling individual
slots while other entries are pending is not supported. Entries may run in
any order: submit independent transfers and finish GPU work that produces
their data first. Batch I/O is not ordered on a CUDA stream.

On libcufile 1.19, host-buffer batches must use ``wait_all()`` without a timeout;
polling and timed waits are rejected for those batches. Active cancellation is
also disabled on that version. Other cuFile versions need validation before
these restrictions can be removed.

5. Usage Examples
-----------------

5.1 File I/O with a Future
~~~~~~~~~~~~~~~~~~~~~~~~~~

Shared file APIs: only the import changes. This writes a file, then reads it
through a future.

.. code-block:: python

    import cufile_python as kvikio

    source = bytearray(b"x" * 4096)
    destination = bytearray(len(source))
    with kvikio.CuFile("data.bin", "w+") as file:
        assert file.write(source) == len(source)
        future = file.pread(destination)
        assert future.get() == len(destination)
    assert destination == source

5.2 Batch I/O Across Files
~~~~~~~~~~~~~~~~~~~~~~~~~~

Batch extension: read the first example's file and write a different file in
one submission. The transfers use separate buffers and can run in any order.
A second submission changes just one entry.

.. code-block:: python

    from dataclasses import replace
    import cufile_python as cufile

    left, right = bytearray(4096), bytearray(b"x" * 4096)
    with (
        cufile.CuFile("data.bin") as first,
        cufile.CuFile("copy.bin", "w+") as second,
        cufile.Batch(2) as batch,
    ):
        entries = [
            cufile.BatchOp(first, left, "read", cookie="left"),
            cufile.BatchOp(second, right, "write", cookie="right"),
        ]
        batch.submit(entries)
        results = batch.wait_all()
        assert {item.cookie: item.get() for item in results} == {
            "left": 4096, "right": 4096
        }
        entries[1] = replace(entries[1], size=2048, file_offset=4096)
        batch.submit(entries)
        assert [item.get() for item in batch.wait_all()] == [4096, 2048]
    assert left == right == bytearray(b"x" * 4096)

Without ``with``, use ``try/finally`` and call ``batch.close()`` before
``file.close()``. For GPU-only batches that support polling, process results
from ``poll()`` as they arrive, then call ``wait_all()`` for any remaining ones.
Calling ``poll()`` first is optional.

6. Performance and Validation
-----------------------------

Python buffer checks and worker/future handling add overhead compared with
KvikIO's C++ backend, especially for small transfers. Existing measurements
show a 4 KiB worker read at 127.9 us versus 97.1 us for KvikIO; a 64 MiB read
was 7.086 ms versus 7.012 ms, within the observed run variation. These are
measured examples, not performance guarantees. Batch avoids the Python worker
pool but still pays for packing parameters and collecting results.

Tests cover the selected upstream KvikIO APIs and the added batch behavior.
Application checks cover Python calls in cuDF/cuCIM workloads; they do not
establish compatibility with those frameworks' compiled I/O paths. Detailed
measurements and test results are in the separate performance and application
reports alongside this local design.

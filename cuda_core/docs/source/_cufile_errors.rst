.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Catch ``cuda.core.cufile.CuFileError`` for both cuFile and filesystem failures.
It derives from cuda.core.CUDAError and carries optional native and filesystem
metadata. It is not an OSError subclass; inspect errno for a known filesystem
error and native_code for a cuFile status. Missing metadata is None.

The low-level module is cuda.bindings.cufile; its exception class is
``cuda.bindings.cufile.cuFileError``. When the Pythonic API converts that exception,
it preserves the native status and the original exception as ``__cause__``. Its
sync-transfer adapter captures native returns and errno directly. Users of the
Pythonic API catch CuFileError.

.. list-table:: Errors callers can catch
   :header-rows: 1
   :widths: 30 70

   * - Exception
     - Meaning
   * - TypeError / ValueError
     - Python argument type, bounds, mode, flags, or buffer layout is invalid.
   * - RuntimeError
     - Closed object, pending result, overlapping reuse, active/uncertain batch,
       or use after fork. Calling bytes_done before async completion is one example.
   * - CuFileError
     - cuFile or filesystem failure, duplicate/overlapping registration, or failed
       batch completion. Also a CUDAError. Inspect native_code/status_name or
       errno/filename/strerror, plus cuda_error, operation, and native_result.
   * - CUDAError
     - A CUDA context, stream, event, or memory API failed directly.
       A cuFile API returning CUDA_DRIVER_ERROR instead raises CuFileError with cuda_error.
   * - TimeoutError
     - A batch wait_all/close deadline expired while I/O remains active.
       The batch stays open. A native per-entry cuda.bindings.cufile.Status.TIMEOUT
       is a separate result.
   * - NotImplementedError
     - A feature unavailable through this API, such as device-buffer registration
       without CUDA GPUDirect capability (including in compatibility mode), or
       cancellation without reliable completion reporting.

CuFileError.status is an alias of native_code and uses the same numeric cuFile
status as cuda.bindings.cufile.cuFileError.status. status_name uses the bindings'
OpError name, for example MEMORY_ALREADY_REGISTERED, IO_NOT_SUPPORTED, or
ASYNC_NOT_SUPPORTED. Unknown raw transfer codes are preserved; status_name is then None.
cuda_error contains the separate CUDA driver status only when supplied by cuFile.
Compare status with int(bindings.OpError.INVALID_VALUE), or inspect status_name;
no duplicate status enumeration is introduced.

Duplicate/overlapping tracked buffer registrations report MEMORY_ALREADY_REGISTERED;
duplicate tracked stream registrations report INVALID_VALUE. Both raise CuFileError
without changing the original registration. Different requested sizes or flags do not
create a separate registration. Native errors for registrations made elsewhere are
also reported as CuFileError; this module does not take over their ownership.

operation identifies the failed wrapper/native operation, for example open, read,
write_async, buf_register, or batch[0]. native_result preserves a signed native
transfer result when one exists. Missing metadata is None. Native binding errors
and Python filesystem exceptions remain available as __cause__ when wrapped.
Direct calls to cuda.bindings.cufile continue to raise its own cuFileError.

CuFileError carries optional filesystem fields errno, strerror, and filename.
For example, File opening a missing path reports errno.ENOENT and the path.
Synchronous native I/O captures errno on the calling thread immediately after
a -1 return. A deferred stream/batch result of -1 does not supply a reliable errno:
the error is CuFileError with errno=None, native_code=None, and native_result=-1.
The wrapper does not read an unrelated thread's errno or label -1 as cuFile status 1.

Sync failures raise at read/write. Stream submission failures raise at
read_async/write_async; transfer failures raise when bytes_done is read after
completion. Batch.poll returns IOCompletion objects with per-entry errors;
completion.result() raises its stored error. wait_all() checks only results it collects,
so an error already returned by poll() is not raised again by wait_all() or close().
If wait_all() raises a transfer error or TimeoutError, error.completions contains the
results collected by that call, including partial results before the deadline.
A poll failure after collecting native results also carries error.completions.
The batch does not retain or replay those delivered results. EOF and short transfers
are successful counts, not errors.

If cleanup fails while handling another exception, the original exception is
preserved and the cleanup failure is attached as a note. Errors from finalizers
are reported as CUDAWarning. Missing libraries/symbols retain their loader
exception types; they are not converted into successful fallback I/O.

For example, path and destination are supplied by the application:

.. code-block:: python

   import os
   from cuda.core import cufile
   from cuda.bindings import cufile as bindings

   try:
       with cufile.File(path, mode="r", flags=os.O_DIRECT) as file:
           params = cufile.IOParams(
               destination, size=None, file_offset=0, buffer_offset=0
           )
           bytes_read = file.read(params)
   except cufile.CuFileError as error:
       if error.errno is not None:
           print("Filesystem failure:", error.errno, error.filename, error.strerror)
       elif error.status == int(bindings.OpError.IO_NOT_SUPPORTED):
           print("cuFile does not support this operation")
       print("cuFile failure:", error.operation, error.status_name,
             error.native_code, error.cuda_error, error.native_result)
       raise

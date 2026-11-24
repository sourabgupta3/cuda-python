.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. list-table:: Constructor defaults
   :header-rows: 1
   :widths: 25 75

   * - Class
     - Required inputs and defaults
   * - Driver
     - No arguments. Opens immediately; is_closed starts False.
   * - File
     - path required. mode="r", flags=os.O_DIRECT. Read-only, existing file;
       flags=0 omits O_DIRECT.
   * - IOParams
     - buf required. size=None means bytes remaining after buffer_offset;
       file_offset=0 and buffer_offset=0. No stream or result exists initially.
   * - RegisteredBuffer
     - buf required. size=None registers the supplied buffer's byte length;
       explicit size must be positive and fit the buffer. flags=0; only 0 is accepted.
       Registers existing memory; allocates no buffer. Duplicate/overlapping
       registrations raise CuFileError; no registration sharing.
   * - RegisteredStream
     - stream required. flags=0; size and offsets are read at execution time.
       Creates no CUDA stream and makes no alignment promise. Duplicate registration
       raises CuFileError even when flags match.
   * - Batch
     - capacity required; no default queue size. Starts idle with no outstanding requests.
       Create a File or Driver first to initialize native cuFile.
   * - BatchIOParams
     - file, buf, opcode, and size required. file_offset=0, buffer_offset=0,
       cookie=None. size must be positive; neither opcode nor size is inferred.
   * - IOCompletion
     - Returned by Batch. index, cookie, status, and bytes_transferred are required
       if constructed manually. error=None and native_result=None.
   * - IOOpcode
     - Enum value required. No default operation.

   * - CuFileError
     - message required. native_code=None, cuda_error=None, errno=None,
       filename=None, strerror=None, operation=None, native_result=None.
       One CUDAError subclass for cuFile and filesystem failures.

IOCompletion.status uses cuda.bindings.cufile.Status directly; there is no separate
completion-status enum in cuda.core.cufile.

File flags use os.open modifiers; registration flags use cuFile's separate
definitions. Supplying File flags replaces the default os.O_DIRECT value;
mode controls access, creation, and truncation.
Creation uses permissions 0o666 subject to the process umask. The supported
modes are r (read existing), r+ (read/write existing), w (write, create/truncate),
and w+ (read/write, create/truncate). O_DIRECT does not guarantee durability;
opening without it does not force a particular cuFile transfer path.

Methods read(params) and write(params) require IOParams and return completed
counts directly. Methods read_async(params, stream) and write_async(params, stream)
require both arguments; no default stream is selected. Set size and offsets before
submission in ordinary use. Use a separate IOParams for overlapping transfers.

.. list-table:: Batch method defaults
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Default behavior
   * - submit(operations)
     - Pack and enqueue new entries that fit the remaining capacity; entries are required.
   * - poll(min_complete=0, max_complete=None, timeout=0.0)
     - Do not wait for a minimum count. Return newly available completions;
       max_complete=None uses queue capacity, limited to entries still outstanding.
   * - wait_all(timeout=None)
     - Wait without a deadline; return only remaining results in submission order and raise transfer errors.
   * - close(*, timeout=None)
     - Wait without a deadline before releasing the queue. A deadline expiry leaves it open.

Timeouts are seconds. None means no deadline; 0 requests no waiting.
cancel() has no arguments. poll() and wait_all() never repeat delivered completions.
libcufile 1.19 rejects submit() while requests remain outstanding; drain first.
The other resource classes have close() with no arguments and no timeout.

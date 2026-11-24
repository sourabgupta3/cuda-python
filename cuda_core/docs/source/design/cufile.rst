.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _cufile_design:

Native cuFile API design
========================

Purpose
-------

cuda.core.cufile provides Pythonic file I/O over cuda.bindings.cufile. Native libcufile
selects a GDS path when its GPU, memory, driver, filesystem, and backend requirements
are met. It can use POSIX compatibility mode with host-memory staging when native
configuration allows or forces it. Routing follows cuFile's support matrix and
configuration; using these Python APIs does not guarantee a GDS transfer.

The API defines how to submit transfers, check results, and manage file, registration,
stream, and driver lifetimes when several Python libraries use cuFile in one process.

Scope
-----

The API covers Linux local files, synchronous calls, CUDA stream I/O, batches,
file/buffer/stream registration, and driver open/close. Version, properties, and
settings use cuda.bindings.cufile. All three I/O methods accept contiguous host or
device buffers, including cuda.core Buffer, CUDA Array Interface arrays, and Python
buffers such as NumPy arrays, bytearray, and mmap. Host memory can be pinned or
pageable. Stream I/O requires a CUDA stream/context. Batches may mix host and device
entries, but device buffers in one batch must use one CUDA context.

CUDA graphs are not supported. This includes capturing cuFile stream submissions
with ordinary host or device buffers.

The initial API excludes host thread pools, automatic transfer splitting, Python
environment-variable settings, remote protocols, format adapters, statistics/cache
administration, and a cache that keeps unused registrations open. Native cuFile settings
control compatibility behavior and the storage path.

Using cuFile from multiple libraries
------------------------------------

All Driver objects in a process use the same cuFile settings. Calling Driver() does not
create a separate configuration. Set initialization-only options through
cuda.bindings.cufile before any library opens the driver. Closing a Driver does not
restore previous settings.

Only Driver() explicitly calls native driver-open. File registers its descriptor and
relies on cuFile's implicit initialization. The wrapper maintains no driver reference
count and holds no hidden driver reference. Native cuFile manages its own open/close
counts. Each library using an explicit Driver must keep it open until its I/O and
resource cleanup finish.

Each registration object owns one native registration. Creating another object for
the same stream or a duplicate/overlapping buffer range raises CuFileError; registrations
are not shared or resized. close() drains pending I/O before deregistration.

This module does not track registrations created through raw bindings or other
libraries. If two libraries use the same buffer or stream, they must agree who registers
and deregisters it.

Importing cuda.core.cufile does not open the driver or change cuFile settings.

Architecture
------------

Application code calls cuda.core.cufile. These classes check arguments, keep resources
alive while I/O uses them, and report native failures as Python exceptions. They use
cuda.bindings.cufile to call libcufile, which performs I/O and selects the storage path.

The Python layer keeps resources alive during I/O, checks buffer bounds, rejects
overlapping registrations, preserves completion results, and provides batch wait
timeouts. Supported operations and storage paths depend on native cuFile and its
configuration. This version also rejects device-buffer registration without
GPUDirect capability, including in compatibility mode.

RegisteredBuffer registers existing memory; it does not allocate memory.
RegisteredStream registers an existing CUDA stream; it does not create one. Closing a
registration does not free the buffer or destroy the CUDA stream. Driver() opens one
native driver reference, and its first successful close() releases that reference.

The repository's `Resource Handles Design
<https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/cuda/core/_cpp/rt/DESIGN.md>`_
and `VirtualMemoryResource design
<https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/cuda/core/_cpp/rt/VMM_DESIGN.md>`_
show how core keeps native handles alive, releases them after their last user, and
cleans up when construction fails.

Design
------

Open a File, provide an existing buffer, and choose one of the three I/O methods below.
After the transfer finishes, check the actual byte count before using the result. Reuse
files, buffers, and registrations across transfers.

The application allocates buffers and creates streams through cuda.core or another
library. The I/O classes use those supplied resources. They do not choose an allocator
or add a host thread pool.

IOParams(buffer, size, file_offset=..., buffer_offset=...) supplies the buffer, byte
count, and offsets for both synchronous and stream I/O. file.read(params) copies file
bytes into the supplied host or device buffer; file.write(params) copies buffer bytes
into the file. Each call uses explicit offsets rather than advancing a shared file
position.

Both synchronous calls wait on the host thread and return actual bytes, which may be
fewer than requested. Failures raise CuFileError directly. Sync calls do not set
params.bytes_done, create CUDA events, or synchronize a GPU stream. Writes do not call
os.fsync().

Construct IOParams(buffer, size), then pass it to file.read_async(params, stream) or
file.write_async(params, stream). Both methods queue the transfer and return None.
cuFile may read the size and offsets, and write the result, after the Python call
returns; IOParams keeps memory for these values alive. Work queued on the same CUDA
stream runs in order, so a read, computation, and write can share a stream.

Stream I/O requires cuFile's CUDA stream-memory-operation support and a usable native
execution backend. The POSIX stream path requires an enabled cuFile worker pool:
execution.parallel_io=True and execution.max_io_threads > 0. Direct userspace-RDMA
stream I/O may require compatibility mode. Sync I/O success does not establish stream
support. Native submission/completion failures propagate; the wrapper does not replace
stream I/O with synchronous I/O or a Python thread pool. CUDA graph capture is excluded
for all stream submissions, including ordinary buffers; capturing streams must be
rejected before calling cuFile.

Wait for the supplied stream, or for an event recorded after the transfer, then read
params.bytes_done. It returns the actual byte count or raises the transfer error.
Reading bytes_done does not wait: if the transfer is still running, it raises
RuntimeError. Use a separate IOParams for each overlapping transfer; reuse an open
IOParams after its previous transfer completes. Host buffers use the same stream
interface. Batch results use the batch polling/waiting methods.

Create Batch(capacity) for a list of transfers. Each BatchIOParams contains a File,
buffer, read/write operation, byte count, file offset, buffer offset, and optional
cookie for identifying the result. Entries can use different files and mix host and
device buffers. Device buffers in one batch must belong to the same CUDA context.
Create a File or Driver first so native cuFile is initialized before batch setup.

Call batch.submit(entries) to validate, pack, retain, and enqueue the supplied requests.
After collecting a completion, replace that entry with dataclasses.replace() and submit
it again. Other requests may stay outstanding, subject to queue capacity. libcufile
1.19 has a native partial-resubmission defect: on that version the wrapper rejects
submission while requests remain outstanding. Drain them before submitting again.

poll() returns newly completed entries once. wait_all() collects only the requests
still outstanding and returns those results in submission order; it never repeats
results already delivered by poll(). An empty tuple means no completions were collected;
is_active tells whether requests remain outstanding. The application keeps any history
it needs. Entries may finish out of order; consume a completed buffer before reusing it.

RegisteredBuffer(buffer) and RegisteredStream(stream) add cuFile registrations to
existing memory and streams. They do not allocate memory or create a CUDA stream. Keep
registrations open across transfers; close them after the I/O using them finishes. This
avoids repeating registration setup in the transfer loop.

Driver.close() calls native driver-close once for that Driver's explicit open. The
native library decides when its remaining references permit shutdown. Complete I/O
and close files, registrations, and batches before closing the final Driver. Pending
I/O retain native files, allocations, and streams, but
do not keep the cuFile driver open independently.

API surface
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Python API
     - Native cuFile API
     - Result / purpose
   * - File.read / write
     - cuFileRead / Write
     - Synchronous; actual byte count
   * - File.read_async / write_async
     - cuFileReadAsync / WriteAsync
     - IOParams.bytes_done after stream/event completion
   * - Batch / BatchIOParams / IOCompletion
     - BatchIOSetUp / Submit / GetStatus / Destroy / Cancel
     - Multi-file entries; polling and checked waits
   * - RegisteredBuffer
     - BufRegister / BufDeregister
     - Registration of existing memory
   * - RegisteredStream
     - StreamRegister / StreamDeregister
     - Registration of an existing stream
   * - Driver
     - DriverOpen / Close
     - Explicit native driver reference

Closing resources
~~~~~~~~~~~~~~~~~

File, Driver, IOParams, Batch, RegisteredBuffer, and RegisteredStream support with.
Entry returns the existing open object without reopening it. Exit calls close(), even
after a body exception. Once close() succeeds, another close() does nothing.

Using with
^^^^^^^^^^

A with block closes its object, even if you created it earlier. Entering a closed object
again raises RuntimeError. Use a new object for another block, and keep Driver creation
outside transfer loops.

.. code-block:: python

   from cuda.core import cufile

   driver = cufile.Driver()  # Opens immediately.
   with driver:
       assert not driver.is_closed
   assert driver.is_closed

   # Use a new object for another block; keep this outside I/O loops.
   with cufile.Driver() as another:
       assert not another.is_closed
   assert another.is_closed

Avoid using the same Driver in nested with blocks. The inner block closes it before the
outer block ends:

.. code-block:: python

   # Avoid nesting the same object.
   driver = cufile.Driver()
   with driver:
       with driver:
           pass
       assert driver.is_closed  # The inner block already closed it.

Different objects can be nested. Usually one Driver block around the workload is enough.
Rejecting nested use of the same object remains an open review item.

Resource ownership
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Resource
     - Owned state
     - Close / scope exit
   * - Driver
     - One explicit native open per object; reference counting is native
     - Call native driver-close once; close dependent resources first
   * - File
     - Descriptor and registered handle
     - Wait for pending I/O, then deregister and close
   * - RegisteredBuffer
     - cuFile registration and a reference to the existing allocation/context
     - Drain pending I/O, then deregister
   * - RegisteredStream
     - cuFile registration and a reference to the existing stream/context
     - Drain pending stream I/O, then deregister
   * - Batch
     - Native queue and outstanding requests with their files/buffers
     - Collect outstanding completions, release requests, destroy queue
   * - IOParams
     - Size, offsets, result memory, and resources used by pending I/O
     - close waits for pending I/O; bytes_done checks the result without waiting
   * - BatchIOParams / IOCompletion
     - Immutable operation description / completion result
     - No close method; returned results remain usable after batch reuse

Pending I/O retains native file handles, allocations, streams, CUDA contexts, and
argument memory until completion is collected. File.close() or registration.close()
can drain a batch that uses the resource; collect results explicitly first if the
application needs them. A batch close timeout leaves it open. Cleanup failures while
handling another exception are attached as notes; finalizers issue CUDAWarning.

Class reference
---------------

.. include:: ../_cufile_defaults.rst

The following sections list constructors, methods, properties, and fields. The close(),
is_closed, and with behavior applies to the six resource classes listed above.

Methods shared by resource classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   close() -> None
   is_closed: bool  # read-only property
   __enter__() -> Self
   __exit__(exc_type, exc, traceback) -> bool

File, Driver, IOParams, Batch, RegisteredBuffer, and RegisteredStream support with.
Entry returns the existing open object without reopening it. Exit calls close(), even
after a body exception. Once close() succeeds, another close() does nothing.
Batch.close() also accepts a timeout. IOParams.close() waits for pending I/O. The data
classes, enums, and exception class have no native resource to close.

Driver
~~~~~~

.. code-block:: text

   Driver()
   is_closed: bool  # read-only property
   close() -> None
   __enter__() -> Self
   __exit__(exc_type, exc, traceback) -> bool

Driver() immediately calls cuda.bindings.cufile.driver_open(). Its first successful
close() calls driver_close(). Entering with driver does not open it again.
driver.is_closed reports whether that object's reference was released; it does not
report whether cuFile is closed for the process. Native cuFile owns the reference
count. The current bindings use native cuFileDriverOpen and cuFileDriverClose_v2.

Keep one Driver open around the workload. Creating and closing one for every transfer
adds native calls, even when cuFile is already initialized. Put the Driver's with block
outside the transfer loop. Leaving that block closes the Driver object even if it was
constructed earlier.

Importing the module does not open the driver. File relies on native implicit
initialization during file-handle registration; it never calls driver-open or
driver-close. Buffer registration also follows native implicit initialization.
Create a File or Driver before Batch setup. The wrapper owns no hidden driver
reference and does not duplicate native driver reference counting. Without an
explicit Driver, cuFile manages implicit initialization and process-exit cleanup;
File.close() releases the file, not the driver. To control driver cleanup explicitly,
create Driver before the workload and close all dependent resources before it exits.

Standalone driver properties and configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use bindings.get_version(), driver_get_properties(), and ``get_parameter_*`` /
``set_parameter_*`` directly. driver_get_properties() takes the address of a native
CUfileDrvProps_t structure. Use individual parameter getters when a Python bool, int, or
string is sufficient.

Set initialization-only options before Driver() or any library opens an I/O resource.
Typed parameter setters reject changes after driver-open. The legacy ``driver_set_*``
initialization setters can return success without changing settings after driver-open.
All libraries in the process share these settings. A property value does not identify
the DMA path used by a transfer. Closing a Driver does not restore settings. For
example, set options at application startup::

   from cuda.bindings import cufile as bindings
   from cuda.core import cufile

   # Before any library opens the cuFile driver.
   parameter = bindings.BoolConfigParameter.PROPERTIES_USE_POLL_MODE
   bindings.set_parameter_bool(parameter, True)
   with cufile.Driver():
       poll_mode = bindings.get_parameter_bool(parameter)
       version = bindings.get_version()
       assert poll_mode

Driver properties report settings and capabilities. They do not identify the DMA path
used by an individual transfer.

File
~~~~

.. code-block:: text

   File(path: os.PathLike[str] | str, mode: str = "r", *, flags: int = os.O_DIRECT)
   read(params: IOParams) -> int
   write(params: IOParams) -> int
   read_async(params: IOParams, stream: object) -> None
   write_async(params: IOParams, stream: object) -> None
   fileno() -> int

File opens one descriptor and registers it with cuFile. mode selects access, creation,
and truncation: r and r+ open an existing file; w and w+ create or truncate it. The +
modes allow both reading and writing. Keyword-only flags supplies os.open modifiers
and defaults to os.O_DIRECT. Pass flags=0 to open without O_DIRECT. Supplying flags
replaces the default.

For example, with a contiguous source buffer whose earlier GPU work has finished::

   import os
   from cuda.core import cufile

   # Direct I/O is the default.
   with cufile.File("direct.bin", "w+", flags=os.O_DIRECT) as file:
       assert file.write(cufile.IOParams(source)) == source.nbytes

   # Open without O_DIRECT; cuFile still selects the transfer path.
   file = cufile.File("without-direct.bin", "w+", flags=0)
   try:
       assert file.write(cufile.IOParams(source)) == source.nbytes
   finally:
       file.close()

Since GDS 1.7 / CUDA Toolkit 12.2, cuFile accepts descriptors opened without O_DIRECT
and may still use a direct GPU/storage path. flags=0 does not force page-cache I/O.
See the `native handle registration reference
<https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilehandleregister>`_.

read/write wait on the host thread and return actual bytes; async
methods queue work on the supplied CUDA stream. fileno() returns File's descriptor; do
not close it separately. File.close() waits for its pending I/O, including batch requests using that file.

Sizes and offsets are bytes. size=None uses the bytes remaining after buffer_offset.
Explicit sizes must fit the buffer; use the intended byte count when an allocation is
larger than the data. Invalid bounds, noncontiguous buffers, and read-only destinations
raise errors. Zero-length I/O returns immediately; short transfers return actual bytes.
Writes do not call os.fsync(). O_DIRECT alone does not guarantee write durability;
use os.fsync(file.fileno()) when durability is required.

IOParams
~~~~~~~~

.. code-block:: text

   IOParams(buf: object, size: int | None = None, *,
            file_offset: int = 0, buffer_offset: int = 0)
   buf: object                # read-only property
   size: int | None           # mutable; resolved to bytes at submission
   file_offset: int           # mutable byte offset
   buffer_offset: int         # mutable byte offset
   bytes_done: int            # read-only result of stream I/O; never waits

IOParams supplies the same buffer, size, and offsets to read/write and
read_async/write_async. Sync methods return the actual byte count directly and raise
errors at the call. They do not set bytes_done or allocate a CUDA completion event.
For stream I/O, IOParams keeps native arguments, results, and required resources alive.
After stream/event completion, bytes_done returns actual bytes or raises the transfer
error. Earlier access raises RuntimeError. A completed stream result remains readable
after close(); an open object can be reused after completion. A sync submission clears
the previous stream result, so bytes_done raises RuntimeError until another stream
transfer is submitted and completes. Async methods return None. IOParams has no wait()
method or separate future.

IOParams stores size, offsets, and results in memory that stays alive during I/O. Set
arguments before submission unless explicitly ordering later updates. Changing a field
fixed by stream flags raises ValueError. Any later update must finish before cuFile
reads that field. Use separate objects for overlapping transfers. While I/O is pending,
size cannot exceed the maximum supplied at submission, and offsets must stay inside the
supplied buffer. For unregistered memory, the prototype adds buffer_offset to the
submitted address, so changing buffer_offset while pending requires registered memory.

A private CUDA event tells the implementation whether stream I/O has finished. Reading
bytes_done can release completed resources without waiting for later stream work. If
recording the event fails after submission, cleanup waits for the stream before freeing
argument memory.

RegisteredBuffer
~~~~~~~~~~~~~~~~

.. code-block:: text

   RegisteredBuffer(buf: object, size: int | None = None, *, flags: int = 0)
   size: int                 # read-only registered byte count
   flags: int                # read-only; reserved, always 0

RegisteredBuffer calls cuFileBufRegister once and owns that registration. size=None
registers the supplied buffer's byte length starting at its pointer; it does not expand
a view to the underlying CUDA allocation. An explicit size must be positive and no
larger than the supplied buffer. The read-only size property reports registered bytes,
including after close(). I/O through this object is limited to that range.

Creating another object for the same address, even with a different size, raises
CuFileError with MEMORY_ALREADY_REGISTERED. Overlapping tracked ranges are also
rejected. The wrapper never shares or enlarges an existing registration. Registrations
created elsewhere are not adopted; native failures propagate as CuFileError.
close() drains pending I/O before native deregistration. Closing the registration leaves the buffer allocated.
Do not remap or resize registered memory.

The wrapper obtains the buffer pointer and byte count and retains its owner; it does
not maintain an allocator-class support list. Before calling cuFileBufRegister
for device memory, it queries CUDA pointer attributes. Device memory whose
IS_GPU_DIRECT_RDMA_CAPABLE attribute is false raises NotImplementedError in this
version, including when compatibility fallback is allowed or forced. Native cuFile
then validates eligible buffers for the active mode and system. A registration
failure does not rule out unregistered I/O.
Python buffer-protocol memory also keeps an active export throughout registration,
pending I/O. bytearray resizing and mmap.close() raise
BufferError until that export is released after native use ends.
GDS device-buffer registration requires memory eligible for native peer-DMA mapping.
For VMM allocations, request gpu_direct_rdma=True. Device memory without the required
GPUDirect capability is not supported for registration in either mode because native
registration does not reliably fall back. Managed memory registration is also
unsupported in both modes. Unregistered I/O can use internal
staging buffers. Direct storage-to-GPU I/O also requires a supported GPU, driver stack,
and filesystem/storage path.
See :doc:`../cufile` for allocation meanings and registration support by mode.

These rules apply to the underlying CUDA memory, not just cuda.core allocators.
Conventional cudaMalloc/cuMemAlloc memory, VMM, stream-ordered pools, pinned/pageable
host memory, and managed memory follow their native mapping requirements even when
exposed by CuPy, RMM, Numba, or CUDA tensors. A cudaMalloc-backed caching pool is not
the same allocation type as a CUDA stream-ordered pool. Supported contiguous CUDA
Array Interface or Python buffer-protocol objects provide the pointer, bounds, and
owner. Bare device addresses need an owning view, such as a CuPy UnownedMemory-backed
array; the caller must keep externally owned memory valid. DLPack-only CUDA objects
need a supported view first. The guide includes external-allocation examples; a view
does not change the backing memory's registration eligibility.

The required buf has no default. flags defaults to 0 and is passed to
cuFileBufRegister; the native argument is reserved, so nonzero flags raise
ValueError before touching the buffer or driver. The bindings' reference to
CU_FILE_RDMA_REGISTER does not define an additional supported flag. See the
`native buffer registration specification
<https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilebufregister>`_.

RegisteredStream
~~~~~~~~~~~~~~~~

.. code-block:: text

   RegisteredStream(stream: object, flags: int = 0)
   flags: int                # read-only native bitmask

RegisteredStream calls cuFileStreamRegister once and owns that registration. Another
object for the same stream/context raises CuFileError with INVALID_VALUE, whether its
flags match or differ. close() drains pending I/O and deregisters the stream; the CUDA
stream stays open. Registrations created elsewhere are not adopted.

The required stream has no default. flags defaults to 0: size and offsets are read
at execution time, with no alignment promise. Flags 0x1/0x2/0x4 fix
buffer_offset/file_offset/size at submission. Flag 0x8 requires
4096-byte alignment of the effective buffer address, offsets, and size. Unknown flags
are rejected. Combine bits with |; 0xF fixes all
three arguments and promises alignment. It is an explicit optimization for workloads
that satisfy those promises, rather than the default. Registration is not a readiness
barrier: establish stream dependencies or synchronize earlier CUDA work explicitly.
Closing a registration drains its I/O, and native deregistration may synchronize the
stream. Keep registration and deregistration outside the transfer loop.

Batch
~~~~~

.. code-block:: text

   Batch(capacity: int)
   capacity: int    # read-only property
   is_active: bool  # read-only property
   submit(operations: Iterable[BatchIOParams]) -> None
   poll(min_complete=0, max_complete=None,
        timeout=0.0) -> tuple[IOCompletion, ...]
   wait_all(timeout=None) -> tuple[IOCompletion, ...]
   cancel() -> None
   close(*, timeout=None) -> None

Batch(capacity) creates the native queue; initialize native cuFile with a File or
Driver first. Capacity limits requests submitted but not yet collected. Each submitted
request retains its file, buffer, and native parameter storage until a terminal
completion is collected. Entries can use several files and mix host/device buffers;
all outstanding device buffers must share one CUDA context. Completions may arrive out
of order. Reuse a buffer only after its previous completion has been collected.

submit(entries) requires a nonempty list that fits the remaining capacity and packs it
internally. After a completion, use dataclasses.replace(entry, file_offset=...) or a
new BatchIOParams and submit it. No prepare, clear, or separate wait method is needed.
libcufile 1.19 partial resubmission raises NotImplementedError before native mutation;
collect all outstanding completions before another submission on that version.

BatchIOParams
~~~~~~~~~~~~~

.. code-block:: text

   BatchIOParams(file: File, buf: object, opcode: IOOpcode | str,
            size: int, file_offset: int = 0, buffer_offset: int = 0,
            cookie: object = None)

BatchIOParams is an immutable description: File, buffer, operation, positive byte
count, offsets (default zero), and optional cookie. Creating one starts no I/O.
submit(entries) acquires its resource references. Change an entry with
dataclasses.replace(entry, file_offset=...) after its previous request completes;
the other outstanding entries remain unchanged.

IOCompletion
~~~~~~~~~~~~

.. code-block:: text

   index: int
   cookie: object
   status: cuda.bindings.cufile.Status
   bytes_transferred: int
   error: CuFileError | None = None
   native_result: int | None = None
   result() -> int

IOCompletion is an immutable result returned by polling/waiting. index is a unique,
increasing request number within one Batch, including across submit() calls. cookie
carries the caller's value; it need not be unique. native_result preserves the signed
native result. result() returns actual bytes or raises the transfer error. Returned
results remain usable after reuse or close and need no cleanup.

IOOpcode
~~~~~~~~

.. code-block:: text

   READ = "read"
   WRITE = "write"

IOOpcode is a string enum with READ and WRITE values. BatchIOParams accepts the enum or
the valid strings "read" and "write"; invalid values raise ValueError.

Native batch status
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   cuda.bindings.cufile.Status:
       WAITING = 1
       PENDING = 2
       INVALID = 4
       CANCELED = 8
       COMPLETE = 16
       TIMEOUT = 32
       FAILED = 64

IOCompletion.status uses cuda.bindings.cufile.Status directly, preserving the native
enum and numeric value. There is no separate status enum in cuda.core.cufile.
WAITING and PENDING describe unfinished requests; returned completion records must
have a terminal status. Status.TIMEOUT is a result for one native operation.
Python TimeoutError instead means a batch is still active when a wait or close
deadline expires.

CuFileError
~~~~~~~~~~~

.. code-block:: text

   CuFileError(message: str, native_code: int | None = None, *,
               cuda_error=None, errno=None, filename=None, strerror=None,
               operation=None, native_result=None)
   status: int | None         # alias of native_code; same status domain as bindings
   status_name: str | None    # bindings OpError name, or None for unknown code
   native_code: int | None
   cuda_error  # native CUDA status or None
   errno: int | None
   filename: str | None
   strerror: str | None
   operation: str | None
   native_result: int | None

CuFileError derives from CUDAError and represents both native cuFile and filesystem
failures. Inspect native_code/status_name for cuFile status and errno/filename/strerror
for available filesystem details. These are separate code domains. CuFileError is not
an OSError subclass; the original exception is retained as __cause__. Error objects
own no native resources and have no close() method.

Errors, CUDA contexts, and threads
----------------------------------

.. include:: ../_cufile_errors.rst

Native I/O and waits release the GIL and run in the required CUDA context; the caller's
context is restored afterward. Pending
I/O retain native file handles and cuda.core allocation/stream
handles. Keep resources owned by other libraries valid until their pending
I/O releases them. Before synchronous or batch I/O, finish earlier GPU work on the
buffer. For stream I/O, use stream order or explicit dependencies between streams; the
API does not synchronize a default stream automatically. Concurrent reads are supported.
The caller must coordinate close(), other mutation, and shared driver settings across
threads. A forked child cannot use the parent's cuFile runtime.

Performance
-----------

Create the Driver, files, buffers, registrations, and batch outside the transfer loop.
Reuse the native batch, entries, and existing buffers. Packing occurs in submit(). Each new Driver calls native
driver-open, and its first successful close balances that call. Repeating those calls
per transfer adds overhead even if cuFile is already initialized. Keep one Driver open
for the workload. Each nonempty read/write request makes one cuFile transfer.

Measure sync, stream, and batch I/O separately. Include Python argument checks, resource
references, and the private stream completion event. Report actual bytes, latency,
throughput, warm-up, and data verification.

Successful I/O, an O_DIRECT descriptor, or a driver property does not prove which DMA
path was used. A completed write does not mean os.fsync() has been called.

Examples
--------

Shared setup
~~~~~~~~~~~~

The examples share these imports and CUDA setup. NumPy and CuPy demonstrate host/device
buffer inputs. Each sample creates private temporary files under /mnt/nvme0.

.. code-block:: python

   from pathlib import Path
   import tempfile
   import numpy as np
   import cupy as cp
   from cuda.core import Device, cufile

   device = Device(0)
   device.set_current()
   block = 4096
   expected = np.arange(block, dtype=np.uint8)

Synchronous file I/O
~~~~~~~~~~~~~~~~~~~~

Read and write existing host or device buffers. Each method waits for I/O and returns
the actual byte count.

.. code-block:: python

   with tempfile.TemporaryDirectory(dir="/mnt/nvme0") as directory:
       path = Path(directory) / "data.bin"
       host = np.empty(block, dtype=np.uint8)
       gpu = cp.empty(block, dtype=cp.uint8)
       with cufile.File(path, "w+") as file:
           assert file.write(cufile.IOParams(expected, block)) == block
           assert file.read(cufile.IOParams(host, block)) == block
           assert file.read(cufile.IOParams(gpu, block)) == block
           assert file.write(cufile.IOParams(gpu, block)) == block
       np.testing.assert_array_equal(host, expected)
       np.testing.assert_array_equal(cp.asnumpy(gpu), expected)
       assert path.read_bytes() == expected.tobytes()

Stream file I/O
~~~~~~~~~~~~~~~

The supplied stream runs the transfers in order. Wait for the stream, then read each
IOParams.bytes_done. Host buffers use the same interface.

.. code-block:: python

   stream = device.create_stream()
   try:
       with tempfile.TemporaryDirectory(dir="/mnt/nvme0") as directory:
           source = Path(directory) / "input.bin"
           output = Path(directory) / "output.bin"
           source.write_bytes(expected.tobytes())
           gpu = cp.empty(block, dtype=cp.uint8)
           host = bytearray(block)
           with (
               cufile.File(source) as reader,
               cufile.File(output, "w+") as writer,
               cufile.RegisteredStream(stream) as io_stream,
               cufile.IOParams(gpu, block) as read_params,
               cufile.IOParams(gpu, block) as write_params,
               cufile.IOParams(host, block) as host_params,
           ):
               # The same stream orders the read, write, and host read.
               reader.read_async(read_params, io_stream)
               writer.write_async(write_params, io_stream)
               writer.read_async(host_params, io_stream)
               stream.sync()
               assert read_params.bytes_done == block
               assert write_params.bytes_done == block
               assert host_params.bytes_done == block
               assert host == expected.tobytes()
           assert output.read_bytes() == expected.tobytes()
   finally:
       stream.sync()
       stream.close()

Multi-file batch I/O
~~~~~~~~~~~~~~~~~~~~

Each BatchIOParams supplies its own File and buffer. One batch can mix host/device
entries. This example needs all results, so it calls wait_all() directly.

.. code-block:: python

   with tempfile.TemporaryDirectory(dir="/mnt/nvme0") as directory:
       first_path = Path(directory) / "a.bin"
       second_path = Path(directory) / "b.bin"
       first_path.write_bytes(expected.tobytes())
       second_path.write_bytes(expected[::-1].tobytes())
       host = bytearray(block)
       gpu = cp.empty(block, dtype=cp.uint8)
       with (
           cufile.File(first_path) as first,
           cufile.File(second_path) as second,
           cufile.Batch(2) as batch,
       ):
           batch.submit([
               cufile.BatchIOParams(first, host, "read", block, cookie="host"),
               cufile.BatchIOParams(second, gpu, "read", block, cookie="gpu"),
           ])

           results = batch.wait_all(timeout=10)
           assert [item.cookie for item in results] == ["host", "gpu"]
           assert [item.result() for item in results] == [block, block]
       assert host == expected.tobytes()
       np.testing.assert_array_equal(cp.asnumpy(gpu), expected[::-1])

Driver and file scopes
~~~~~~~~~~~~~~~~~~~~~~

Driver opens when constructed. The file's with block ends first, so File.close() runs
before Driver.close().

.. code-block:: python

   with tempfile.TemporaryDirectory(dir="/mnt/nvme0") as directory:
       path = Path(directory) / "scoped.bin"
       with cufile.Driver() as driver:
           assert not driver.is_closed
           with cufile.File(path, "w+") as file:
               assert file.write(cufile.IOParams(expected, block)) == block
           assert file.is_closed
       assert driver.is_closed
       assert path.read_bytes() == expected.tobytes()

Buffer registration scope
~~~~~~~~~~~~~~~~~~~~~~~~~

Registration keeps the existing memory alive for I/O. Removing it leaves the allocation
usable. The allocation must support cuFile registration.

.. code-block:: python

   with tempfile.TemporaryDirectory(dir="/mnt/nvme0") as directory:
       path = Path(directory) / "registered.bin"
       gpu = cp.asarray(expected)
       cp.cuda.get_current_stream().synchronize()  # Producer readiness.
       with (
           cufile.RegisteredBuffer(gpu) as buffer,
           cufile.File(path, "w") as file,
       ):
           assert file.write(cufile.IOParams(buffer, block)) == block
       assert buffer.is_closed
       # Deregistration leaves the application's allocation usable.
       np.testing.assert_array_equal(cp.asnumpy(gpu), expected)
       assert path.read_bytes() == expected.tobytes()

Rolling batch I/O
~~~~~~~~~~~~~~~~~

Use dataclasses.replace() after an entry completes. The other entry can remain
outstanding. libcufile 1.19 rejects this path before native submission; use full drains
on that version. The complete runnable example is
:cuda-core-example:`cufile_rolling_batch.py <cufile_rolling_batch.py>`.

.. code-block:: python

   from dataclasses import replace

   # file contains count blocks; keep two read buffers in flight.
   buffers = [bytearray(block), bytearray(block)]
   entries = [
       cufile.BatchIOParams(file, buf, "read", block, file_offset=i * block, cookie=i)
       for i, buf in enumerate(buffers)
   ]
   with cufile.Batch(2) as batch:
       batch.submit(entries)
       next_block = 2
       while batch.is_active:
           for completion in batch.poll(min_complete=1, max_complete=1, timeout=None):
               completion.result()
               slot = completion.cookie
               consume(buffers[slot])  # Finish consuming this data before reusing it.
               if next_block < count:
                   entries[slot] = replace(entries[slot], file_offset=next_block * block)
                   batch.submit([entries[slot]])
                   next_block += 1

Repeated I/O and cleanup
~~~~~~~~~~~~~~~~~~~~~~~~

Create the Driver, registrations, and files before the transfer loop. Leave their with
blocks after all transfers to close them.

.. code-block:: python

   with tempfile.TemporaryDirectory(dir="/mnt/nvme0") as directory:
       path = Path(directory) / "loop.bin"
       source = bytearray(expected.tobytes())
       destination = bytearray(block)
       with (
           cufile.Driver(),
           cufile.RegisteredBuffer(source) as registered,
           cufile.File(path, "w+") as file,
       ):
           for _ in range(3):
               assert file.write(cufile.IOParams(registered, block)) == block
               assert file.read(cufile.IOParams(destination, block)) == block
       assert destination == source
       assert file.is_closed and registered.is_closed

Explicit cleanup without context managers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These examples use close() instead of with. Import cufile from cuda.core; the caller
supplies paths, ready buffers, streams, and size in bytes unless shown otherwise. Close
objects created by each example; the caller closes supplied resources.

BatchIOParams, IOCompletion, IOOpcode, and bindings Status values have no close().

Driver lifetime
^^^^^^^^^^^^^^^

Driver() calls native driver-open immediately; finally calls close() for that reference.
Version and parameter queries use standalone bindings functions and return Python
values. Closing the Driver does not restore settings.

.. code-block:: python

   from cuda.bindings import cufile as bindings

   driver = cufile.Driver()
   try:
       version = bindings.get_version()
       poll_mode = bindings.get_parameter_bool(bindings.BoolConfigParameter.PROPERTIES_USE_POLL_MODE)
   finally:
       driver.close()

Synchronous I/O
^^^^^^^^^^^^^^^

This example creates and closes the File. source and destination are existing buffers
provided by the caller. IOParams can be reused; sync results come directly from the
method and do not require bytes_done.

.. code-block:: python

   write_params = cufile.IOParams(source, size)
   read_params = cufile.IOParams(destination, size)
   file = cufile.File(path, "w+")
   try:
       assert file.write(write_params) == size
       assert file.read(read_params) == size
   finally:
       file.close()

Stream I/O
^^^^^^^^^^

The caller provides the file, buffer, and stream. The helper queues I/O and returns
IOParams without waiting. Queue other stream work if needed, wait for completion, then
read bytes_done. Keep params if the result is needed. close() or finalization can wait
for pending I/O; the completed result remains readable after close().

.. code-block:: python

   def enqueue_read(file, buffer, stream, size):
       params = cufile.IOParams(buffer, size)
       file.read_async(params, stream)
       return params

   params = enqueue_read(file, buffer, stream, size)
   try:
       # Enqueue dependent computation on this stream or do other work here.
       stream.sync()  # The caller chooses the completion point.
       bytes_read = params.bytes_done
   finally:
       params.close()

   assert params.bytes_done == bytes_read  # The completed result remains readable.

Buffer registration
^^^^^^^^^^^^^^^^^^^

buffer is existing memory that is ready and supports registration, such as a core VMM
allocation with gpu_direct_rdma=True or pinned host memory. The example reuses one
registration across reads and closes it afterward. Closing the registration does not
free buffer. size is the number of bytes to register; omit it to register the supplied
buffer's full length. A second object for the same or an overlapping range raises
CuFileError, even if it requests a different size.

.. code-block:: python

   registered_buffer = cufile.RegisteredBuffer(buffer, size=size)
   try:
       for _ in range(3):
           assert file.read(cufile.IOParams(registered_buffer, size)) == size
   finally:
       registered_buffer.close()

Stream registration
^^^^^^^^^^^^^^^^^^^

The caller provides an existing cuda.core Stream, an open read/write File, and two
IOParams objects for source and destination. The stream runs the write before the read.
Closing RegisteredStream leaves the CUDA stream open. The caller closes the IOParams
objects separately.

.. code-block:: python

   registered_stream = cufile.RegisteredStream(stream)
   try:
       file.write_async(write_params, registered_stream)
       file.read_async(read_params, registered_stream)
       stream.sync()
       assert write_params.bytes_done == size
       assert read_params.bytes_done == size
   finally:
       registered_stream.close()

   assert not stream.is_closed

Borrowed stream interoperability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CuPy owns the stream. The caller provides the File and destination IOParams. This
example closes the registration and borrowed wrapper, while keeping the CuPy owner and
CUDA context alive until I/O completes.

.. code-block:: python

   import cupy as cp
   from cuda.core import Stream

   owner = cp.cuda.Stream(non_blocking=True)
   borrowed = Stream.from_handle(owner.ptr)
   try:
       registration = cufile.RegisteredStream(borrowed)
       try:
           file.read_async(params, registration)
           owner.synchronize()
           bytes_read = params.bytes_done
       finally:
           registration.close()
   finally:
       borrowed.close()

   owner.synchronize()  # The CuPy-owned stream remains usable.

Batch I/O
^^^^^^^^^

first and second are open File objects supplied by the caller. Each entry supplies its
operation and cookie. poll() returns new results once; wait_all() returns only remaining
results. Combine the two tuples explicitly when the application needs both. Returned
results remain readable after the batch closes.

.. code-block:: python

   from cuda.bindings import cufile as bindings

   batch = cufile.Batch(2)
   try:
       batch.submit([
           cufile.BatchIOParams(first, first_buffer, cufile.IOOpcode.READ, size, cookie="first"),
           cufile.BatchIOParams(second, second_buffer, cufile.IOOpcode.READ, size, cookie="second"),
       ])

       polled = batch.poll(max_complete=2)
       remaining = batch.wait_all(timeout=10)
       results = polled + remaining
   finally:
       batch.close()

   counts = [result.result() for result in results]
   cookies = [result.cookie for result in results]
   assert all(result.status == bindings.Status.COMPLETE for result in results)

Error handling
^^^^^^^^^^^^^^

Opening a missing path in read mode raises CuFileError with errno.ENOENT, the
filename, and the original FileNotFoundError as __cause__. errno identifies an OS failure;
native_code identifies a cuFile failure. If construction fails, it cleans up any file
descriptor it opened, so the caller has no File object to close.

.. code-block:: python

   try:
       file = cufile.File(missing_path)
   except cufile.CuFileError as error:
       os_errno = error.errno
       native_code = error.native_code
   else:
       file.close()

Validation requirements
-----------------------

Validate sync, stream, and batch transfers with host/device buffers, offsets, short
results, and byte-for-byte payload checks. Cover registration capabilities and bounds,
duplicate requests, failure rollback, independent driver references, and resource
lifetimes through completion. Validate interoperability,
CUDA-context restoration, exceptions, and cleanup across supported backends.

Unsupported native features raise NotImplementedError before native mutation.
Cancellation requires reliable completion reporting; wait_all() remains the completion
path when cancellation is unavailable. Hardware/version matrices, commands, measured
results, and trace findings belong in :doc:`cufile-validation`.

Possible native API changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A native session handle could give each library its own settings and define who closes
its registrations. This requires changes to cuFile itself and is outside this Python API
proposal.

Open review item: context managers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open requests: allow later with blocks to reuse the same manager, and reject nesting
that manager before any native call. Current classes close on exit and allow nesting.
Adding reuse must not silently reopen files or repeat native registrations.

References
----------

* `cuFile API reference <https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html>`_.
* `GDS best practices <https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html>`_.
* `GDS release notes <https://docs.nvidia.com/gpudirect-storage/release-notes/index.html>`_.
* :doc:`../cufile` for the complete API reference.

.. toctree::
   :maxdepth: 1

   cufile-validation

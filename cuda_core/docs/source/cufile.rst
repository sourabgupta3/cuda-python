.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

GPUDirect Storage with cuFile
=============================

``cuda.core.cufile`` provides Pythonic file I/O on Linux over ``cuda.bindings.cufile``.
Native libcufile selects a GDS path when its GPU, memory, driver, filesystem, and
backend requirements are met, or POSIX compatibility mode with host-memory staging
when native configuration allows or forces it. Routing follows cuFile's support
matrix and configuration; a successful Python call does not guarantee a GDS transfer.

Use synchronous methods, CUDA stream I/O, or batches that can use several files.
The classes validate arguments, retain resources, restore CUDA contexts, and report
errors. CUDA graphs are not supported, including capturing cuFile stream submissions
with ordinary buffers.

The system needs libcufile, a supported CUDA driver, and a compatible storage
filesystem. Importing ``cuda.core`` does not initialize cuFile; the library is loaded
when first needed. This is an unreleased prototype. CUDA 12, packaging, other systems,
and free-threaded Python still need testing.

cuFile settings are shared across the process. This module does not track registrations
made elsewhere, so their owners must coordinate access. A with block calls close() on
exit, even after a body exception. A closed object cannot enter another block. Avoid
nesting blocks using the same object: the inner exit closes it before the outer block
ends. See :doc:`design/cufile` for examples and open review items.

File operations
---------------

.. include:: _cufile_defaults.rst

.. list-table:: Execution and completion
   :header-rows: 1

   * - Methods
     - Native operation
     - Completion
   * - ``File.read`` / ``write``
     - One synchronous cuFile transfer on the calling host thread
     - Returns actual bytes transferred
   * - ``File.read_async`` / ``write_async``
     - Native I/O on the supplied CUDA stream
     - Uses ``IOParams``; read ``bytes_done`` after stream/event completion
   * - ``Batch.submit``
     - Native submission of operations with individual file handles
     - ``poll`` copies available completions; ``wait_all`` returns all results and checks transfer errors

``File(path, mode="r", *, flags=os.O_DIRECT)`` opens and registers one file descriptor.
``mode`` selects access, creation, and truncation: ``r``, ``r+``, ``w``, and ``w+``.
``flags`` supplies os.open modifiers and defaults to ``os.O_DIRECT``.
Pass ``flags=0`` to open without ``O_DIRECT``. Supplying flags replaces the default.

``fileno()`` returns the descriptor
owned by File, for example for ``os.fsync()``; do not close it separately. A completed
write does not by itself mean the data is durable; ``O_DIRECT`` alone does not ensure
durability. Use ``os.fsync()`` when durability is required.
Opening without ``O_DIRECT`` does not force page-cache I/O: cuFile may still use a
direct GPU/storage path. See the `native handle registration reference
<https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilehandleregister>`_.

For example, after earlier work on ``source`` has finished::

   import os
   from cuda.core import cufile

   with cufile.File("direct.bin", "w+", flags=os.O_DIRECT) as file:
       assert file.write(cufile.IOParams(source)) == source.nbytes

   file = cufile.File("without-direct.bin", "w+", flags=0)
   try:
       assert file.write(cufile.IOParams(source)) == source.nbytes
   finally:
       file.close()

``IOParams(buf, size=None, *, file_offset=0, buffer_offset=0)`` supplies arguments for
both sync and stream I/O. ``read(params)`` and ``write(params)`` return the completed
byte count directly; errors raise at the call. Sync calls do not set ``bytes_done`` or
create CUDA completion events. Sizes and offsets are bytes. ``size=None`` transfers
the bytes remaining after ``buffer_offset``.
EOF and short transfers return actual counts. Invalid bounds, negative offsets,
read-only destinations, and noncontiguous buffers are rejected before I/O submission.

Core allocations can be rounded up to a larger capacity. Pass the intended byte count
when the data uses only part of the allocation.

All three I/O methods accept contiguous host or device buffers: cuda.core Buffer, CUDA
Array Interface arrays, and Python buffer-protocol objects. Both C and F contiguous
layouts work. Host examples include NumPy arrays, bytearray, mmap, and pinned core
allocations. Stream I/O still uses a CUDA stream/context. The API adds no host
scheduler, automatic transfer splitting, or executor.

Before synchronous calls or batch submission, finish earlier GPU work that uses the
buffer. The API does not synchronize a default stream automatically. Stream I/O runs in
order on the supplied stream; use explicit dependencies for work on other streams.
Integer handles are borrowed. For core streams, pending I/O keeps the native stream
resource alive.

For example, after earlier work on the supplied buffers has finished::

   from cuda.core import cufile

   with cufile.File("data.bin", "w+") as file:
       write_params = cufile.IOParams(source)
       read_params = cufile.IOParams(destination)
       assert file.write(write_params) == source.nbytes
       assert file.read(read_params) == destination.nbytes
       file.write_async(write_params, stream)
       file.read_async(read_params, stream)
       stream.sync()
       assert write_params.bytes_done == source.nbytes
       assert read_params.bytes_done == destination.nbytes

See :cuda-core-example:`cufile_io.py <cufile_io.py>` for a complete application using
core VMM buffers, buffer/stream registration, data verification, and measurements after
warm-up.

Stream arguments, results, and cleanup
--------------------------------------

Stream I/O requires cuFile's CUDA stream-memory-operation support and a usable native
execution backend. The POSIX stream path requires an enabled cuFile worker pool:
execution.parallel_io=True and execution.max_io_threads > 0. Direct userspace-RDMA
stream I/O may require compatibility mode. Successful sync I/O does not establish
stream support. Native failures propagate without a synchronous or Python-thread-pool
substitute. CUDA graph capture is outside the scope of cuFile stream I/O, including
transfers using ordinary allocations.

``IOParams`` supplies arguments for both execution methods. For stream I/O it also
keeps native arguments and the result alive. ``read_async(params, stream)`` and ``write_async`` queue
I/O and return None. Call ``stream.sync()`` or wait for an event recorded after the
transfer, then read ``params.bytes_done``. It returns actual bytes or raises the
transfer error. Reading it does not wait: access while I/O is pending raises
RuntimeError. Repeated reads use the saved result. IOParams has no ``wait()`` method or
separate future object. Reuse an open IOParams after its previous transfer completes.
Sync calls clear any previous stream result; ``bytes_done`` then raises RuntimeError
until another stream transfer is submitted and completes. Sync results are returned
directly by ``read`` and ``write``.

Set size and offsets before submission unless explicitly ordering later updates. Stream
registration flags mark which fields are fixed at submission; changing a fixed field
while pending raises ValueError. Later sizes cannot exceed the submitted maximum, and
offsets must remain inside the buffer. For unregistered memory, the prototype adds
buffer_offset to the submitted address, so changing that offset while pending requires
registered memory. Finish any later update before cuFile reads the value.

``IOParams.close()`` and context exit wait for pending I/O before freeing arguments or
resource references. If recording a completion event fails after submission, the
implementation waits for the stream before raising the error. If it cannot confirm I/O
has stopped, it keeps the resources alive.

File, IOParams, Driver, RegisteredBuffer, RegisteredStream, and Batch support close()
and with. File.close() waits for I/O through that file and can drain a batch using it.
Collect batch results first if the application needs them. is_closed reports whether
the Python object is closed, rather than whether every native reference was released.

Pending I/O keeps native cuda.core allocation/stream handles, file state, CUDA context,
and argument/result memory alive. This protects it when a caller closes the original
Buffer or Stream wrapper. Finalizer cleanup failures issue ``CUDAWarning``. Normal
runtime shutdown waits for owned work and closes resources before explicit Driver
objects; cleanup during late interpreter finalization is not guaranteed.
Keep resources owned by other libraries
valid until pending I/O releases them.

Host buffer exports stay active through registration and pending I/O. Batch completion
collection releases each request's export unless another owner, such as an open
registration, still holds it. bytearray cannot be resized while an export exists.
Do not change a buffer's contents while I/O uses it.

Explicit cleanup without context managers
-----------------------------------------

These examples use close() instead of with. Import cufile from cuda.core; the caller
supplies paths, ready buffers, streams, and size in bytes unless shown otherwise. Close
objects created by each example; the caller closes supplied resources.

BatchIOParams, IOCompletion, IOOpcode, and bindings Status values have no close().

Driver lifetime
~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~

This example creates and closes the File. source and destination are existing buffers
provided by the caller.

.. code-block:: python

   file = cufile.File(path, "w+")
   try:
       assert file.write(cufile.IOParams(source, size)) == size
       assert file.read(cufile.IOParams(destination, size)) == size
   finally:
       file.close()

Stream I/O
~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~

buffer is existing memory that is ready and supports registration, such as a core VMM
allocation with gpu_direct_rdma=True or pinned host memory. The example reuses one
registration across reads and closes it afterward. Closing the registration does not
free buffer.

.. code-block:: python

   registered_buffer = cufile.RegisteredBuffer(buffer, size=size)
   try:
       for _ in range(3):
           assert file.read(cufile.IOParams(registered_buffer, size)) == size
   finally:
       registered_buffer.close()

Stream registration
~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~

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
~~~~~~~~~~~~~~

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

Registration
------------

``RegisteredBuffer(buf, size=None, *, flags=0)`` adds a cuFile registration to existing
host/device memory. size=None uses the supplied buffer's byte length starting at its
pointer; a view is not expanded to the underlying allocation. An explicit size must be
positive and fit the supplied buffer. The read-only size property reports registered
bytes, including after close(). I/O through this object is limited to that range.
The native buffer flags are reserved: only 0 is supported. Nonzero flags raise
ValueError before accessing the buffer or driver. Both registration objects expose
their supplied bitmask through a read-only ``flags`` property.
``RegisteredStream(stream, flags=0)`` registers an existing CUDA stream. Neither
allocates memory nor creates a stream. Each object owns one native registration;
there is no sharing between registration objects. Duplicate/overlapping tracked
buffer ranges raise CuFileError with MEMORY_ALREADY_REGISTERED, even with different
sizes. Duplicate tracked streams raise CuFileError with INVALID_VALUE, even with
matching flags. A failed duplicate request leaves the original object valid.
close() drains pending I/O before deregistration. Registrations created elsewhere are not taken over,
and native errors propagate. Closing a registration leaves the buffer or stream alive.

Stream flags default to 0: size and offsets are read at execution time, with no
alignment promise. Registration is not a readiness barrier; establish dependencies or
synchronize earlier CUDA work explicitly. Deregistration can synchronize the stream,
so keep registration and deregistration outside transfer loops. Flags
0x1/0x2/0x4 make buffer_offset/file_offset/size fixed at submission. Flag ``0x8``
requires 4096-byte alignment of the effective buffer address, file offset, and size; the
wrapper checks these before submitting I/O. Unknown flags are rejected.

Buffers are not limited to cuda.core allocations. The API accepts contiguous CUDA
Array Interface objects, including CuPy arrays, RMM DeviceBuffer, Numba device arrays,
and CUDA tensors that provide that interface. Contiguous NumPy arrays, bytearray,
mmap, and pinned host views use the Python buffer protocol. Read destinations must
be writable. These interfaces describe an existing memory view; they do not allocate
or copy its data. Sizes and offsets are bytes.

The table defines this version's explicit RegisteredBuffer registration support for the
underlying memory, regardless of which library allocated it. GDS
device-buffer registration requires GPU memory eligible for native peer-DMA mapping.
Compatibility mode uses POSIX I/O and host staging instead of direct GPU mapping.
All entries require native configuration/backend support and sufficient resources.
Host registration requires properties.allow_system_memory=True. Device registration
requires a valid CUDA context and memory eligible for the selected mapping backend.
Allowing compatibility fallback does not force compatibility or make an ineligible
explicit registration succeed; native registration failures propagate.

.. list-table:: Buffer registration requirements by mode
   :header-rows: 1
   :widths: 46 27 27

   * - Allocation and examples from CUDA/Python libraries
     - GDS mode
     - Compatibility mode
   * - Conventional device memory: ``cudaMalloc`` / ``cuMemAlloc``;
       CuPy Memory and CUDA-malloc-backed CuPy/RMM pools
     - Supported when eligible for native peer-DMA mapping
     - Supported
   * - ``VirtualMemoryResource`` with ``gpu_direct_rdma=True`` in its options
       (native ``cuMemCreate`` / ``cuMemMap`` with RDMA-capable backing)
     - Supported
     - Supported
   * - ``VirtualMemoryResource`` without GPUDirect capability
       (native VMM without the required peer-DMA capability)
     - Unsupported
     - Unsupported
   * - ``DeviceMemoryResource`` allocation without GPUDirect capability
       (also apply mapping checks to ``cudaMallocAsync`` / ``cuMemAllocFromPoolAsync``,
       CuPy malloc_async, and RMM CudaAsyncMemoryResource)
     - Unsupported
     - Unsupported
   * - ``PinnedMemoryResource`` host pool allocation
       or external pinned host pools, such as CuPy's pinned-memory pool
     - Accepted as host memory
     - Accepted as host memory
   * - ``LegacyPinnedMemoryResource``
       or ``cudaHostAlloc`` / ``cudaMallocHost`` / ``cuMemHostAlloc`` memory
     - Accepted as host memory
     - Accepted as host memory
   * - Pageable host memory: malloc-backed views, NumPy, bytearray, mmap
     - Accepted as host memory
     - Accepted as host memory
   * - ``ManagedMemoryResource``
       or ``cudaMallocManaged`` / ``cuMemAllocManaged``;
       CuPy malloc_managed and RMM ManagedMemoryResource
     - Unsupported
     - Unsupported

Request gpu_direct_rdma=True when creating VMM memory for registration. In this
version, RegisteredBuffer rejects device memory whose CUDA
IS_GPU_DIRECT_RDMA_CAPABLE attribute is false with NotImplementedError before
calling cuFileBufRegister. This restriction applies
in GDS mode and when compatibility fallback is allowed or forced. Native
registration does not reliably fall back for these allocations. Managed memory
registration is unsupported in both modes.
Pool names alone do not identify registration support: a cudaMalloc-backed caching
pool differs from a CUDA stream-ordered pool. Changing a framework's allocator can
change its backing memory and eligibility. CUDA pointer attributes help identify
memory type and RDMA capability; native cuFile still validates the selected backend.

Direct storage-to-GPU I/O additionally requires a supported GPU, cuFile/driver stack,
and filesystem/storage path. Forcing compatibility selects POSIX I/O through host
memory. If the GDS path is unavailable and compatibility fallback is disabled, I/O
can fail. Accepting host-memory registration does not imply storage-to-GPU DMA.

The registration restrictions above do not prohibit unregistered I/O. cuFile may stage
through internal buffers; successful I/O does not prove direct DMA to the application's
allocation. Registration does not allocate memory. Pass the intended byte count when
an allocation is larger than the data.
Managed-memory ``read`` / ``write`` can use internal staging without explicit
registration; the table's unsupported registration entry does not prohibit that I/O.

VirtualMemoryResource maps explicitly allocated physical memory into a reserved virtual
address range; gpu_direct_rdma=True requests RDMA-capable backing memory at allocation
time. DeviceMemoryResource supplies GPU memory from a stream-ordered pool.
PinnedMemoryResource supplies pinned host RAM from a pool; LegacyPinnedMemoryResource
uses the older synchronous host allocation API. ManagedMemoryResource supplies unified
memory that CUDA can migrate between host and GPU.

RegisteredBuffer does not accept or reject Python allocator classes by name. It obtains
the pointer and available byte count, retains the memory owner, and checks CUDA
pointer attributes before registering device memory. For eligible buffers it calls
native cuFileBufRegister with that pointer, the requested size, and flags=0. Native cuFile
validates the allocation for the current mode/system and reports registration failures.
For Python buffer-protocol memory, an active export is retained throughout registration,
pending I/O. bytearray resizing and mmap.close() raise BufferError
while the export is held; it is released only after native use ends. Retaining the
Python owner alone is insufficient to protect the pointer.
The wrapper's host/device accessibility and CUDA-context information supports argument
handling and lifetime management; it is not a complete registration-support test.

External allocation examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an existing ``path`` containing at least 4096 bytes, ordinary CuPy and NumPy
buffers work without explicit registration. Finish earlier producer work first:

.. code-block:: python

   import cupy as cp
   import numpy as np
   from cuda.core import cufile

   gpu = cp.empty(4096, dtype=cp.uint8)
   host = np.empty(4096, dtype=np.uint8)
   cp.cuda.get_current_stream().synchronize()
   with cufile.File(path) as file:
       gpu_count = file.read(cufile.IOParams(gpu))
       host_count = file.read(cufile.IOParams(host))
   assert gpu_count == host_count == 4096

For an existing linear CUDA allocation, ``pointer`` is its integer address, ``size``
is its valid byte count, and ``owner`` is the object responsible for its lifetime.
Expose a view without copying, then register it if its backing memory is eligible:

.. code-block:: python

   import cupy as cp
   from cuda.core import cufile

   memory = cp.cuda.UnownedMemory(pointer, size, owner)
   view = cp.ndarray(
       (size,), dtype=cp.uint8,
       memptr=cp.cuda.MemoryPointer(memory, 0),
   )
   with cufile.RegisteredBuffer(view, size=size) as registered:
       with cufile.File(path) as file:
           bytes_read = file.read(cufile.IOParams(registered, size))
   assert bytes_read == size

Bare integer addresses and opaque CUDA arrays are not accepted directly. A DLPack-only
CUDA object needs a supported view, for example ``cp.from_dlpack(obj)``; that view
does not change its memory type or registration eligibility. The wrapper retains the
view/owner, but cannot prevent an external ``cudaFree``, ``cuMemFree``, resize, remap,
or pool reset. Keep the allocation valid through pending I/O,
and deregister before freeing it. In the raw-pointer example, the native caller still
owns allocation and cleanup; the integer returned by bindings does not own memory.
See `CuPy interoperability
<https://docs.cupy.dev/en/stable/user_guide/interoperability.html>`_ for pointer and
DLPack views and the `native cuFile memory contract
<https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilebufregister>`_.

cuda.core allocation example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an allocation that is ready to use and an open file::

   from cuda.core import VirtualMemoryResource, VirtualMemoryResourceOptions

   memory = VirtualMemoryResource(
       device, VirtualMemoryResourceOptions(gpu_direct_rdma=True)
   )
   with memory.allocate(size, stream=stream) as allocation:
       stream.sync()  # Establish allocation readiness before registration.
       with cufile.RegisteredBuffer(allocation) as buffer:
           with cufile.RegisteredStream(stream) as registered_stream:
               with cufile.IOParams(buffer, size) as params:
                   file.read_async(params, registered_stream)
                   stream.sync()
                   count = params.bytes_done

To use the same registration/I/O code with pinned host memory, select another memory
resource::

   from cuda.core import LegacyPinnedMemoryResource

   memory = LegacyPinnedMemoryResource()

Direct stream registration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep the RegisteredStream object open across the transfer loop. Leaving its with block
closes the cuFile registration. Close the CUDA stream separately with stream.close()
when the application has finished using it. For example::

   from cuda.core import Device, cufile

   device = Device(0)
   device.set_current()
   stream = device.create_stream()

   try:
       with cufile.RegisteredStream(stream) as registered:
           assert not registered.is_closed
           # Submit transfers here; reuse this registration across the loop.
       assert registered.is_closed
       assert not stream.is_closed
   finally:
       stream.close()

Streams created by other libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stream.from_handle(handle) wraps an existing CUDA stream without taking responsibility
for destroying it. Here CuPy owns the stream, and its .ptr supplies the handle. Keep the
CuPy owner and CUDA context alive until all transfers finish. Closing RegisteredStream
or the borrowed cuda.core wrapper leaves the CUDA stream owned by CuPy. For example::

   import cupy as cp
   from cuda.core import Device, Stream, cufile

   device = Device(0)
   device.set_current()
   owner = cp.cuda.Stream(non_blocking=True)
   borrowed = Stream.from_handle(owner.ptr)

   try:
       with cufile.RegisteredStream(borrowed) as registered:
           assert not registered.is_closed
   finally:
       borrowed.close()

   owner.synchronize()  # CuPy still owns the underlying CUDA stream.

Accepted streams include cuda.core Stream, __cuda_stream__ providers, and borrowed
integer handles. Direct __cuda_stream__ forwarding uses a deprecated core adapter and
needs review. Default streams belong to a CUDA context; the per-thread default also
depends on the calling thread. Native null stream 0 makes cuFile stream I/O synchronous.
Examples use explicit streams.

Native failures are reported to the caller. Do not change the mapping of registered
memory; deregister before remapping it. Core allocator defaults are unchanged.

Driver lifetime and standalone bindings
--------------------------------------

``Driver()`` immediately calls ``cuda.bindings.cufile.driver_open()`` (native
``cuFileDriverOpen``). Constructor failures raise an exception. ``Driver.close()`` calls
``driver_close()`` (currently native ``cuFileDriverClose_v2``) for that reference.
Entering with adds no open call. After close() succeeds, another close() has no effect.

Keep one Driver open around the workload. Creating and closing one for every transfer
adds native calls, even when cuFile is already initialized. Put the Driver's with block
outside the transfer loop. Leaving that block closes the Driver object even if it was
constructed earlier.

Importing the module does not open the driver. File registers its descriptor and relies
on cuFile's implicit initialization; it never explicitly calls driver-open or
driver-close. Buffer registration also uses native implicit initialization. Create a
File or Driver before Batch setup. The wrapper holds no hidden driver reference and
does not duplicate native reference counting. Keep an explicit Driver open until its
I/O and resource cleanup finish::

   with cufile.Driver():
       with cufile.File("input.bin") as file:
           count = file.read(cufile.IOParams(buffer))
       # The file is closed; the Driver remains open until its block exits.
   # Native driver-close is called; cuFile handles its remaining references.

Without an explicit Driver, native cuFile manages implicit initialization and
process-exit cleanup. File.close() releases the file, not the driver. For explicit
driver cleanup, construct Driver before the workload and close its resources before
leaving the Driver block. ``Driver.is_closed`` describes only that object, not whether
the process-wide driver is closed. A final native driver-close can invalidate live
registrations and files; Python references do not override that native lifetime.

Use bindings.get_version(), driver_get_properties(), and ``get_parameter_*`` /
``set_parameter_*`` directly. driver_get_properties() takes the address of a native
CUfileDrvProps_t structure. Use individual parameter getters when a Python bool, int, or
string is sufficient.

Set initialization-only options before Driver() or any library opens an I/O resource.
Typed parameter setters reject changes after driver-open. The legacy ``driver_set_*``
initialization setters can return success without changing settings after driver-open.
All libraries in the process share these settings. A property value does not identify
the DMA path used by a transfer. Driver.close() does not restore settings. For example,
set options at application startup::

   from cuda.bindings import cufile as bindings
   from cuda.core import cufile

   # Before any library opens the cuFile driver.
   parameter = bindings.BoolConfigParameter.PROPERTIES_USE_POLL_MODE
   bindings.set_parameter_bool(parameter, True)
   with cufile.Driver():
       poll_mode = bindings.get_parameter_bool(parameter)
       version = bindings.get_version()
       assert poll_mode

Driver properties report settings and capabilities, not the DMA path used by one
transfer.

Native batches
--------------

Each ``BatchIOParams`` describes one transfer: File, buffer, read/write operation,
explicit size, offsets, and optional cookie. One batch can mix host/device buffers,
files, and reads/writes. Device buffers must use one CUDA context. Entries can complete
out of order; a batch provides no ordering between entries and no transaction guarantee.
Capacity limits outstanding requests, including requests from multiple submit() calls.

``submit(entries)`` packs new entries internally and retains their resources until
completion is collected. It can add requests while others remain outstanding, up to
capacity, when the native version supports partial resubmission. libcufile 1.19 is
blocked before native submission because partial resubmission can corrupt memory.
``poll()`` delivers new results once. ``wait_all()`` returns only remaining results,
sorted by their submission index; earlier poll results are excluded. The batch keeps
no completion history. ``IOCompletion.result()`` returns actual bytes or raises its
error; returned completions remain usable after reuse or close::

   with cufile.File("a.bin") as a, cufile.File("b.bin") as b:
       with cufile.Batch(2) as batch:
           batch.submit([
               cufile.BatchIOParams(a, first_buffer, "read", 4096, cookie="a"),
               cufile.BatchIOParams(b, second_buffer, "read", 4096, cookie="b"),
           ])

           counts = [item.result() for item in batch.wait_all(timeout=10)]

If ``close(timeout=...)`` raises TimeoutError, the batch stays open for polling or
another close attempt. Cancellation never undoes completed writes. If the native
implementation cannot provide reliable completion reporting, cancel() raises
``NotImplementedError`` before mutation; use ``wait_all()`` to finish pending I/O.

See :cuda-core-example:`cufile_batch.py <cufile_batch.py>` for a complete two-file
read/write example using registered core VMM memory.

Errors, CUDA contexts, and threads
----------------------------------

.. include:: _cufile_errors.rst

Native calls use the required CUDA context and restore the caller's context afterward.
Native I/O and waits release the GIL. The :doc:`core concurrency policy <concurrency>`
applies: coordinate close(), other mutation of a shared object, and shared driver
settings across threads. Concurrent reads are supported; do not mutate one batch from
several threads at once. A forked child cannot use the parent's cuFile runtime.

Validation
----------

Tests and applications create private temporary directories on the selected mount. See
:doc:`design/cufile-validation` for commands, results, measurements, and the
systems/features that still need testing.

API reference
-------------

.. currentmodule:: cuda.core.cufile

.. autoclass:: File
   :members:
   :undoc-members:

.. autoclass:: IOParams
   :members:
   :undoc-members:

.. autoclass:: Driver
   :members:
   :undoc-members:

.. autoclass:: RegisteredBuffer
   :members:
   :undoc-members:

.. autoclass:: RegisteredStream
   :members:
   :undoc-members:

.. autoclass:: Batch
   :members:
   :undoc-members:

.. autoclass:: BatchIOParams
   :members:
   :undoc-members:

.. autoclass:: IOCompletion
   :members:
   :undoc-members:

.. autoclass:: IOOpcode
   :members:

Batch completion ``status`` values use ``cuda.bindings.cufile.Status`` directly.
The native members are ``WAITING``, ``PENDING``, ``INVALID``, ``CANCELED``,
``COMPLETE``, ``TIMEOUT``, and ``FAILED``. Returned completion records must be
terminal; the wrapper rejects WAITING/PENDING or unknown completion statuses.

.. autoclass:: CuFileError
   :members:

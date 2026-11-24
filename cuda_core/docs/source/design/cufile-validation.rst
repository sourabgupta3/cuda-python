.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Native cuFile prototype validation
==================================

This report describes the native API in :doc:`cufile`. The implementation is a
local unreleased prototype. Validation was recorded on October 21-23 and
November 2-5, 2026; code was not pushed.

Batch API update, November 5
----------------------------

The current API uses submit(entries), poll(), and wait_all(). Public prepare(),
clear(), no-argument submit(), and the wait() alias were removed. Older validation
entries below describe the preparation-based prototype at their recorded dates.
Each request now retains its own file, buffer, and native argument array until its
completion is collected. No completion history or consumed-result counter is kept.
poll() delivers results once; wait_all() returns only remaining results. If a wait
raises, its completions attribute carries the results collected during that call.

The contract and native I/O suites passed 286 tests with 3 skips on /mnt/nvme0.
The skips cover the existing two-device check and two real rolling-I/O cases blocked
by the native defect below. Injected binding completions exercise out-of-order rolling
replacement, capacity checks, rollback, partial timeout results, error delivery without
replay, and per-request ownership release. They do not qualify another native version.

A raw cuda.bindings.cufile probe initially completed eight rolling host/GPU reads
using one native batch, refilling six requests before the original queue drained.
Delaying the first poll changed the completion order and reproduced an assertion
failure in native nvfs_bio at line 236. The probe kept all argument arrays, files,
and buffers alive until teardown, so the failure also occurs without the core wrapper.
The larger core rolling test independently aborted with memory corruption.

Native source review found that cufile_batch_submit_mixed_ios() passes an internal
queue-slot index into cufile_batch_enque_io(), which indexes the newly submitted
parameter array with that index. A free slot beyond the new array length appears to
explain the invalid access; this is a source-based diagnosis, not a confirmed native
fix. Read-only source: nvme-direct/lib/cufile/cufile_batch.cc, functions at lines 401
and 666 in checkout 1111f7e558e5a4620c73f895024fa6d2b27b0ec1.

On native version 1190, submit(entries) therefore raises NotImplementedError before
calling native submit if any request remains outstanding. Polling and full draining
still work, followed by submission of new or replaced entries on the same Batch.
The rolling example documents this restriction. A native fix and payload validation
across completion orders are required before removing the guard. Rolling operation
on other native versions has not been qualified.

Artifacts: /tmp/probe_cufile_rolling_delayed_20261105.py,
/tmp/cufile-rolling-native-delayed-20261105.log, and
/tmp/cufile-rolling-batch-final-tests-20261105.log. The full regression command was::

   env CUDA_CORE_CUFILE_TEST_DIR=/mnt/nvme0 \
     PYTHONPATH=$PWD/cuda_python_test_helpers \
     /tmp/cufile-core-env/bin/python -m pytest \
     cuda_core/tests/test_cufile_contract.py cuda_core/tests/test_cufile_io.py \
     -q --timeout=35 --tb=short

Environment and build
---------------------

* NVIDIA H200 NVL; one visible GPU; NVIDIA driver 610.43.02.
* CUDA Toolkit 13.4; cuda-bindings 13.4.4; cuda-pathfinder 1.8.3.
* Native cuFile version query 1190 (1.19).
* CPython 3.12; Cython 3.2.9; NumPy 2.5.4; pytest 9.1.1.
* CuPy 14.2.0 is used in array-interoperability tests. The applications use core
  allocations and streams directly.
* Storage: /mnt/nvme0, /dev/md0, ext4. Each run uses a private temporary directory
  and removes only its own files. GPU/storage tests run outside the restricted
  filesystem view.
* Isolated test environment: /tmp/cufile-core-env. The editable build's version
  is derived by setuptools-scm: 1.2.1.dev215+g1f412810b.d20261105.
* The pinned CCCL submodule was initialized locally. No version tags were created.

The native editable build passed::

   env CUDA_HOME=/usr/local/cuda CUDA_PYTHON_PARALLEL_LEVEL=8 \
       /tmp/cufile-core-env/bin/python -m pip install \
       --no-build-isolation --no-deps -e ./cuda_core

Tests
-----

.. list-table:: Recorded results
   :header-rows: 1

   * - Suite
     - Passed
     - Skipped
   * - Native cuFile contract and integration tests
     - 255
     - 1
   * - Existing context, stream, event, memory, and error-handling regressions
     - 387
     - 28

The focused suite was rerun on October 23 after removing registration sharing,
adding optional buffer registration sizes, documenting and validating registration
defaults, adding structured native/filesystem error metadata, using IOParams for both
sync and stream calls, adding File's keyword-only flags argument, and simplifying Driver
to explicit open/close ownership. Sync calls return counts directly and do not
set bytes_done. Coverage includes host/device parameter reuse, reuse between
sync and stream calls without first fetching the stream result, rejection while
the previous transfer is pending, and immediate sync error propagation. It covers
default O_DIRECT, flags=0, O_SYNC/O_DSYNC combinations, O_CLOEXEC, access modes,
exclusive creation without overwriting existing data, and rejected flags without
file creation or truncation. Native sync, stream, and batch I/O were verified
with flags=0 and O_DIRECT | O_SYNC. It also covers immediate initialization,
balanced independent references, idempotent close, open/close failure recovery,
native use counts, duplicate native/wrapper registration errors, and preservation
of the original registration after duplicate requests. Existing core
regression results are from October 21. The focused skip requires a second GPU.
Existing regressions emitted one pytest
plugin assertion-rewrite warning. This checkout's shared test-helper package
was explicitly placed on PYTHONPATH. Existing tests were not edited.

Focused invocation::

   env CUDA_CORE_CUFILE_TEST_DIR=/mnt/nvme0 \
       /tmp/cufile-core-env/bin/python -m pytest \
       cuda_core/tests/test_cufile_contract.py cuda_core/tests/test_cufile_io.py \
       -q --timeout=30

Documentation examples were also executed on /mnt/nvme0: 20 checks passed. These
cover explicit cleanup examples, the File flags examples in the design/user guide
and Google Doc, the user guide's shared IOParams sync/stream example, and the new
OS/native error-handling examples on success and failure paths. Output bytes and
raised exception types were verified. All 26 explicit Python code blocks in the
two local guides and the shared error example compile. A full Sphinx build was
not run; Sphinx/docutils are not installed in this test environment.

Error coverage includes native cuFile/CUDA status preservation and exception chaining,
unknown raw status values, filesystem error metadata, pickling, immediate
sync errno capture, cached stream errors before/after close, and signed batch failures.
Deferred -1 results raise CuFileError with errno=None and no invented cuFile status.
Buffer registration accepts only reserved flags=0; stream masks outside 0x0..0xF
are rejected before resource access.

The comment-review follow-up added eight bytearray/mmap pointer-stability cases:
registration, pending stream I/O, prepared batches, and a closed registration retained
by prepared entries. Resize/close attempts were blocked by active buffer exports;
they succeeded after native use/preparation released the exports. Those eight cases,
the two existing host-export tests, and the native cancellation guard test passed
in one focused run (11 passed). This was not a rerun of the full suite above.

Registration-size tests check omitted/explicit sizes for host and device memory,
the supplied view pointer and byte count passed to native registration, invalid
sizes, duplicate requests with equal/different sizes, and overlapping views.
Sync, stream, and batch I/O enforce the registered size, including deferred
offset changes. Prepared batches retain the registration until their entries
are released. Duplicate stream objects fail with either matching or different
flags. Registrations made through raw bindings remain valid after a failed
wrapper registration. Failed size validation, duplicate requests, runtime checks,
and native registration release extra host-buffer exports even when the caller
retains the exception traceback; the original mmap can then close.

One full-suite run reported 0 instead of 4096 bytes in the existing
test_stream_params_reuse_and_short_result host-stream write after stream sync.
The test passed alone and the full suite passed on both subsequent runs. Two
500-write probes also passed, including one after the full suite in the same
process (215 tests passed, one skipped with that probe included). The origin of
the intermittent result remains unresolved; successful reruns do not establish
that this finding is fixed.

During the November 5 driver-lifetime change, another full-suite run reported
0 instead of 4096 bytes in test_stream_params_respect_fixed_registration_arguments
for the fixed-size flag on a host-stream write. All three fixed-argument cases
passed in isolation, and the next full run passed (279 passed, one skipped).
This observation remains unresolved and is not claimed fixed by the driver change.

Existing regression invocation::

   env CUDA_CORE_CUFILE_TEST_DIR=/mnt/nvme0 \
       PYTHONPATH=$PWD/cuda_python_test_helpers \
       /tmp/cufile-core-env/bin/python -m pytest \
       cuda_core/tests/test_context.py cuda_core/tests/test_stream.py \
       cuda_core/tests/test_event.py cuda_core/tests/test_memory.py \
       cuda_core/tests/test_error_handling.py -q --timeout=30

Coverage includes native sync and stream transfers; C/F host buffers; byte
counts and EOF; zero and unaligned transfers; buffer offsets and remaining-size
defaults; read-only and noncontiguous rejection; scoped registration; stream
alignment promises; core allocation/stream retention after explicit close;
driver leases, standalone parameter queries, and cleanup failures; multiple files; batch preparation,
generations, cookies, snapshots, partial/out-of-order completion, reuse, signed
errors, timed-close retention, and cleanup retry; context-restoration faults;
concurrent reads; dropped requests/files; fork rejection; and process shutdown
with pending stream or batch work.

System-memory coverage includes bytearray, C/F NumPy arrays, mmap, and core
pinned-memory allocations on sync, core/integer stream, and batch paths, with
and without registration. Guard bytes verify file/buffer offsets and bounds.
Mixed host/device batches, retained host exports, and prepared host-buffer
lifetimes are also verified. There is no device-only restriction on these paths.

A native host callback deliberately blocks later stream work while another host
thread reads IOParams.bytes_done after an application-recorded event completes.
Result access returns before the callback is released, proving it does not wait
for later stream work. Additional tests cover early-result rejection, deferred
size/file/buffer offsets, fixed-argument flags, parameter reuse, short results,
and result reads after stream synchronization without entering a cleanup wait.
Another fault test forces event-recording failure after accepted I/O and verifies
that the write is drained before the original exception propagates.

Deterministic failure tests inject at the binding or completion-recording
boundary and complement actual storage I/O. All new tests carry per-test
agent_authored provenance markers. Ruff checks, formatting, and git diff --check
passed for the new code and examples.

Applications and measurements
-----------------------------

The application measurement below was recorded on October 21, 2026.
The file application uses core VMM memory and registration scopes, executes sync
and stream paths, checks actual counts and all bytes, and hashes the outputs.
One warm-up round is excluded per path; five timed rounds each read and write
16 MiB. Allocation, driver setup, registration, verification, and hashing are
outside the timing intervals.

.. list-table:: One warmed application run; aggregate read plus write MiB/s
   :header-rows: 1

   * - Path
     - MiB/s
   * - Native synchronous
     - 12145
   * - CUDA stream synchronization followed by IOParams.bytes_done
     - 8043

Both paths produced::

   341aacac661ccb210720bedaa9ead5d668fe5ea41a73532fc147c71e34040df1

This is a short application measurement, not a statistical latency study,
durability benchmark, or proof of a particular DMA route. The native library's
routing and compatibility configuration remain authoritative. Event allocation
and recording, together with stream synchronization, are included in the async
timing.

The standalone batch application also passed: two input files, two output
files, 8192 bytes read, 8192 bytes written, and byte-for-byte verification.
Recorded commands::

   /tmp/cufile-core-env/bin/python cuda_core/examples/cufile_io.py \
       --directory /mnt/nvme0 --size-mib 16 --iterations 5
   /tmp/cufile-core-env/bin/python cuda_core/examples/cufile_batch.py \
       --directory /mnt/nvme0

Native findings and release work
--------------------------------

Native batch status enum
~~~~~~~~~~~~~~~~~~~~~~~~

On November 4, 2026, the duplicate string completion-status enum was removed.
IOCompletion.status now preserves cuda.bindings.cufile.Status directly, including
the native numeric value. Terminal-state checks use bindings enum members rather
than hard-coded status numbers. The enum import does not load libcufile.

The focused batch checks passed (42 passed, 227 deselected), including five new
cases for COMPLETE, CANCELED, FAILED, INVALID, and TIMEOUT. They verify native enum
identity, error handling, and retained completion values after batch reuse. Real
host/device and multi-file batch paths were also covered. This was not a full-suite
rerun. Examples now compare results against bindings.Status.COMPLETE.

Unified cuFile and filesystem errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On November 5, 2026, native cuFile and filesystem failures were consolidated into
one CuFileError class derived from CUDAError. Filesystem metadata is supplied
through optional errno, filename, and strerror fields; native_code/status_name
retain their separate cuFile status domain. Missing metadata remains None.
The former filesystem-specific subclass was removed from the public API.
CuFileError is not an OSError subclass; original filesystem and binding exceptions
remain available through __cause__.

The Cython sync adapter was rebuilt and verified to capture errno immediately,
preserve the original OSError, and provide its OS error description. Tests also
cover missing paths, pickled error metadata, native/CUDA statuses, and deferred
stream/batch failures with unknown errno. The complete contract/integration suite
passed: 280 passed, one skipped (requires a second GPU). Ruff check/format and
git diff --check passed. This change does not resolve the intermittent async
zero-byte observation recorded above.

Registration capability policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On November 5, 2026, RegisteredBuffer was restricted to device memory whose CUDA
IS_GPU_DIRECT_RDMA_CAPABLE pointer attribute is true. The wrapper raises
NotImplementedError before a cuFile buffer-register call when the
attribute is false, including when compatibility fallback is allowed or forced.
Host-memory registration retains its existing native requirements. CUDA pointer
query failures propagate as CUDAError; they are not treated as unsupported memory.

The full contract and integration suite passed: 273 passed, one skipped (requires
a second GPU). Ruff check and format checks passed for the changed Python files.
Five new test cases cover isolated allowed/forced/disabled compatibility settings,
default core VMM, core stream-ordered pool memory, CuPy asynchronous allocations,
and foreign views of those allocations. The checks verify rejection before native
cuFile mutation, context preservation, optional registration sizes, continued
unregistered sync/stream/batch payload transfers, and registration of eligible
VMM, conventional device memory, and host memory. Separate cases verify that CUDA
pointer-query errors remain CUDAError.

Earlier native allocation results below predate this wrapper restriction. Native
registration succeeding with forced compatibility does not make the allocation
supported by RegisteredBuffer in this version. Native fallback support is tracked
in CUF-145.

External CUDA and host allocations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On November 2, 2026, three isolated subprocess checks validated external allocations
with cuFile 1.19, cuda-bindings 13.4.4, CuPy 14.2.0, and KvikIO 26.10.0. Configurations
were normal native routing, forced native compatibility, and compatibility disabled;
each was set before driver-open. All data files were private temporary files under
/mnt/nvme0 and removed on completion.

Nine inputs were checked: raw bindings cuMemAlloc exposed through a CuPy UnownedMemory
view, CuPy cudaMalloc, CuPy's default pool, cudaMallocAsync, cudaMallocManaged, pageable
NumPy, CuPy pinned host memory, bytearray, and mmap. All passed unregistered sync,
stream, and batch read/write payload checks. Eligible registrations were tested with
the same three execution models: 48 round trips in normal mode, 51 in forced compat,
and 48 with compat disabled (147 total). Async-pool registration returned native 5048
except in forced compat; managed registration returned 5013 in all three modes. Other
listed inputs registered successfully. These are measured results, not portable
allocator guarantees or proof of a particular DMA route.

Both new external-memory documentation examples passed in every configuration.
Passing a bare integer device address was rejected. KvikIO pwrite with its own
compatibility mode OFF and threshold zero verified ordinary CuPy, async CuPy, and
NumPy buffers in each native configuration (nine verified writes); native forced
compatibility remained a separate setting. RMM, Numba, and PyTorch were not installed,
so their protocol/allocator descriptions were checked against primary documentation,
not runtime-tested here.

Probe: /tmp/test_cufile_external_allocations_20261102.py. The three pytest cases passed;
JSON results are /tmp/cufile-external-allocations-20261102-<mode>.json.

Earlier core allocation checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Additional allocation and stream checks on October 22, 2026 verified 27 read/write
round trips on /mnt/nvme0: six core allocation types across sync, stream, and batch
without registration, plus three registrable types across those models. The types
were RDMA-enabled VMM, default VMM, device pool, host pinned pool, legacy pinned host,
and managed memory. Explicit registration accepted RDMA-enabled VMM and both pinned
host types; it rejected default VMM and device-pool memory with native code 5048,
and managed memory with code 5013. These are observed native results, not portable
allocation guarantees or proof of a particular DMA route.

Three isolated process probes on October 23 compared force_compat_mode=False with
allow_compat_mode=True, force_compat_mode=True, and both controls False. Each set the
typed bindings parameters before driver-open and read them back. Private files under
/mnt/nvme0 were used for 4096-byte read/write payloads. All six allocation types passed
unregistered sync, stream, and batch I/O in all three modes. Registration accepted
RDMA-enabled VMM and the two pinned-host types in every mode. Default VMM and the
device pool registered only with forced compatibility. Managed memory registration
returned CUDA_MEMORY_TYPE_INVALID (5013) in every mode. Registered transfers were
also byte-verified: 87 total round trips across the three probes.

CUDA pointer attributes reported IS_GPU_DIRECT_RDMA_CAPABLE=1 for RDMA-enabled VMM,
and 0 for default VMM and device-pool allocations. Managed memory reported
IS_MANAGED=1. These support the allocation distinction, but the generic
NVFS_INTERNAL_DRIVER_ERROR (5048) does not establish the complete cause of a failed
registration. Forced-compatibility registration success does not establish a GPU
peer-memory mapping. Allowing fallback alone is different from forcing compatibility.
The probe and JSON results are in /tmp/test_cufile_allocation_modes_20261023.py and
/tmp/cufile-allocation-mode-{normal,forced_compat,compat_disabled}-20261023.json.

Registration TRACE diagnosis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fourteen isolated registration probes used CUFILE_LOGGING_LEVEL=TRACE and separate
CUFILE_LOGFILE_PATH files. Each compared cuda.core and raw bindings on the same
allocation after releasing any successful registration. Both paths returned matching
results. Default VMM, device-pool, and RDMA-enabled VMM were checked with 4096-byte
and 2 MiB allocations/registrations; managed memory was checked at 4096 bytes. Each
case ran with normal and forced-compatibility settings. All 14 probe checks passed.

For default VMM and device-pool memory, cuFile's initial pointer/range checks passed.
The failure occurred when the nvidia-fs MAP ioctl tried to pin GPU pages. Kernel
messages for the same buffer addresses identify nvfs_pin_gpu_pages calling
nvidia_p2p_get_pages_persistent, which returned -22 (EINVAL). cuFile then reported
NVFS_INTERNAL_DRIVER_ERROR (5048). The 2 MiB cases reproduced the same failure,
excluding the small 4096-byte registration size as the explanation. The failing
allocations reported IS_GPU_DIRECT_RDMA_CAPABLE=0; RDMA-enabled VMM reported 1 and
successfully completed the same mapping path at both sizes. The evidence identifies
peer-page pinning rejection for these allocations on this system; it does not expose
which internal NVIDIA resource-manager check produced EINVAL.

The SYNC_MEMOPS warning with CUDA error 801 also appeared on successful RDMA-enabled
VMM registration, so it does not explain the registration failure. In forced
compatibility mode, the trace instead showed cufile_posix_alloc creating a host
shadow buffer and registration completing without the failing GPU MAP operation.
Managed memory was rejected earlier in cuFile's memory checks: the trace explicitly
reported that cudaMallocManaged memory is unsupported by nvidia_p2p_pages, followed
by CUDA_MEMORY_TYPE_INVALID (5013), in both modes.

Trace logs, JSON results, and kernel.log are retained in
/tmp/cufile-registration-trace-20261023-itxcf95g/. The probe is
/tmp/test_cufile_registration_trace_20261023.py. Logging settings were scoped to
the child processes; /etc/cufile.json and other applications were not changed.

Five stream adapters were checked with registration and verified reads: an owned core
stream, a stream-protocol provider, a borrowed integer, Stream.from_handle, and the
device's default stream. Direct protocol forwarding emitted a core deprecation warning.
The exact GPU VMM/pinned-host allocation example, direct core stream registration
example, and CuPy borrowed-stream interoperability example used in the Google design
document also executed successfully. The two stream registration examples confirm
that deregistration leaves the original stream usable and closing a borrowed wrapper
does not destroy the CuPy-owned CUDA stream.

Eight explicit-cleanup examples passed ten checks against /mnt/nvme0. They cover
Driver lifetime and standalone binding queries; File; IOParams; RegisteredBuffer;
RegisteredStream;
borrowed CuPy streams; batches with BatchIOParams, IOCompletion, IOOpcode, and
batch completion statuses; and CuFileError handling. Buffer registration was checked with host,
RDMA-enabled core VMM, and pinned-host allocations. Transfers were byte-verified;
the original allocations and streams remained usable after deregistration. The
stream result remained readable after IOParams.close(), and batch snapshots
survived clear() and close(). These examples use try/finally rather than with.

These checks establish transfer mechanics and selected lifetime behavior only. They do
not establish isolation from other libraries, configuration restoration, or ownership of
external registrations. Context exit calls close(); entering a closed object again raises
RuntimeError. The code does not check for a nested with on the same object. Review
requested different behavior; see the context manager review item in :doc:`cufile`.

An isolated cuFile 1.19 cancellation probe completed all 32768 bytes of writes
but lost unreaped completion events after Cancel. Active cancellation is gated
off on that version before mutation; normal wait_all and close remain usable.
Injected tests cover canceled outcomes and retained ownership on timeout, but
are not qualification of native cancellation on another version.

Native driver and registration ownership
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Isolated subprocess probes checked native cuFile 1.19 behavior and the
Python prototype separately. Driver counts were read from the native long-returning
cuFileUseCount symbol; the current Python use_count binding discards that result.

* Native implicit initialization by file-handle registration changed the count
  from 0 to 1. File deregistration left 1; one native driver-close returned it to 0.
* After the November 5 simplification, Driver() changed the count from 0 to 1;
  File() and File.close() both left it at 1. Driver.close() returned it to 0;
  runtime shutdown left it at 0. The wrapper's internal driver reference and
  Python driver counter were removed. Only Driver calls native driver-open/close.
* File alone used native implicit initialization: 0 to 1. File.close() and
  Python runtime shutdown left that native implicit reference at 1. A raw native
  driver-close returned it to 0. Patched bindings rejected any explicit Python
  driver-open/close call during this check, including shutdown.
* Two Drivers raised the native count to 2. File creation added nothing;
  closing one Driver left 1 and the file remained usable under the other Driver.
  Closing the file left 1; closing the other Driver returned 0. Repeated close
  left 0. File, buffer/stream registrations, and Batch also completed sync,
  stream, and batch reads without issuing explicit driver-open/close calls.
* Shutdown drained pending work and closed resources before explicit Driver
  objects. Isolated stream and batch shutdown cases passed both with and without
  an explicit Driver. Native batch setup without prior initialization returned
  INVALID_VALUE on cuFile 1.19; Batch now follows native requirements and needs a
  File or Driver to initialize cuFile first.
* Repeating native stream registration returned CU_FILE_INVALID_VALUE (5022),
  with either identical or different flags. Repeating native buffer registration
  returned CU_FILE_MEMORY_ALREADY_REGISTERED (5023) for both host and device memory.
* The prototype rejects a second registration object without issuing another
  native register call. The original object remains usable for verified I/O.
  Closing it makes one native deregister call; a new object can then register
  the resource again. Different sizes or flags do not enable sharing.

Regression cases cover native counts, duplicate errors, wrapper call counts,
retained usability, and verified payloads. Each registration has one Python
owner; retained I/O references protect the native registration until release.
They do not retain a driver reference; close dependent resources and finish I/O
before releasing the final explicit Driver. The final contract/integration run
passed 279 tests with one second-GPU skip. Both complete examples verified their
payloads under /mnt/nvme0: 1 MiB sync/stream transfers and a two-file batch with
8192 bytes read and written. Ruff check/format and git diff --check passed.

An isolated native probe corrected the earlier interpretation of driver setters:
legacy initialization setters are no-ops after driver open. Modern typed parameter
setters reject post-open changes with CU_FILE_DRIVER_ALREADY_OPEN (5026). Driver
therefore only owns open/close; standalone binding functions handle version,
properties, and configuration. The prototype's property snapshot and local
configuration mirror were removed because they could report unapplied values.
The standalone bindings startup example and rejection of post-open typed
configuration changes both passed in a separate process (two checks).

Explicit registration worked for ordinary external device allocations and
core VMM with gpu_direct_rdma=True. Pool allocations and VMM without that flag
failed registration with native code 5048 on this machine. Unregistered pool
I/O worked. Allocator defaults remain unchanged.

For unregistered buffers, offsets are resolved to an effective address before
native submission. Registered device buffers keep the base plus an offset.
This avoids the observed async error when passing a nonzero offset with an
unregistered buffer; the public offset semantics are verified on both paths.

Registered mmap and pinned host buffers exposed a cuFile 1.19 batch case where
dev_ptr_offset was ignored. Host batch entries now resolve the effective CPU
address and pass native offset zero while retaining registration ownership.
Payload and untouched guard bytes are checked for both transfer directions.

Host buffers have no device allocation context. Integer stream adapters therefore
retain the caller's current CUDA context when the buffer provides none. Core
stream adapters retain their stream's context as usual.

Further qualification includes CUDA 12 and floor versions, free-threaded Python,
multiple GPUs, packaging without GDS headers, further GDS systems, stream capture,
subinterpreters, additional native failures, and cancellation. The entire
monorepo and a Sphinx documentation build were not run. Setup/latency budgets,
batch scaling, event pooling, actual route statistics, and durability need
separate measurements before stabilization.

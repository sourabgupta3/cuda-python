cufile_python performance investigation
=======================================

This report concerns the local file I/O backend with raw and stream APIs using
the caller's CUDA context, matching KvikIO. Worker context management remains
in place. Both implementations select cuFile for the device I/O comparison;
there is no Python POSIX/staging fallback in this backend. Selecting cuFile
does not identify the physical GDS route of each request.

What costs time
---------------

The raw path has no worker queue, yet it is slower than KvikIO. ``arr.Array`` reads
the CUDA Array Interface, computes contiguity and bounds, and retains the
buffer owner. KvikIO implements buffer parsing in Cython. The previous Python
wrapper also resolved the allocation's context and pushed/popped it around
native work. That extra behavior is removed from raw and stream APIs, including
stream completion and failure cleanup. cuFile's valid-current-context
requirement does not require the wrapper to perform those operations on every
call. The caller now supplies an appropriate current context, as with KvikIO.

``read`` calls ``pread(...).get()``. Its measured time includes submission to a
Python ThreadPoolExecutor, worker wakeup, the native transfer, and waking the
waiting caller. The wrapper also creates completion state and tracks pending
work so closing the file can drain it. A minimal worker that bypasses buffer
parsing and file/future bookkeeping still pays Python queue and result-delivery
costs. KvikIO performs the corresponding queue/future work in C++.

``raw_read_async(...).check_bytes_done()`` uses a CUDA stream, not the worker
pool. Its wrapper creates stable scalar argument/result storage, retains
ownership, submits, synchronizes the stream, and releases the retained state.
This benchmark completes each request before submitting another; it measures
latency and does not measure useful compute/I/O overlap.

Both implementations release the GIL around native I/O. Python metadata,
callbacks, and task scheduling require Python execution. A wall-time gap is
not evidence that the whole transfer holds the GIL, nor does it identify an
internal libcufile defect.

Measurements and controlled probes
----------------------------------

Fresh-process medians for the updated complete APIs are below. The 4 KiB rows
use six processes per backend, two rounds per process, and 500 measured calls per round.
The 1 MiB and 64 MiB rows use three processes per backend and 80/20 calls per
round respectively. Latency includes completion. Smaller values are better.
These runs preceded the batch extension; their source hashes remain in the
original evidence. Batch measurements use the current implementation separately.

.. list-table:: Complete API latency
   :header-rows: 1
   :widths: 48 26 26

   * - Device-buffer workload
     - KvikIO
     - Python backend
   * - 4 KiB read, one worker
     - 97.1 us
     - 127.9 us (+31.8%)
   * - 4 KiB write, one worker
     - 85.0 us
     - 115.7 us (+36.1%)
   * - 4 KiB raw read
     - 59.5 us
     - 63.9 us (+7.3%)
   * - 4 KiB stream read + completion
     - 228.9 us
     - 240.8 us (+5.2%)
   * - 1 MiB read, one worker
     - 426.8 us
     - 463.1 us (+8.5%)
   * - 64 MiB read, one worker
     - 7.012 ms
     - 7.086 ms (+1.1%)
   * - 64 MiB write, one worker
     - 4.432 ms
     - 4.605 ms (+3.9%)

The comparison also reruns the previous Python implementation in six fresh
processes at 4 KiB. Order rotates between previous, updated, and reference
implementations. Raw read falls from 68.9 to 63.9 us (5.0 us, 7.2%); stream read
plus completion falls from 255.5 to 240.8 us (14.7 us, 5.8%). Worker read/write
medians change from 126.7/115.8 to 127.9/115.7 us; this change does not improve
worker dispatch.

Raw-read process medians span 68.8-69.6 us before and 59.6-64.7 us after the
change, versus 59.1-60.1 us for KvikIO. Stream ranges overlap: 240.1-257.1 us
before and 225.1-248.3 us after, versus 221.2-230.0 us for KvikIO. The stream
median reduction is an observed end-to-end result, not an isolated 14.7 us
cost of context calls. The 64 MiB Python read range is 7.077-7.199 ms; its
1.1% difference from KvikIO is smaller than that variation. These are observed
process ranges, not confidence intervals.

Earlier diagnostic probes
~~~~~~~~~~~~~~~~~~~~~~~~~

The following probes used the previous implementation, before removal of raw
and stream context management. Its source is recorded by hash. The paired run
has eight bracketed rounds of 500 calls. Its public raw/read/stream medians are
68.7/125.4/242.3 us, versus KvikIO's 59.2/96.7/219.0 us in that process. The
following are measured savings against the adjacent Python baselines, in us.
The ranges are the minimum and maximum savings across the eight rounds.

.. list-table:: Controlled 4 KiB interventions
   :header-rows: 1
   :widths: 64 16 20

   * - Change made only inside the diagnostic
     - Median saving
     - Range
   * - Raw read: reuse parsed buffer view
     - 3.8
     - 0.8 to 4.1
   * - Raw read: cache context resolution, keep push/pop
     - 3.5
     - 2.9 to 4.5
   * - Raw read: use known current context, omit context handling
     - 5.0
     - 4.7 to 5.4
   * - Raw read: prepared view plus known current context
     - 9.0
     - 5.9 to 9.4
   * - Worker read: prepared view plus cached context resolution
     - 8.8
     - 7.6 to 9.6
   * - Worker read: minimal executor job calling bindings
     - 18.7
     - 17.1 to 19.0
   * - Worker read: single reusable queue/event slot
     - 23.4
     - 21.1 to 24.2
   * - Stream read: prepared view plus known current context
     - 9.0
     - 8.1 to 22.3
   * - Raw read: bypass empty registration lookup
     - 0.4
     - 0.2 to 0.5
   * - Stream read: omit global retain/release, keep file retention
     - 0.3
     - -0.2 to 0.8

The combined raw probe reduces latency to 59.7 us, close to KvikIO's 59.2 us.
Direct bindings, including default-stream synchronization, take 55.3 us. This
accounts for most of the raw gap and argues against a large unavoidable
bindings penalty in this fixture.

The minimal worker still takes 106.6 us, compared with 102.4 us for the reusable
single-slot worker and 96.7 us for KvikIO. Bypassing the wrapper's parsing,
context, and ownership work saves about 19 us; approximately 4 us more
separates these two Python dispatch mechanisms. The remaining difference is not
an isolated GIL measurement. These stripped-down probes do not implement the
full public API, and their differences must not be added to the earlier rows.

In that earlier fixture, stream storage alone did not explain the gap. Direct bindings with four
ctypes scalars, a fresh packed structure, and reused packed storage take
216.7, 217.0, and 215.3 us respectively. Context/view reuse saves about 9 us;
global retention has no substantial demonstrated effect. The rest of the
stream gap is not fully apportioned between remaining Python work and native
submission/completion timing. Its absolute size also changes between the
paired and fresh-process fixtures. No single component is claimed to explain
all 32 us of the earlier fresh-process gap. These diagnostic numbers are not
measurements of the updated backend.

The measurement table and probe results are drawn from
``benchmarks/performance.json.gz``. This compressed evidence contains the exact
commands, source hashes, raw samples, per-round medians, process audits, and
loaded library paths. ``context_policy_change`` contains the 30 new process
runs, the comparison driver, and the previous runtime sources. Older fields
retain pre-change measurements and probes. The reproduction drivers are
included so fresh-process comparisons can be reconstructed without retaining
more one-off scripts in the source tree.

Method and limits
-----------------

The fixture uses an H200 NVL, CPython 3.12.3, CUDA bindings 13.4.4, libcufile 1.19,
and KvikIO 26.10.0. Private files live on ``/mnt/nvme0``, an ext4 filesystem on a
four-NVMe RAID0. Buffers are contiguous CuPy device allocations, with no explicit
buffer registration; the CUDA stream is also unregistered. Both packages use a
4 MiB task size. KvikIO compatibility mode is OFF and its GDS threshold is zero
to force the cuFile worker path even at 4 KiB. Those two settings are no-ops in
this backend. These are not KvikIO's default small-I/O routing settings.

Complete-backend comparisons use separate processes. The 4 KiB comparison
rotates previous/updated/reference order; larger comparisons alternate updated
and reference order. Probes share one file, allocation, and stream, and rotate case order.
Every probe is bracketed by public-API baseline measurements. The reported
saving is the mean of those two baseline medians minus the probe median,
aggregated across rounds. Interventions overlap: their savings cannot be added
to produce an exact breakdown.

Twenty warmup operations precede each timed block. Payloads and transferred
byte counts are checked. Open/close, registration, allocation, payload checks,
and write flushing are outside timing. Writes are not durability measurements.
Reads are warm repeated reads; no cold-cache or sustained-storage-throughput
claim follows from them. Monitoring and profiling are off for comparisons.
The harness can record separate instrumented spans with ``--trace``. Such spans
include bindings/GIL reacquisition and are not native device execution times.

All 30 new process runs pass the before/between-round audits. The older 24-run
comparison also passed those audits. Earlier attempts that detected another
native cuFile workload remain excluded. The old paired diagnostic has clean
start/end audits; its harness predates the per-round check. No audit is a
hardware reservation, and work between observations cannot be excluded.
Four-worker qualification and clean additional traces from the earlier
investigation remain incomplete because other GPU workloads kept restarting.
This update measures one worker; partial four-worker samples do not support a
throughput claim. No confidence interval or cross-system guarantee is claimed.

Interpretation for implementation work
--------------------------------------

Raw and stream context lookup/push/pop is now removed from the implementation.
The caller-context requirement covers submission, stream completion, and file
close with pending stream I/O. Worker threads still establish and restore
their context, as KvikIO does. Context rules are the same for Python and C++.

Reusing a parsed buffer and resolved worker context remains worth investigating
for repeated I/O. It must preserve the allocation owner's lifetime and reject
stale metadata; a global cache keyed only by a pointer is not sufficient.

Worker dispatch is the next substantial cost for small blocking/worker I/O.
Replacing a general future with reusable state needs the same error delivery,
ownership, close/drain behavior, and concurrent-submission semantics. The
single-slot probe intentionally supports one outstanding operation and does
not prove that a replacement executor would retain the measured saving.

Registration lookup and global stream retention are checked independently.
Their measured contributions must justify any change; removing ownership
tracking to improve a benchmark would invalidate the resource contract.
Likewise, fewer ctypes objects are useful only if the packed/reused-storage
probe demonstrates a meaningful benefit.

Native batch measurements
-------------------------

``benchmarks/batch.py`` measures submit plus blocking completion for repeated
4 KiB GPU reads. Queues and entries are reused; the public API validates and
packs every submission. One process runs six shuffled rounds of 100 calls per
case, with warmups and payload verification. These are warm-cache latency
measurements, excluding queue creation, file open, and durable storage flush.
Source hashes, raw samples, environment, and idle-workload audits are in
``docs/batch-performance.json``.

.. list-table:: Batch latency, microseconds per complete batch
   :header-rows: 1

   * - Entries
     - Public Batch API
     - Bindings, repacked
     - Bindings, prepacked
   * - 1
     - 106.9
     - 87.4
     - 79.6
   * - 8
     - 246.5
     - 194.6
     - 163.7
   * - 32
     - 809.3
     - 670.6
     - 564.4

There is no KvikIO Python batch baseline. Both direct-binding probes omit
public buffer validation, lifetime tracking, and completion objects; they are
diagnostic lower bounds, not equivalent APIs. Against repacking through bindings,
the wrapper adds 19.4, 51.9, and 138.7 us per batch in this process. The difference
includes Python buffer parsing, ownership, overlap validation, locking, and result
creation; the experiment does not isolate each cost. Cross-process/throughput
qualification remains outstanding.

The first implementation repeatedly accessed nested bindings record properties
while packing and reading results. Switching to structured NumPy field access
using the bindings' public ABI-derived dtypes and ``from_data`` removed that
overhead without a native extension or a new prepare API. At 32 entries, complete
public batch latency fell from 2.662 ms in the initial run to 0.809 ms. The initial
run and its source hashes are preserved in ``docs/validation/batch-before-packing.json``;
these two runs were sequential, not a paired statistical study.

Native host polling qualification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An isolated direct-bindings reproducer on libcufile 1.19 returned FAILED/-1
for a host write and then zero bytes for the read of that unwritten file.
It imports no ``cufile_python`` code. The source and output are in
``docs/validation/native-host-poll-repro.py`` and its ``.log``.

The native debug trace showed status collection observing ``hostMemHandle->active``
as false before the host worker set it true. The checked native source releases
inactive buffer state in ``cuFileBatchIOGetStatusInternal``, even when no completion
was returned. This explains the null internal shadow buffer seen by the worker.
The wrapper therefore rejects nonblocking collection of pending host entries on
1.19. Blocking ``wait_all()`` collects every entry before native cleanup; GPU-only
polling remains enabled. Tests verify rejection occurs before any native status
call. After this guard, ten consecutive runs passed all 87 prototype tests.
No retry, arbitrary delay, or Python fallback hides the native failure.

Reproduce and inspect
---------------------

Use the commands in ``README.rst`` with other GPU/storage workloads idle.
``benchmarks/investigate.py`` supports complete-backend comparisons and the
diagnostic ``--probes --trace`` run. Read compressed evidence with::

    import gzip
    import json

    with gzip.open("prototypes/kvikio_python/benchmarks/performance.json.gz", "rt") as source:
        report = json.load(source)

The importable package contains no benchmark helpers. Retired experiments,
earlier fallback measurements, and their source snapshots are consolidated in
``archive/performance-history.tar.gz``; they are not current performance claims.

Source anchors
--------------

* ``cufile_python/_lib/arr.py``: Array and parse.
* ``cufile_python/_lib/_runtime.py``: Context and pool.
* ``cufile_python/_lib/file_handle.py``: _parallel, _raw, and _async.
* ``cufile_python/_lib/future.py``: worker and stream completion ownership.
* `cuFile context requirements <https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#usage>`_.
* `Pinned KvikIO file implementation <https://github.com/rapidsai/kvikio/blob/5530d767aeb55f63db59a7bb64f2745e8061eee5/cpp/src/file_handle.cpp>`_: read_impl, pread, and read_async.
* `Pinned KvikIO stream result storage <https://github.com/rapidsai/kvikio/blob/5530d767aeb55f63db59a7bb64f2745e8061eee5/cpp/src/stream.cpp>`_.
* `Pinned Cython adapter <https://github.com/rapidsai/kvikio/blob/5530d767aeb55f63db59a7bb64f2745e8061eee5/python/kvikio/kvikio/_lib/file_handle.pyx>`_.

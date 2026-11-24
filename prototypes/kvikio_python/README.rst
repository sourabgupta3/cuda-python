cufile_python: local file I/O over CUDA bindings
================================================

This separate prototype provides KvikIO-shaped CuFile, driver, buffer, stream,
settings, and future APIs, plus native cuFile batch I/O, over ``cuda.bindings``.
It does not load libkvikio or
build a native extension. ``cuda.core.cufile`` remains a separate implementation.
The compatibility target is KvikIO v26.10.00, commit
``5530d767aeb55f63db59a7bb64f2745e8061eee5``; this is not an installable release.

Start here
----------

* ``cufile_python/cufile.py``: public file and future APIs.
* ``cufile_python/_lib/file_handle.py``: I/O submission and worker dispatch.
* ``cufile_python/_lib/future.py``: completion and retained resources.
* ``cufile_python/_lib/buffer.py`` and ``_lib/stream.py``: registrations.
* ``cufile_python/_lib/cufile_driver.py``: driver lifecycle and properties.
* ``cufile_python/batch.py`` and ``_lib/batch.py``: native batch API and backend.
* ``examples/porting_demo.py``: executable usage.
* ``examples/batch_io.py``: batch reads/writes across files and host/GPU buffers.

``docs/design.rst`` describes the API. ``docs/performance.rst`` explains the
current measurements and controlled probes. ``docs/upstream_applications.rst``
records migration checks. ``docs/framework_applications.rst`` records cuDF and
cuCIM application checks, the compiled-I/O boundary, and a native cuCIM/library
initialization issue. The Google design is at
https://docs.google.com/document/d/183_EGMZONpgtYP5bOBMOrqxjehK-QMPqMTSj8iU-IIA/edit.

``tests/`` contains behavioral tests; ``benchmarks/`` contains the current
validation and performance tools. ``docs/validation/`` contains current logs.
Superseded tools, measurements, and source snapshots are consolidated in
``archive/performance-history.tar.gz``. Neither the archive nor benchmark helpers
are imported by the runtime package. There are no bundled Zarr, remote-storage,
or benchmark integrations in ``cufile_python``.

Imports and behavior
--------------------

For code using top-level attributes, change the import::

    import cufile_python as kvikio

For unchanged applications with submodule imports, enable aliases at startup,
before importing KvikIO or starting threads::

    from cufile_python.compat import enable_kvikio_compat
    enable_kvikio_compat()

    import kvikio
    from kvikio.cufile import CuFile

The helper only redirects supported I/O imports. It refuses to replace an
already imported KvikIO. Ordinary ``import cufile_python`` installs no aliases.

All host/device I/O requires libcufile. ``compat_mode`` (ON/OFF/AUTO),
``gds_threshold``, ``bounce_buffer_size``, and ``auto_direct_io_read/write`` retain
get/set/context-manager behavior, but never enable Python fallback I/O. Their
environment settings are also inactive for routing. ``is_compat_mode_preferred()``
returns False. Native cuFile chooses GDS or its own compatibility path; failures
propagate. Buffer registration always calls cuFile; ``bounce_buffer_free()``
returns zero. Worker count, task size, and per-device worker pools remain active.

``read/write`` block; ``pread/pwrite`` return a single-consumption ``IOFuture``.
Raw stream methods return ``IOFutureStream`` with ``check_bytes_done()``.
Applications own raw CUDA streams and must finish I/O before deregistration.
Raw and stream calls use the caller's current CUDA context without looking up
or switching contexts. Keep an appropriate context current for submission,
stream completion, and closing a file with pending stream I/O. Worker-based
``pread/pwrite`` (and therefore ``read/write``) establish and restore the context
on their worker threads.
``Batch(capacity)`` accepts ``BatchOp`` entries carrying their own ``CuFile``.
``submit`` starts work; ``poll`` and ``wait_all`` return each ``BatchCompletion``
once, and its ``get()`` returns bytes or raises ``RuntimeError``. Native status
values are preserved. A queue can be reused after all results are collected.
It supports ``closed``, explicit ``close()``, and ``with``; closing drains work.
On libcufile 1.19, host batches require blocking ``wait_all()`` without a timeout,
and active cancellation is disabled because of native completion/lifetime bugs.
GPU-only batches support polling and timed waits. See the design for details.
Importing the package does not open the driver. See the design for defaults,
errors, ownership, context handling, and qualification limits.

Run and validate
----------------

Use the validated environment from the repository root::

    PYTHONPATH="$PWD/prototypes/kvikio_python" /tmp/cufile-core-env/bin/python \
      prototypes/kvikio_python/examples/porting_demo.py --directory /mnt/nvme0

    PYTHONPATH="$PWD/prototypes/kvikio_python" /tmp/cufile-core-env/bin/python \
      prototypes/kvikio_python/examples/batch_io.py --directory /mnt/nvme0

``requirements.txt`` lists runtime and development dependencies. The runtime
requires CUDA bindings, NumPy (used by the cuFile bindings' native structures),
and native CUDA/cuFile. CuPy supplies device test arrays, pytest runs tests, and
Dask is used by upstream benchmark fixtures. Reference comparisons also require
``kvikio-cu13==26.10.0``.

Run the prototype tests and public signature inventory::

    PYTHONPATH="$PWD/prototypes/kvikio_python" /tmp/cufile-core-env/bin/python -m pytest \
      prototypes/kvikio_python/tests -q --basetemp=/mnt/nvme0/cufile-prototype-tests
    /tmp/cufile-core-env/bin/python prototypes/kvikio_python/benchmarks/api_inventory.py \
      /tmp/kvikio-reference-26.10/python/kvikio/kvikio

The upstream checkout is available at ``/tmp/kvikio-reference-26.10``. The test
runner accepts pytest arguments and redirects imports to this backend::

    PYTHONPATH="$PWD/prototypes/kvikio_python" KVIKIO_COMPAT_MODE=OFF \
      /tmp/cufile-core-env/bin/python prototypes/kvikio_python/benchmarks/run_upstream_tests.py \
      /tmp/kvikio-reference-26.10/python/kvikio/tests/test_basic_io.py \
      /tmp/kvikio-reference-26.10/python/kvikio/tests/test_async_io.py \
      /tmp/kvikio-reference-26.10/python/kvikio/tests/test_cufile_driver.py \
      /tmp/kvikio-reference-26.10/python/kvikio/tests/test_defaults.py \
      /tmp/kvikio-reference-26.10/python/kvikio/tests/test_statistics.py \
      -q -k 'not http and not remote and not test_bounce_buffer_free' \
      --basetemp=/mnt/nvme0/cufile-upstream-tests

The bounce-buffer test expects the intentionally removed Python staging cache.
The prototype tests verify no-op settings and required-cuFile errors for every
legacy compatibility value. Remote configuration tests are outside scope.

Run unchanged upstream applications, including their pinned helper source::

    /tmp/cufile-core-env/bin/python prototypes/kvikio_python/benchmarks/check_upstream_apps.py \
      --directory /mnt/nvme0 --output /tmp/cufile-upstream-applications.json

Performance reproduction
------------------------

Run while other GPU/storage workloads and tests are idle. A native-I/O baseline
can select ``--implementation python``, ``reference``, or ``paired``. Diagnostic
substitutions require the paired process and are never installed in production::

    PYTHONPATH="$PWD/prototypes/kvikio_python" /tmp/cufile-core-env/bin/python \
      prototypes/kvikio_python/benchmarks/investigate.py --probes --trace \
      --rounds 8 --count 500 --output /tmp/cufile-performance.json

Use ``--size 67108864 --threads 4 --count 30`` without ``--probes`` for large I/O.
Each run verifies data and records source hashes, per-round samples, settings,
and process audits. Files are private temporary files under ``/mnt/nvme0``;
flushing is outside timing. The report states which comparisons share a native
path and which diagnostic substitutions relax the public ownership/context
contract. ``docs/performance.rst`` records results and their limits.

Measure batch submit plus completion against direct-binding diagnostic baselines::

    PYTHONPATH="$PWD/prototypes/kvikio_python" /tmp/cufile-core-env/bin/python \
      prototypes/kvikio_python/benchmarks/batch.py --directory /mnt/nvme0 \
      --output /tmp/cufile-batch-performance.json

Original licenses and modifications are recorded in ``LICENSE`` and ``NOTICE``.
All implementation changes remain local.

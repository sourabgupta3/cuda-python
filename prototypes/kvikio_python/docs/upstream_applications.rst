Unmodified upstream application checks
======================================

Two existing local file I/O programs from the KvikIO GitHub repository were downloaded at
commit ``5530d767aeb55f63db59a7bb64f2745e8061eee5`` and run without source edits.
The runner compares the downloaded bytes with the inspected checkout before
execution and records each source URL and SHA-256 hash.

.. list-table:: Executed comparison
   :header-rows: 1
   :widths: 24 46 30

   * - Program
     - Workload and settings
     - Result
   * - hello_world.py
     - GPU write/read, two pread futures and get(), context managers,
       statistics; AUTO, ON, and OFF compatibility settings
     - 6 runs passed
   * - single_node_io.py
     - 16 MiB, four workers, registered buffers, single/multiple files
       and arrays, pwrite/pread futures; ON and OFF
     - 4 runs passed

Each setting runs in separate reference-wheel and Python-backend processes.
All 10 runs passed on files under ``/mnt/nvme0``. The original assertions ran;
the runner additionally checked file contents against the expected arrays and
compared output hashes between backends. GDS threshold was zero, task size was
1 MiB, and the worker count was four. This backend now requires cuFile for every
mode; ON/OFF/AUTO are retained no-op settings for I/O routing. Reference KvikIO
still applies its normal POSIX/cuFile selection. This deliberately checks source
migration across different routing policies, not identical backend selection or
performance. Calling cuFile is not per-request proof of physical GDS.

The reference processes loaded ``libkvikio.so``; none of the Python-backend
processes did. The latter loaded the prototype package and CUDA cuFile bindings.
Application code keeps ``import kvikio``. For the Python backend, the runner
calls ``cufile_python.compat.enable_kvikio_compat()``
before executing the program. This aliases the supported imports to the
canonical ``cufile_python`` modules. The runner separately loads the pinned
upstream benchmark helper as a test fixture; no benchmark package is bundled in
``cufile_python``. Reference runs do not install aliases.
This tests explicit import migration, not a released cuda.core API.

These are upstream examples and a benchmark application, not an independent
production application's training or inference workflow. A larger downstream
application has not yet been qualified. The separately supplied porting_demo.py
was written for this project and is not an externally sourced application.

Run ``benchmarks/check_upstream_apps.py`` as documented in README.rst.
``docs/validation/applications.json`` records the latest runs, source hashes,
and each application's output. ``applications.log`` records the run summary.
An upstream benchmark banner can still describe its ON setting as a POSIX
fallback; that unchanged banner describes KvikIO's policy. The Python backend's
actual routing policy is recorded explicitly in each JSON result.

Sources
-------

* `Local I/O example <https://github.com/rapidsai/kvikio/blob/5530d767aeb55f63db59a7bb64f2745e8061eee5/python/kvikio/examples/hello_world.py>`_
* `Multi-file benchmark <https://github.com/rapidsai/kvikio/blob/5530d767aeb55f63db59a7bb64f2745e8061eee5/python/kvikio/kvikio/benchmarks/single_node_io.py>`_

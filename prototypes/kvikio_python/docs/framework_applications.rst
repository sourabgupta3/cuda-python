cuDF and cuCIM application checks
=================================

Python I/O migration works in the tested cuDF and cuCIM processing applications.
Replacing Python imports does not replace either framework's compiled file
readers. A separate native-library coexistence failure prevents an unconditional
compatibility claim for cuCIM's TIFF reader.

Results
-------

The main matrix completed 22 process-isolated runs: 20 passed and two cuCIM
native-reader runs crashed. Two additional TIFF runs passed with the explicit
initialization-order workaround described below.

.. list-table:: Main matrix
   :header-rows: 1

   * - Workflow
     - Result
     - Uses the Python replacement?
   * - cuDF: read GPU columns, group by key, write GPU aggregates
     - 9 passed
     - Yes in the replacement runs; binding calls were counted
   * - cuCIM: read image, Gaussian filter, label components, write GPU labels
     - 9 passed
     - Yes in the replacement runs; binding calls were counted
   * - cuDF: native Parquet roundtrip, column selection, predicate filter
     - 2 passed
     - No; zero Python file or binding I/O calls
   * - cuCIM: native tiled TIFF crops to CPU and GPU memory
     - 2 crashed before workaround
     - No; native device-name parsing crashed

Each explicit-I/O application runs with ``read/write``, ``pread/pwrite``,
``raw_read/raw_write``, and stream I/O against both original KvikIO and
``cufile_python``. A fifth run exercises the replacement's batch API, which
has no equivalent in the pinned KvikIO release. Application code is identical
between each paired run; the replacement uses ``enable_kvikio_compat()`` before
framework imports.

The cuDF application reads two 65,536-element columns into cuDF-owned,
RMM-backed allocations and checks its GPU group-by result against pandas.
The tested RMM resource is ``CudaMemoryResource``. The cuCIM application reads
a 512-by-512 float image, applies Gaussian smoothing and connected-component
labeling, and checks against scikit-image. Both verify input contents, native
completion byte counts, output file contents, and matching hashes across the
two backends. The batch runs use GPU buffers and blocking completion collection.

These are purpose-built integration applications using the actual installed
frameworks. They are not downloaded upstream applications or complete cuDF/cuCIM
test suites. They do not establish performance, multi-GPU, allocator-wide, or
physical GDS-route guarantees.

Compiled I/O boundary
---------------------

cuDF's ``read_parquet(path)`` and ``to_parquet(path)`` reach libcudf's C++
``kvikio::FileHandle``. Its native loader retains a dependency on libkvikio.
Python import aliases do not intercept these calls or remove that dependency.
The Parquet control verifies coexistence and data correctness, not migration
of this native backend.

cuCIM's ``CuImage.read_region()`` uses native TIFF plugins. It does not call
Python ``kvikio.CuFile``. Its processing functions can consume GPU arrays loaded
through the replacement, as demonstrated by the explicit-I/O application.

Pinned source references:

* `cuDF native file source <https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/io/utilities/datasource.cpp#L140>`_
* `cuDF native file sink <https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/cpp/src/io/utilities/data_sink.cpp>`_
* `cuDF native dependency loading <https://github.com/rapidsai/cudf/blob/2cd6c6c087c6f4ec566e20269bc8e7d87c522ed9/python/libcudf/libcudf/load.py>`_
* `cuCIM native file driver <https://github.com/rapidsai/cucim/blob/0899575ad87cc00ad0e73343c14709dd021f8563/cpp/src/filesystem/cufile_driver.cpp>`_

cuCIM native coexistence failure
--------------------------------

The main test loads libcufile by querying its version before the framework
workload. This is significant: cuCIM's native device-name parser crashes after
that load. The same crash reproduces after importing original KvikIO.
It does not require file I/O, batch submission, or the Python replacement.

Minimal reproduction::

    from cuda.bindings import cufile
    print(cufile.get_version())  # 1190; loads libcufile.
    from cucim.clara.io import Device
    print(Device("cpu"))        # SIGSEGV in the tested environment.

Isolated subprocess results:

.. list-table:: Library initialization checks
   :header-rows: 1

   * - Order
     - Result
   * - cuCIM Device alone
     - Pass
   * - Import cuda.bindings.cufile, without calling it; construct Device
     - Pass
   * - Enable cufile_python import aliases; construct Device
     - Pass
   * - Call cufile.get_version(); construct Device
     - SIGSEGV
   * - Import original KvikIO; construct Device
     - SIGSEGV
   * - Construct Device before loading libcufile or original KvikIO
     - Pass in both checks

GDB locates the failure in ``free()``, called from C++ regex compilation in
``libcucim.so.26``, during ``cucim::io::Device::Device(std::string const&)``.
The root cause of the native-library interaction is not established. In
particular, these observations do not establish a cuFile driver-refcount bug.

Initializing ``Device("cpu")`` before the other native library loads also lets
the full TIFF controls pass with both backends. This is an observed workaround,
not a supported lifetime contract or a native fix. The default main matrix
retains the failures instead of silently applying the workaround.

The TIFF checks select ``ENABLE_CUSLIDE2=1`` and preload the installed
``nvidia.nvimgcodec`` Python module. The legacy plugin's GPU batch decoder only
accepts JPEG; the fixtures use uncompressed and deflate tiles. That separate
decoder restriction was encountered during setup. With cuslide2 and the
initialization workaround, all four combinations of codec and output device
match the expected NumPy crop. Optional nvTIFF/nvJPEG2000 extensions are absent;
these checks do not claim accelerated TIFF decoding or vendor metadata coverage.

* `cuCIM device-name parser <https://github.com/rapidsai/cucim/blob/0899575ad87cc00ad0e73343c14709dd021f8563/cpp/src/io/device.cpp#L29>`_
* `cuCIM plugin selection <https://github.com/rapidsai/cucim/blob/0899575ad87cc00ad0e73343c14709dd021f8563/cpp/src/config/config.cpp#L172>`_
* `Legacy decoder restriction <https://github.com/rapidsai/cucim/blob/0899575ad87cc00ad0e73343c14709dd021f8563/cpp/plugins/cucim.kit.cuslide/src/cuslide/tiff/ifd.cpp#L703>`_

Reproduction and evidence
-------------------------

Tests ran on 2026-11-06 using Linux x86_64, CPython 3.12.3, H200 NVL, and scratch
files under ``/mnt/nvme0/gpuio-agent-scratch``. Versions are cuDF 26.10.2,
cuCIM 26.10.0, KvikIO 26.10.0, cuda-bindings 13.4.4, CuPy 14.2.0,
NumPy 2.4.6, and libcufile reporting 1190. The separate environment
``/tmp/cufile-framework-env`` has the RAPIDS dependencies and references the
existing prototype environment; it does not replace packages there.

Run the full matrix from the repository root::

    /tmp/cufile-framework-env/bin/python \
      prototypes/kvikio_python/benchmarks/check_framework_apps.py \
      --output prototypes/kvikio_python/docs/validation/framework-applications.json

The command returns a failure status while the native coexistence crashes
reproduce. Run the standalone native crash reproducer in its own process::

    /tmp/cufile-framework-env/bin/python \
      prototypes/kvikio_python/docs/validation/cucim-library-order-repro.py

Add ``--device-first`` for the initialization control, or ``--kvikio`` to load
original KvikIO instead of querying cuFile directly.

``validation/framework-applications.json`` records all main runs, library paths,
call counts, checks, and hashes. ``validation/framework-environment.json`` records
resolved packages and pinned source hashes. ``validation/framework-runs/``
contains individual logs, GDB output, the seven isolation cases, and the two
successful initialization-workaround runs. Initial failures are retained in
``validation/framework-applications-initial.json``. No runtime implementation
changes or code pushes were made for these checks.

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare RAPIDS workflows with Python KvikIO and the cuFile Python alias.

Native framework readers are coexistence controls: zero Python I/O calls is
expected. The explicit-I/O applications must actually cross the selected Python
backend. These are purpose-built integration applications, not upstream tests.
"""

import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import threading
import traceback
from collections import Counter
from pathlib import Path


def instrument(target, names, counts, lock):
    for name in names:
        original = getattr(target, name)

        @functools.wraps(original)
        def wrapper(*args, _name=name, _original=original, **kwargs):
            with lock:
                counts[_name] += 1
            return _original(*args, **kwargs)

        setattr(target, name, wrapper)


def transfer(api, kvikio, cp, pairs, writing=False):
    from contextlib import ExitStack

    cp.cuda.get_current_stream().synchronize()
    with ExitStack() as stack:
        handles = [stack.enter_context(kvikio.CuFile(path, "w" if writing else "r")) for path, _ in pairs]
        if api == "batch":
            with kvikio.Batch(len(pairs)) as batch:
                batch.submit(
                    [
                        kvikio.BatchOp(file, buf, "write" if writing else "read")
                        for file, (_, buf) in zip(handles, pairs)
                    ]
                )
                assert [item.get() for item in batch.wait_all()] == [buf.nbytes for _, buf in pairs]
            return
        stream = cp.cuda.Stream() if api == "stream" else None
        pending = []
        operation = "write" if writing else "read"
        for file, (_, buf) in zip(handles, pairs):
            if api == "stream":
                pending.append((getattr(file, "raw_" + operation + "_async")(buf, stream.ptr), buf.nbytes))
            elif api == "pread":
                pending.append((getattr(file, "p" + operation)(buf), buf.nbytes))
            else:
                method = ("raw_" if api == "raw" else "") + operation
                assert getattr(file, method)(buf) == buf.nbytes
        for future, size in pending:
            assert (future.check_bytes_done() if stream else future.get()) == size


def worker(args):
    report = {
        "framework": args.framework,
        "workflow": args.workflow,
        "api": args.api,
        "backend": args.backend,
        "status": "running",
    }
    api_counts, native_counts, lock = Counter(), Counter(), threading.Lock()
    try:
        if args.backend == "python":
            from cufile_python.compat import enable_kvikio_compat

            enable_kvikio_compat()
        import cupy as cp
        import kvikio
        import numpy as np

        from cuda.bindings import cufile

        instrument(
            kvikio.CuFile,
            [
                "__init__",
                "read",
                "write",
                "pread",
                "pwrite",
                "raw_read",
                "raw_write",
                "raw_read_async",
                "raw_write_async",
            ],
            api_counts,
            lock,
        )
        instrument(
            cufile,
            ["read", "write", "read_async", "write_async", "batch_io_submit", "batch_io_get_status"],
            native_counts,
            lock,
        )
        report.update(
            package=kvikio.__file__,
            native_version=cufile.get_version(),
            versions={
                name: importlib.metadata.version(name)
                for name in ["cudf-cu13", "cucim-cu13", "kvikio-cu13", "cuda-bindings", "cupy-cuda13x", "numpy"]
            },
        )
        root = Path(args.directory)
        if args.framework == "cudf":
            import cudf
            import pandas as pd
            import rmm

            report["memory_resource"] = type(rmm.mr.get_current_device_resource()).__name__
            if args.workflow == "native":
                expected = pd.DataFrame(
                    {
                        "key": np.arange(4096, dtype=np.int32) % 32,
                        "value": np.arange(4096, dtype=np.int64),
                        "text": ["row" + str(i % 19) for i in range(4096)],
                    }
                )
                frame = cudf.from_pandas(expected)
                path = root / "table.parquet"
                frame.to_parquet(path, index=False, compression="snappy")
                pd.testing.assert_frame_equal(cudf.read_parquet(path).to_pandas(), expected)
                pd.testing.assert_frame_equal(
                    cudf.read_parquet(path, columns=["value"]).to_pandas(), expected[["value"]]
                )
                filtered = cudf.read_parquet(path, filters=[("key", "==", 3)]).to_pandas().reset_index(drop=True)
                pd.testing.assert_frame_equal(filtered, expected.loc[expected.key == 3].reset_index(drop=True))
                payload = expected.to_json(orient="split").encode()
                report["checks"] = ["parquet roundtrip", "column selection", "predicate filter"]
            else:
                n = 65536
                host = [np.arange(n, dtype=np.int32) % 32, np.arange(n, dtype=np.int64) % 1000]
                owners = [cudf.Series(np.zeros(n, dtype=array.dtype)) for array in host]
                arrays = [owner.values for owner in owners]
                pairs = [(root / f"column-{i}.bin", array) for i, array in enumerate(arrays)]
                for (path, _), expected in zip(pairs, host):
                    path.write_bytes(expected.tobytes())
                transfer(args.api, kvikio, cp, pairs)
                for actual, expected in zip(arrays, host):
                    np.testing.assert_array_equal(cp.asnumpy(actual), expected)
                frame = cudf.DataFrame({"key": arrays[0], "value": arrays[1]})
                result = frame.groupby("key")["value"].sum().sort_index()
                expected = pd.DataFrame({"key": host[0], "value": host[1]}).groupby("key")["value"].sum().sort_index()
                pd.testing.assert_series_equal(result.to_pandas(), expected)
                output = root / "aggregates.bin"
                transfer(args.api, kvikio, cp, [(output, result.values)], writing=True)
                payload = output.read_bytes()
                assert payload == expected.to_numpy().tobytes()
                report["checks"] = ["RMM-backed column input", "GPU groupby", "GPU result output", "pandas reference"]
        else:
            if args.workflow == "native":
                # The legacy plugin's GPU batch decoder supports JPEG only.
                os.environ["ENABLE_CUSLIDE2"] = "1"
                from nvidia import nvimgcodec

                report["image_plugin"] = "cuslide2"
                report["nvimgcodec_module"] = nvimgcodec.__file__
            import cucim

            if args.workflow == "native":
                import tifffile

                expected = np.random.default_rng(91).integers(0, 256, (256, 384, 3), dtype=np.uint8)
                for compression in (None, "deflate"):
                    path = root / f"tiles-{compression}.tiff"
                    tifffile.imwrite(
                        path,
                        expected,
                        tile=(64, 64),
                        photometric="rgb",
                        compression=compression,
                        metadata={"axes": "YXC"},
                    )
                    with cucim.CuImage(str(path)) as image:
                        for device in ("cpu", "cuda"):
                            tile = image.read_region(location=(11, 17), size=(96, 80), level=0, device=device)
                            actual = cp.asnumpy(cp.asarray(tile)) if device == "cuda" else np.asarray(tile)
                            np.testing.assert_array_equal(actual, expected[17:97, 11:107])
                payload = expected.tobytes()
                report["checks"] = ["tiled TIFF", "uncompressed and deflate", "CPU and GPU crop", "NumPy reference"]
            else:
                from cucim.skimage.filters import gaussian
                from cucim.skimage.measure import label
                from skimage.filters import gaussian as cpu_gaussian
                from skimage.measure import label as cpu_label

                expected = np.zeros((512, 512), dtype=np.float32)
                expected[40:180, 30:220] = 1
                expected[300:450, 280:480] = 1
                path = root / "image.bin"
                path.write_bytes(expected.tobytes())
                source = cp.empty_like(cp.asarray(expected))
                transfer(args.api, kvikio, cp, [(path, source)])
                np.testing.assert_array_equal(cp.asnumpy(source), expected)
                smooth = gaussian(source, sigma=1.2, preserve_range=True)
                labels = cp.ascontiguousarray(label(smooth > 0.5).astype(cp.int32))
                cpu_smooth = cpu_gaussian(expected, sigma=1.2, preserve_range=True)
                cpu_labels = cpu_label(cpu_smooth > 0.5).astype(np.int32)
                np.testing.assert_allclose(cp.asnumpy(smooth), cpu_smooth, rtol=1e-5, atol=1e-6)
                np.testing.assert_array_equal(cp.asnumpy(labels), cpu_labels)
                assert int(labels.max()) == 2
                output = root / "labels.bin"
                transfer(args.api, kvikio, cp, [(output, labels)], writing=True)
                payload = output.read_bytes()
                assert payload == cpu_labels.tobytes()
                report["checks"] = [
                    "GPU image input",
                    "Gaussian smoothing",
                    "connected components",
                    "GPU label output",
                    "scikit-image reference",
                ]
        transfers = sum(native_counts.values())
        if args.workflow == "native":
            assert not api_counts, api_counts
            assert transfers == 0, native_counts
            report["routing"] = "framework-native path; Python import alias did not intercept I/O"
        else:
            if args.api != "batch":
                assert api_counts["__init__"] > 0, api_counts
            if args.backend == "python":
                assert transfers > 0, native_counts
            else:
                assert transfers == 0, native_counts
            report["routing"] = "Python cuFile bindings" if args.backend == "python" else "Python KvikIO / C++ backend"
        report.update(status="passed", output_sha256=hashlib.sha256(payload).hexdigest(), output_bytes=len(payload))
    except Exception:
        report.update(status="failed", traceback=traceback.format_exc())
    finally:
        report.update(
            python_api_calls=dict(api_counts),
            bindings_calls=dict(native_counts),
            loaded_libraries=sorted(
                {
                    line.split()[-1]
                    for line in Path("/proc/self/maps").read_text().splitlines()
                    if any(name in line for name in ["libkvikio", "libcudf", "libcufile", "libcucim"])
                }
            ),
        )
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report["status"] != "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="/mnt/nvme0/gpuio-agent-scratch")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["reference", "python"])
    parser.add_argument("--framework", choices=["cudf", "cucim"])
    parser.add_argument("--workflow", choices=["native", "explicit"])
    parser.add_argument("--api", choices=["read", "pread", "raw", "stream", "batch"], default="pread")
    args = parser.parse_args()
    if args.backend:
        return worker(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = output.parent / "framework-runs"
    artifacts.mkdir(exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    reports = []
    for framework in ("cudf", "cucim"):
        cases = [("native", "read")] + [("explicit", api) for api in ("read", "pread", "raw", "stream", "batch")]
        for workflow, api in cases:
            pair = []
            for backend in ("python",) if api == "batch" else ("reference", "python"):
                name = f"{framework}-{workflow}-{api}-{backend}"
                record = artifacts / (name + ".json")
                env = {
                    k: v
                    for k, v in os.environ.items()
                    if k not in ("PYTHONPATH", "LD_PRELOAD") and not k.startswith("KVIKIO_")
                }
                env.update(
                    PYTHONPATH=str(root),
                    KVIKIO_COMPAT_MODE="OFF",
                    KVIKIO_GDS_THRESHOLD="0",
                    KVIKIO_NTHREADS="4",
                    KVIKIO_TASK_SIZE=str(256 * 1024),
                )
                with tempfile.TemporaryDirectory(dir=args.directory, prefix="framework-io-") as directory:
                    command = [
                        sys.executable,
                        "-X",
                        "faulthandler",
                        str(Path(__file__).resolve()),
                        "--backend",
                        backend,
                        "--framework",
                        framework,
                        "--workflow",
                        workflow,
                        "--api",
                        api,
                        "--directory",
                        directory,
                        "--output",
                        str(record),
                    ]
                    with (artifacts / (name + ".log")).open("w") as log:
                        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
                report = (
                    json.loads(record.read_text())
                    if record.exists()
                    else {
                        "status": "failed",
                        "case": name,
                        "framework": framework,
                        "workflow": workflow,
                        "api": api,
                        "backend": backend,
                    }
                )
                if process.returncode:
                    report["status"] = "failed"
                report["returncode"] = process.returncode
                pair.append(report)
                reports.append(report)
                print(name, report["status"], flush=True)
                output.write_text(
                    json.dumps(
                        {
                            "application_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "runs": reports,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            if len(pair) == 2 and all(r["status"] == "passed" for r in pair):
                assert pair[0]["output_sha256"] == pair[1]["output_sha256"], name
    return any(r["status"] != "passed" or r["returncode"] != 0 for r in reports)


if __name__ == "__main__":
    raise SystemExit(main())

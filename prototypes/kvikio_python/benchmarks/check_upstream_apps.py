# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute unchanged upstream programs against both backends in fresh processes."""

import argparse
import hashlib
import importlib.util
import json
import os
import runpy
import subprocess
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

COMMIT = "5530d767aeb55f63db59a7bb64f2745e8061eee5"
SOURCES = {
    "hello_world": "python/kvikio/examples/hello_world.py",
    "single_node_io": "python/kvikio/kvikio/benchmarks/single_node_io.py",
}
SUPPORT_SOURCES = {"benchmark_utils": "python/kvikio/kvikio/benchmarks/utils.py"}


def worker(args):
    import numpy as np

    if args.worker == "python":
        from cufile_python.compat import enable_kvikio_compat

        enable_kvikio_compat()
        # Benchmark helpers are test fixtures, not part of the I/O package.
        package = types.ModuleType("kvikio.benchmarks")
        package.__path__ = []
        sys.modules[package.__name__] = package
        spec = importlib.util.spec_from_file_location(
            "kvikio.benchmarks.utils", Path(args.source) / "benchmark_utils.py"
        )
        helpers = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = helpers
        spec.loader.exec_module(helpers)
    import kvikio
    import kvikio.defaults

    directory = Path(args.directory)
    program = Path(args.source) / (args.application + ".py")
    if args.application == "single_node_io":
        sys.argv = [
            str(program),
            "--dir",
            str(directory),
            "--nbytes",
            "16MiB",
            "--nthreads",
            "4",
            "--nruns",
            "1",
            "--pre-register-buffer",
            "--api",
            "cufile",
            "cufile-mfma",
            "cufile-mf",
            "cufile-ma",
        ]
        runpy.run_path(str(program), run_name="__main__")
        files = [
            directory / name
            for name in ("kvikio-single-file", "kvikio-multiple-arrays", *(f"cufile-p-{i:03d}" for i in range(4)))
        ]
        for path in files:
            actual = np.fromfile(path, dtype=np.uint8)
            np.testing.assert_array_equal(actual, np.arange(actual.size, dtype=np.uint8))
    else:
        main = runpy.run_path(str(program))["main"]
        main(directory / "example.bin")
        files = [directory / "example.bin"]
        for path in files:
            np.testing.assert_array_equal(np.fromfile(path, dtype=np.int64), np.arange(100))

    loaded = "libkvikio.so" in Path("/proc/self/maps").read_text()
    if loaded != (args.worker == "reference"):
        raise AssertionError(f"Unexpected libkvikio loading: {loaded}")
    package = str(Path(kvikio.__file__).resolve())
    prototype = str(Path(__file__).resolve().parents[1])
    if (args.worker == "python") != package.startswith(prototype + os.sep):
        raise AssertionError(f"Unexpected package: {package}")
    report = {
        "implementation": args.worker,
        "application": args.application,
        "package": package,
        "version": kvikio.__version__,
        "libkvikio_loaded": loaded,
        "cufile_bindings_imported": "cuda.bindings.cufile" in sys.modules,
        "routing_policy": "cuFile required; compatibility settings are no-ops" if args.worker == "python" else "KvikIO",
        "settings": {
            key: str(kvikio.defaults.get(key)) for key in ("compat_mode", "num_threads", "task_size", "gds_threshold")
        },
        "verified_files": {
            p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files
        },
        "status": "passed",
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/tmp/kvikio-reference-26.10")
    parser.add_argument("--directory", default="/mnt/nvme0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", choices=["reference", "python"])
    parser.add_argument("--application", choices=list(SOURCES))
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return

    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(args.source).resolve()
    manifest, reports = {}, []
    profiles = [("hello_world", mode) for mode in ("AUTO", "ON", "OFF")]
    profiles += [("single_node_io", mode) for mode in ("ON", "OFF")]
    with tempfile.TemporaryDirectory(prefix="kvikio-upstream-apps-", dir=args.directory) as temporary:
        temporary = Path(temporary)
        for name, relative in (SOURCES | SUPPORT_SOURCES).items():
            url = f"https://raw.githubusercontent.com/rapidsai/kvikio/{COMMIT}/{relative}"
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read()
            if payload != (source / relative).read_bytes():
                raise AssertionError(f"Online source differs from inspected checkout: {relative}")
            (temporary / (name + ".py")).write_bytes(payload)
            manifest[name] = {"url": url, "sha256": hashlib.sha256(payload).hexdigest()}

        for application, mode in profiles:
            pair = []
            for implementation in ("reference", "python"):
                label = f"{application}-{mode}-{implementation}"
                directory = temporary / label
                directory.mkdir()
                record = directory / "result.json"
                env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith("KVIKIO_")}
                env.update(
                    KVIKIO_COMPAT_MODE=mode,
                    KVIKIO_NTHREADS="4",
                    KVIKIO_GDS_THRESHOLD="0",
                    KVIKIO_TASK_SIZE=str(1024**2),
                )
                if implementation == "python":
                    env["PYTHONPATH"] = str(root)
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        implementation,
                        "--application",
                        application,
                        "--source",
                        str(temporary),
                        "--directory",
                        str(directory),
                        "--output",
                        str(record),
                    ],
                    cwd=directory,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode:
                    print(result.stdout, end="")
                    print(result.stderr, file=sys.stderr, end="")
                    result.check_returncode()
                item = json.loads(record.read_text())
                item.update(stdout=result.stdout, stderr=result.stderr)
                pair.append(item)
                reports.append(item)
                print(f"PASS {label}", flush=True)
            if pair[0]["verified_files"] != pair[1]["verified_files"]:
                raise AssertionError(f"Output differs across backends: {application}/{mode}")
    output.write_text(json.dumps({"source_commit": COMMIT, "sources": manifest, "runs": reports}, indent=2) + "\n")
    print(f"{len(reports)} application runs passed; sources unchanged and output hashes match.")


if __name__ == "__main__":
    main()

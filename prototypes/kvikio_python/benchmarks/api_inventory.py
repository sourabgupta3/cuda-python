# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare the selected CuFile API signatures with an exact upstream checkout."""

import argparse
import ast
import json
from pathlib import Path

MODULES = frozenset({"cufile", "buffer", "stream", "cufile_driver", "defaults", "statistics", "utils"})


def inventory(root, modules=MODULES, namespace="kvikio"):
    result = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if len(relative.parts) != 1 or path.stem not in modules:
            continue
        module = namespace + "." + str(relative.with_suffix("")).replace("/", ".")
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                result[module + "." + node.name] = ast.unparse(node.args)
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                if path.stem == "utils" and node.name == "LocalHttpServer":
                    continue
                result[module + "." + node.name] = {"bases": [ast.unparse(base) for base in node.bases]}
                for method in node.body:
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                        not method.name.startswith("_") or method.name in ("__init__", "__enter__", "__exit__")
                    ):
                        result[module + "." + node.name + "." + method.name] = dict(
                            arguments=ast.unparse(method.args),
                            returns=ast.unparse(method.returns) if method.returns else None,
                            decorators=[
                                ast.unparse(d).replace("cufile_python.", "kvikio.") for d in method.decorator_list
                            ],
                        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "docs/public_api.json")
    args = parser.parse_args()
    reference = inventory(args.reference)
    current = inventory(Path(__file__).resolve().parents[1] / "cufile_python")
    missing = sorted(reference.keys() - current.keys())
    changed = [key for key in reference.keys() & current.keys() if reference[key] != current[key]]
    added = sorted(current.keys() - reference.keys())
    report = dict(
        target="KvikIO 26.10.0",
        implementation="cufile_python (namespace normalized for comparison)",
        scope="CuFile local file I/O and supporting APIs; not the complete KvikIO package",
        modules=sorted(MODULES),
        excluded=[
            "RemoteFile",
            "LocalHttpServer",
            "remote settings",
            "Mmap",
            "NumPy file helpers",
            "Zarr",
            "Numcodecs",
            "Nsight NIC helper",
        ],
        source_commit="5530d767aeb55f63db59a7bb64f2745e8061eee5",
        members=len(reference),
        intentional_behavior_differences=[
            "All host/device I/O requires cuFile; no Python POSIX fallback",
            "compat_mode, gds_threshold, bounce_buffer_size, and auto_direct_io_read/write do not affect I/O",
            "is_compat_mode_preferred always returns False",
            "Memory registration always calls cuFile; bounce_buffer_free returns zero",
            "The retained gds statistics bucket denotes cuFile calls, not proof of physical GDS",
        ],
        missing=missing,
        changed=changed,
        added=added,
        extensions=inventory(
            Path(__file__).resolve().parents[1] / "cufile_python", frozenset({"batch"}), "cufile_python"
        ),
        api=reference,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "api"}))
    if missing or changed or added:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare two directories of result JSON files, number by number.

    python scripts/compare_results.py results/ /tmp/results_new/

Exit status 0 if every file holds the same structure and every number agrees
within the tolerances, 1 otherwise. On the software versions listed in
requirements-lock.txt the agreement is exact; the default tolerance only absorbs
last-digit differences that a different NumPy/SciPy build may introduce.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _diff(a, b, path, rtol, atol, out, ignore):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in ignore and not path:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present on one side only")
            else:
                _diff(a[k], b[k], f"{path}/{k}", rtol, atol, out, ignore)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", rtol, atol, out, ignore)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        if math.isnan(a) and math.isnan(b):
            return
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            out.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--max-report", type=int, default=15)
    args = ap.parse_args()

    ref = {p.name for p in args.reference.glob("*.json")}
    new = {p.name for p in args.candidate.glob("*.json")}
    bad = 0
    for name in sorted(ref | new):
        if name not in ref or name not in new:
            print(f"[FAIL] {name}: missing on one side")
            bad += 1
            continue
        a = json.loads((args.reference / name).read_text())
        b = json.loads((args.candidate / name).read_text())
        diffs: list[str] = []
        _diff(a, b, "", args.rtol, args.atol, diffs, ignore=())
        if diffs:
            bad += 1
            print(f"[FAIL] {name}: {len(diffs)} difference(s)")
            for d in diffs[: args.max_report]:
                print("       ", d)
        else:
            print(f"[ ok ] {name}")
    print("IDENTICAL" if not bad else f"{bad} file(s) differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

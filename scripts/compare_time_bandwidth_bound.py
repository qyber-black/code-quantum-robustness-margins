#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare MATLAB and Python Kosut-comparison tables.

Peer of compare_matlab_python_full.py for the supplementary Kosut et al. bound
(arXiv:2507.01215). Defaults to the MATLAB and Python trees; pass explicit
paths to compare any other pair (e.g. an Octave run).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/matlab_python_kosut_compare.txt"

ATOL = 1e-10
RTOL = 1e-8

STRUCTURES = ("H0", "H1", "H2")
PER_STRUCTURE = ("M", "KM", "ratio", "KTOb", "Kflb", "wunc", "wavg", "wdev")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--FT", type=float, default=0.999)
    ap.add_argument("--a", type=Path, default=None, help="First CSV (default: MATLAB tree)")
    ap.add_argument("--b", type=Path, default=None, help="Second CSV (default: Python tree)")
    ap.add_argument("--label-a", default="matlab")
    ap.add_argument("--label-b", default="python")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    name = f"kosut_comparison_{args.FT:g}.csv"
    path_a = args.a or ROOT / "results/time-bandwidth-bound-matlab" / name
    path_b = args.b or ROOT / "results/time-bandwidth-bound-python" / name
    for p, hint in ((path_a, "make time-bandwidth-bound-matlab"), (path_b, "make time-bandwidth-bound-python")):
        if not p.exists():
            print(f"Missing {p}; run {hint}", file=sys.stderr)
            return 2

    A = np.genfromtxt(path_a, delimiter=",", names=True)
    B = np.genfromtxt(path_b, delimiter=",", names=True)
    if A.shape[0] != B.shape[0]:
        print(f"Row count mismatch: {args.label_a}={A.shape[0]} {args.label_b}={B.shape[0]}",
              file=sys.stderr)
        return 1

    fields = ["fid", "err"] + [f"{f}_{tag}" for tag in STRUCTURES for f in PER_STRUCTURE]
    lines = [
        f"Kosut-bound comparison, {A.shape[0]} controllers, FT={args.FT:g}",
        f"{args.label_a}={path_a}",
        f"{args.label_b}={path_b}",
        f"atol={ATOL} rtol={RTOL}",
    ]
    ok = True
    for f in fields:
        a, b = A[f], B[f]
        absd = np.abs(a - b)
        denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-15)
        close = absd <= (ATOL + RTOL * denom)
        nfail = int((~close).sum())
        lines.append(f"{f}: max_abs={absd.max():.6e} max_rel={(absd/denom).max():.6e} fail={nfail}")
        if nfail:
            ok = False
            for i in np.where(~close)[0][:5]:
                lines.append(f"  fail row {i+1}: {args.label_a}={a[i]!r} {args.label_b}={b[i]!r}")

    lines.append(f"overall={'PASS' if ok else 'FAIL'}")
    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text, end="")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

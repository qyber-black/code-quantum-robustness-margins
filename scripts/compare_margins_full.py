#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare two engines' full case-study margin tables.

Python is the reference implementation, so the useful comparison is
always <engine> against python; comparing two non-reference engines to
each other only establishes that they agree, not that either is right.
Defaults reproduce the historical matlab-vs-python invocation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ATOL = 1e-10
RTOL = 1e-8


def table(engine: str) -> Path:
    return ROOT / f"results/lipschitz-margin-{engine}/margins_table_0.999.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="matlab", help="Engine under test (default: matlab)")
    ap.add_argument("--b", default="python", help="Reference engine (default: python)")
    ap.add_argument("--a-csv", type=Path, default=None, help="Override the --a table")
    ap.add_argument("--b-csv", type=Path, default=None, help="Override the --b table")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    a_csv = args.a_csv or table(args.a)
    b_csv = args.b_csv or table(args.b)
    out = args.out or ROOT / f"build/{args.a}_{args.b}_full_compare.txt"

    for engine, csv_path in ((args.a, a_csv), (args.b, b_csv)):
        if not csv_path.exists():
            print(
                f"Missing {csv_path}; run make paper-QRM-margins ENGINE={engine}",
                file=sys.stderr,
            )
            return 2

    A = np.genfromtxt(a_csv, delimiter=",", names=True)
    B = np.genfromtxt(b_csv, delimiter=",", names=True)
    if A.shape[0] != B.shape[0]:
        print(
            f"Row count mismatch: {args.a}={A.shape[0]} {args.b}={B.shape[0]}",
            file=sys.stderr,
        )
        return 1

    fields = ["fid", "err"]
    for tag in ("H0", "H1", "H2"):
        fields += [f"M_{tag}", f"Mm_{tag}", f"Mp_{tag}", f"zeta_{tag}"]

    lines = [
        f"Full {A.shape[0]}-controller {args.a} vs {args.b} comparison",
        f"{args.a}={a_csv}",
        f"{args.b}={b_csv}",
        f"atol={ATOL} rtol={RTOL}",
    ]
    ok = True
    for f in fields:
        a, b = A[f], B[f]
        absd = np.abs(a - b)
        denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-15)
        close = absd <= (ATOL + RTOL * denom)
        nfail = int((~close).sum())
        lines.append(
            f"{f}: max_abs={absd.max():.6e} "
            f"max_rel={(absd / denom).max():.6e} fail={nfail}"
        )
        if nfail:
            ok = False
            for i in np.where(~close)[0][:5]:
                lines.append(f"  fail row {i + 1}: {args.a}={a[i]!r} {args.b}={b[i]!r}")

    lines.append(f"overall={'PASS' if ok else 'FAIL'}")
    text = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text, end="")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

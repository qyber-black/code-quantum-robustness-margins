#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare MATLAB and Python Kosut-comparison tables.

Peer of compare_margins_full.py for the supplementary Kosut et al. bound
(arXiv:2507.01215). Defaults to the MATLAB and Python trees; pass explicit
paths to compare any other pair (e.g. an Octave run).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _compare import ATOL, RTOL, compare_fields, load_pair, write_report
from _drivers import DEFAULT_FT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/matlab_python_kosut_compare.txt"

STRUCTURES = ("H0", "H1", "H2")
PER_STRUCTURE = ("M", "KM", "ratio", "KTOb", "Kflb", "wunc", "wavg", "wdev")


def main() -> int:
    """Compare the two engines' Kosut tables and write the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--FT", type=float, default=DEFAULT_FT)
    ap.add_argument(
        "--a", type=Path, default=None, help="First CSV (default: MATLAB tree)"
    )
    ap.add_argument(
        "--b", type=Path, default=None, help="Second CSV (default: Python tree)"
    )
    ap.add_argument("--label-a", default="matlab")
    ap.add_argument("--label-b", default="python")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    name = f"kosut_comparison_{args.FT:g}.csv"
    path_a = args.a or ROOT / "results/time-bandwidth-bound-matlab" / name
    path_b = args.b or ROOT / "results/time-bandwidth-bound-python" / name
    for p in (path_a, path_b):
        if not p.exists():
            # Name the engine whose file is missing and a target that
            # exists, so the hint can be followed as printed.
            engine = p.parent.name.rsplit("-", 1)[-1]
            print(
                f"Missing {p}; run make paper-QRM-time-bandwidth ENGINE={engine}",
                file=sys.stderr,
            )
            return 2

    A, B, err = load_pair(path_a, path_b, args.label_a, args.label_b)
    if err is not None:
        print(err, file=sys.stderr)
        return 1

    fields = ["fid", "err"] + [
        f"{f}_{tag}" for tag in STRUCTURES for f in PER_STRUCTURE
    ]
    lines = [
        f"Kosut-bound comparison, {A.shape[0]} controllers, FT={args.FT:g}",
        f"{args.label_a}={path_a}",
        f"{args.label_b}={path_b}",
        f"atol={ATOL} rtol={RTOL}",
    ]
    ok, body = compare_fields(A, B, fields, args.label_a, args.label_b)
    return write_report(args.out, lines + body, ok)


if __name__ == "__main__":
    raise SystemExit(main())

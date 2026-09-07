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

from _compare import ATOL, RTOL, compare_fields, load_pair, write_report

ROOT = Path(__file__).resolve().parents[1]


def table(engine: str) -> Path:
    """The margins table an engine's driver writes."""
    return ROOT / f"results/lipschitz-margin-{engine}/margins_table_0.999.csv"


def main() -> int:
    """Compare the two engines' tables and write the report."""
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

    A, B, err = load_pair(a_csv, b_csv, args.a, args.b)
    if err is not None:
        print(err, file=sys.stderr)
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
    ok, body = compare_fields(A, B, fields, args.a, args.b)
    return write_report(out, lines + body, ok)


if __name__ == "__main__":
    raise SystemExit(main())

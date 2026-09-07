#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The comparison shared by the cross-engine table checks.

compare_margins_full.py and compare_time_bandwidth_bound.py differed only
in which columns they read and what they printed above the results; the
tolerances, the mixed absolute/relative test, the zero-guarded denominator
and the failure listing were copied between them. Two copies of a
tolerance is how a peer check comes to pass on one table and not another,
so they live here once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Peer engines run the same algorithm in a different language, so the
# disagreement to allow for is accumulated rounding, not method error.
ATOL = 1e-10
RTOL = 1e-8

# Floor on the relative denominator: below this a column is numerically
# zero and the relative test would divide by rounding noise.
DENOM_FLOOR = 1e-15

# A failing column lists offenders rather than every row, since a genuine
# divergence usually shows in the first few.
MAX_LISTED = 5


def load_pair(path_a: Path, path_b: Path, label_a: str, label_b: str):
    """Read two named-column CSVs, or report why they cannot be compared.

    Returns ``(A, B, None)`` on success and ``(None, None, message)`` when
    the row counts differ, so the caller decides the exit code.
    """
    A = np.genfromtxt(path_a, delimiter=",", names=True)
    B = np.genfromtxt(path_b, delimiter=",", names=True)
    if A.shape[0] != B.shape[0]:
        return (
            None,
            None,
            (f"Row count mismatch: {label_a}={A.shape[0]} {label_b}={B.shape[0]}"),
        )
    return A, B, None


def compare_fields(A, B, fields, label_a: str, label_b: str):
    """Compare the named columns of two tables.

    Returns ``(ok, lines)``: one summary line per field, followed by up to
    ``MAX_LISTED`` offending rows for each field that failed.
    """
    ok = True
    lines = []
    for f in fields:
        a, b = A[f], B[f]
        absd = np.abs(a - b)
        denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), DENOM_FLOOR)
        close = absd <= (ATOL + RTOL * denom)
        nfail = int((~close).sum())
        lines.append(
            f"{f}: max_abs={absd.max():.6e} "
            f"max_rel={(absd / denom).max():.6e} fail={nfail}"
        )
        if nfail:
            ok = False
            for i in np.where(~close)[0][:MAX_LISTED]:
                lines.append(
                    f"  fail row {i + 1}: {label_a}={a[i]!r} {label_b}={b[i]!r}"
                )
    return ok, lines


def write_report(out: Path, lines, ok: bool) -> int:
    """Write the report, echo it, and return the process exit code."""
    lines = list(lines) + [f"overall={'PASS' if ok else 'FAIL'}"]
    text = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text, end="")
    return 0 if ok else 1

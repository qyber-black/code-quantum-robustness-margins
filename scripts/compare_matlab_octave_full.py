#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare MATLAB and Octave full case-study margin tables."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MATLAB_CSV = ROOT / "results/lipschitz-margin-matlab/margins_table_0.999.csv"
OCTAVE_CSV = ROOT / "results/lipschitz-margin-octave/margins_table_0.999.csv"
OUT = ROOT / "build/matlab_octave_full_compare.txt"

ATOL = 1e-10
RTOL = 1e-8


def main() -> int:
    if not MATLAB_CSV.exists():
        print(f"Missing {MATLAB_CSV}; run make lipschitz-margin-matlab", file=sys.stderr)
        return 2
    if not OCTAVE_CSV.exists():
        print(f"Missing {OCTAVE_CSV}; run make lipschitz-margin-octave", file=sys.stderr)
        return 2

    mat = np.genfromtxt(MATLAB_CSV, delimiter=",", names=True)
    oct_ = np.genfromtxt(OCTAVE_CSV, delimiter=",", names=True)
    if mat.shape[0] != oct_.shape[0]:
        print(
            f"Row count mismatch: matlab={mat.shape[0]} octave={oct_.shape[0]}",
            file=sys.stderr,
        )
        return 1

    fields = ["fid", "err"]
    for tag in ("H0", "H1", "H2"):
        fields += [f"M_{tag}", f"Mm_{tag}", f"Mp_{tag}", f"zeta_{tag}"]

    lines = [
        f"Full {mat.shape[0]}-controller MATLAB vs Octave comparison",
        f"matlab={MATLAB_CSV}",
        f"octave={OCTAVE_CSV}",
        f"atol={ATOL} rtol={RTOL}",
    ]
    ok = True
    for f in fields:
        a = mat[f]
        b = oct_[f]
        absd = np.abs(a - b)
        denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-15)
        close = absd <= (ATOL + RTOL * denom)
        nfail = int((~close).sum())
        lines.append(
            f"{f}: max_abs={absd.max():.6e} max_rel={(absd / denom).max():.6e} fail={nfail}"
        )
        if nfail:
            ok = False
            idx = np.where(~close)[0][:5]
            for i in idx:
                lines.append(f"  fail row {i + 1}: matlab={a[i]!r} octave={b[i]!r}")

    lines.append(f"overall={'PASS' if ok else 'FAIL'}")
    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(text, end="")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

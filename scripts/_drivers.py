#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plumbing shared by the experiment drivers.

Nineteen drivers repeated the same ROOT/CTRL preamble and ensemble load,
fourteen the same DictWriter block, fifteen the same --FT/--out options.
That is plumbing, not science, and duplicating it is how the three paper
generators drifted apart from each other.

Nothing here computes a result. The helpers reproduce what the copies did
exactly -- same default error filter, same CSV dialect, same field order
-- so moving a driver onto them cannot change its output.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from qrobustness import load_controllers, load_problem

#: Repository root, from this file's location.
ROOT = Path(__file__).resolve().parents[1]

#: The shipped three-qubit ensemble every xQRM driver starts from.
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"

#: The error filter ten drivers hardcoded and seven exposed as --max-error.
DEFAULT_MAX_ERROR = 1e-4


def load_ensemble(ctrl_dir: Path = CTRL, max_error: float = DEFAULT_MAX_ERROR):
    """The problem definition and the controllers passing the error filter.

    Returns ``(problem, controllers)``, exactly as the nineteen inline
    copies did.
    """
    problem = load_problem(ctrl_dir / "problem9.mat")
    controllers = load_controllers(ctrl_dir / "controllers.csv", max_error)
    return problem, controllers


def write_rows(path: Path, rows: list, fieldnames: list | None = None) -> Path:
    """Write ``rows`` as CSV, creating the parent directory.

    ``lineterminator="\\n"`` is not cosmetic: the MATLAB and Octave peers
    write LF, and the parity comparison is byte-oriented. ``fieldnames``
    defaults to the first row's key order, which is what the copies used.
    """
    if not rows:
        raise SystemExit(f"ERROR: refusing to write {path} with no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            # `or` would treat an explicitly empty list as "use them all";
            # an empty selection should write a header-only file.
            fieldnames=list(rows[0].keys()) if fieldnames is None else fieldnames,
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")
    return path


def base_parser(
    out_dir: Path, description: str | None = None
) -> argparse.ArgumentParser:
    """Parser carrying the --FT and --out options every driver defines."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--FT", type=float, default=0.999)
    ap.add_argument(
        "--out",
        type=Path,
        default=out_dir,
        help="write results here instead of the default tree",
    )
    return ap


#: Single-qubit operators the CNOT drivers also use outside the model
#: itself (run_cnot_case_study builds its dephasing structure from them).
PAULI_Z = np.diag([1.0, -1.0]).astype(complex)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
EYE2 = np.eye(2, dtype=complex)


def cnot_model():
    """The two-qubit CNOT model three drivers define identically.

    Returns ``(H0, X1, X2, CNOT)``: the ZZ-coupled drift with a local-Z
    detuning, the two control structures, and the target gate. The
    expressions are those the three copies used, unchanged, so the
    Kronecker products and their floating-point results are the same.
    """
    Z, X, I2 = PAULI_Z, PAULI_X, EYE2
    H0 = 2 * np.pi * 0.5 * np.kron(Z, Z) + np.pi * 0.1 * (
        np.kron(Z, I2) - np.kron(I2, Z)
    )
    X1 = np.kron(X, I2)
    X2 = np.kron(I2, X)
    CNOT = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )
    return H0, X1, X2, CNOT


def three_structure_specs(problem, c):
    """The p=3 uncertainty spec six drivers build identically.

    Drift plus the two multiplicative control errors, in the order
    ``mp.structure_constants`` expects. Returned as a fresh list of the
    same tuples the inline copies built, so the constants it yields are
    unchanged.
    """
    return [
        ("drift", problem["H0"]),
        ("control", problem["H1"], c["u1"]),
        ("control", problem["H2"], c["u2"]),
    ]

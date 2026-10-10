#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Helpers shared by the experiment drivers.

Paths, default settings, ensemble loading, CSV writing, the common --FT/--out
options, the CNOT model and the three-structure uncertainty spec. Nothing
here computes a result."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from qrobustness import load_controllers, load_problem

#: Repository root, from this file's location.
ROOT = Path(__file__).resolve().parents[1]

#: The shipped three-qubit ensemble that every xQRM driver starts from.
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"

#: Default nominal-error filter for the ensemble (--max-error).
DEFAULT_MAX_ERROR = 1e-4

#: Default fidelity threshold F_T (--FT). Output file names carry the
#: threshold they were written at (..._0.999.csv); it is a default, not a constant.
DEFAULT_FT = 0.999

#: Safe-radius continuation step eta of the margin algorithm (numerical setting).
DEFAULT_ETA = 1e-6

#: Magnus refinement levels: sub-steps per control interval of the
#: piecewise-constant trajectory. Index 0 is the control grid itself.
REFINEMENTS = (1, 4, 16)

#: Controller and structure of the constant-margin counterexample, and the
#: default of the budget-sweep figure. The macro generator emits these only
#: when the validity file records exactly this one violation.
WITNESS_CONTROLLER = 16
WITNESS_STRUCTURE = "H1"

#: Starts per production attack of run_kosut_validity (--starts mixed).
#: Two are the constant extremes; the rest split evenly into sign-modulated
#: boundary trajectories and uniform interior draws.
ATTACK_STARTS = 12

#: Slack below F_T before an adversarial fidelity counts as a violation
#: (rounding of one fidelity evaluation). Shared by the validity drivers so
#: their counts are comparable.
VIOLATION_TOL = 1e-10

#: Quadrature nodes for the differential sensitivity zeta (case-study driver
#: and consistency tests).
ZETA_N_QUAD = 32

#: Bracketing for :func:`true_crossing`: grow the upper end by GROWTH from
#: max(GROWTH * seed, FLOOR) up to CAP, then bisect STEPS times.
CROSSING_GROWTH = 10.0
CROSSING_FLOOR = 1e-6
CROSSING_CAP = 1e3
CROSSING_STEPS = 40

#: Seed strides giving each (controller, structure, refinement, budget)
#: attack its own random stream: budget in the units, refinement in the
#: tens, structure in the hundreds. Shared by run_fs_validity and
#: run_kosut_validity.
SEED_STRIDE_REFINEMENT = 10
SEED_STRIDE_STRUCTURE = 100
SEED_STRIDE_CTRL = 3 * SEED_STRIDE_STRUCTURE


def load_ensemble(ctrl_dir: Path = CTRL, max_error: float = DEFAULT_MAX_ERROR):
    """Load the problem definition and the controllers that pass the error filter.

    Returns ``(problem, controllers)``."""
    problem = load_problem(ctrl_dir / "problem9.mat")
    controllers = load_controllers(ctrl_dir / "controllers.csv", max_error)
    return problem, controllers


def write_rows(path: Path, rows: list, fieldnames: list | None = None) -> Path:
    """Write ``rows`` as CSV, creating the parent directory.

    LF line endings, as the MATLAB and Octave peers write, since the parity
    comparison is byte-oriented. ``fieldnames`` defaults to the first row's
    key order. Refuses to write an empty ``rows``."""
    if not rows:
        raise SystemExit(f"ERROR: refusing to write {path} with no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            # An explicitly empty fieldnames list gives a header-only file.
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
    """Parser that carries the --FT and --out options every driver defines."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--FT", type=float, default=DEFAULT_FT)
    ap.add_argument(
        "--out",
        type=Path,
        default=out_dir,
        help="write results here instead of the default tree",
    )
    return ap


#: Single-qubit Pauli Z, X and identity (also used by run_cnot_case_study).
PAULI_Z = np.diag([1.0, -1.0]).astype(complex)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
EYE2 = np.eye(2, dtype=complex)


def true_crossing(
    F,
    ft: float,
    seed: float,
    *,
    lo: float | None = None,
    steps: int = CROSSING_STEPS,
    rel_tol: float | None = None,
) -> float:
    """Certified lower end of the true threshold crossing.

    ``F`` is a one-parameter fidelity and ``ft`` the threshold F_T. ``seed``
    is a point known to satisfy F >= ft (normally a certified margin) and sets
    the initial upper end; ``lo`` starts the lower end elsewhere. The upper end
    grows geometrically until F drops below ft or the cap is reached, then the
    bracket is bisected ``steps`` times, stopping early once its relative width
    is below ``rel_tol`` if given.

    Returns the bracket's lower end, where F >= ft was evaluated: a lower
    bound on the crossing, not an estimate of it."""
    hi = max(CROSSING_GROWTH * seed, CROSSING_FLOOR)
    lo = seed if lo is None else lo
    while F(hi) >= ft and hi < CROSSING_CAP:
        lo, hi = hi, CROSSING_GROWTH * hi
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if F(mid) >= ft:
            lo = mid
        else:
            hi = mid
        if rel_tol is not None and (hi - lo) / hi < rel_tol:
            break
    return lo


def cnot_model():
    """The two-qubit CNOT model used by the CNOT drivers.

    Returns ``(H0, X1, X2, CNOT)``: the ZZ-coupled drift with a local-Z
    detuning, the two control Hamiltonians, and the target gate."""
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
    """The p=3 uncertainty spec for controller ``c``: drift plus the two
    multiplicative control errors, in the order ``mp.structure_constants``
    expects. Returns a fresh list."""
    return [
        ("drift", problem["H0"]),
        ("control", problem["H1"], c["u1"]),
        ("control", problem["H2"], c["u2"]),
    ]

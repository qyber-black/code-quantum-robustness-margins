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

#: The paper's fidelity threshold, declared once. Twelve drivers each carried
#: their own FT = 0.999 and fourteen expose it as --FT, so the value lived in a
#: dozen places while being one number. Output files encode the threshold they
#: were produced at (..._0.999.csv), and the threshold sweep deliberately runs
#: 0.99 / 0.999 / 0.9999, so this is a default to override, never a constant to
#: assume.
DEFAULT_FT = 0.999

#: Safe-radius continuation step for Algorithm 1. Ten drivers carried their own
#: ETA = 1e-6; it is a numerical setting with one correct value, not an
#: experiment parameter.
DEFAULT_ETA = 1e-6

#: Magnus refinement levels: a piecewise-constant trajectory resolved at
#: one, four and sixteen sub-steps per control interval. Five drivers and
#: the table generator sweep exactly these, and a certificate that only
#: fails under refinement is the thing they are looking for, so the levels
#: are named once rather than retyped.
REFINEMENTS = (1, 4, 16)

#: Slack on the threshold before an adversarial fidelity counts as a
#: violation: one fidelity evaluation of rounding, not a tolerance on the
#: certificate. Three validity drivers apply the same allowance, so a
#: change to it must move all three together or their counts stop being
#: comparable.
VIOLATION_TOL = 1e-10

#: Quadrature nodes for the differential sensitivity zeta, shared by the
#: case-study driver and the consistency tests so both engines are held to
#: the same setting.
ZETA_N_QUAD = 32

#: Bracketing for :func:`true_crossing`: grow the upper end by this factor
#: from a floor that is non-zero even when the certified margin is, stop at
#: the cap, then halve. Forty halvings take the bracket far below the
#: precision at which a crossing is ever reported.
CROSSING_GROWTH = 10.0
CROSSING_FLOOR = 1e-6
CROSSING_CAP = 1e3
CROSSING_STEPS = 40

#: Seed strides that keep every attack on its own random stream. The
#: budget index occupies the units place, the refinement index the tens,
#: the structure the hundreds, and a controller advances past all three of
#: its structures. run_fs_validity and run_kosut_validity attack the same
#: (controller, structure, refinement, budget) grid, so they share the
#: layout; a driver that changed it alone would silently reuse another's
#: streams.
SEED_STRIDE_REFINEMENT = 10
SEED_STRIDE_STRUCTURE = 100
SEED_STRIDE_CTRL = 3 * SEED_STRIDE_STRUCTURE


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
    ap.add_argument("--FT", type=float, default=DEFAULT_FT)
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

    ``F`` is a one-parameter fidelity and ``ft`` the threshold. ``seed`` is
    a point known to satisfy it -- normally a certified margin, so what is
    returned measures that certificate's conservatism rather than assuming
    it -- and sets the initial upper end; pass ``lo`` to start the lower
    end somewhere else. The upper end grows geometrically until the
    fidelity drops or the cap is reached, then the bracket is halved
    ``steps`` times, stopping early once its relative width falls below
    ``rel_tol`` if one is given.

    The value returned is the bracket's LOWER end, at which the fidelity
    was evaluated and found to meet the threshold. It is therefore a
    guarantee -- a point the gate provably survives, hence a lower bound
    on the crossing -- where the bracket's midpoint would be an estimate
    with nothing behind it. Every conservatism ratio built on this is a
    lower bound in the same way, and the paper reports it as one.
    """
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

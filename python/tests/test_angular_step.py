# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Angular (Choi-FS) safe-radius rule in the certified directional
iteration: dominance, crossing agreement, island safety, and the
bit-for-bit default contract."""

from pathlib import Path

import numpy as np
import pytest

from qrobustness import iterative_margin, load_controllers, load_problem
from qrobustness import multiparam as mp

ROOT = Path(__file__).resolve().parents[2]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
FT = 0.999


@pytest.fixture(scope="module")
def setup():
    problem = load_problem(CTRL / "problem9.mat")
    c = load_controllers(CTRL / "controllers.csv", 1e-4)[0]
    dt = c["tf"] / c["tau"]
    dHs = [
        [problem["H0"]] * c["tau"],
        [c["u1"][k] * problem["H1"] for k in range(c["tau"])],
        [c["u2"][k] * problem["H2"] for k in range(c["tau"])],
    ]
    specs = [
        ("drift", problem["H0"]),
        ("control", problem["H1"], c["u1"]),
        ("control", problem["H2"], c["u2"]),
    ]
    C, L = mp.structure_constants(specs, dt, c["tau"], FT, problem["dim"])
    fn = mp.make_multiparam_fidelity_fn(
        problem["H0"],
        problem["H1"],
        problem["H2"],
        c["u1"],
        c["u2"],
        problem["Uf"],
        dt,
        ("H0", "H1", "H2"),
    )
    return problem, c, dt, dHs, L, fn


def test_angular_step_dominates_lipschitz(setup):
    """Theorem (full-gauge dominance) on ensemble data: the angular safe
    step is never smaller than the Lipschitz step, at any safe fidelity."""
    problem, c, dt, dHs, L, fn = setup
    G = mp.joint_gauge(dHs, dt)
    AG = mp.angular_gauge(dHs, dt)
    directions = np.vstack([mp.axis_directions(3), mp.diagonal_directions(3)])
    acFT = np.arccos(FT)
    for d in directions:
        L_dir = G.L_dir(d, FT, problem["dim"])
        C_FS = AG.C(d)
        for F_nu in (FT + 1e-6, 0.9995, 0.9999, 1.0 - 1e-9, 1.0):
            lip = (F_nu - FT) / L_dir
            ang = (acFT - np.arccos(min(F_nu, 1.0))) / C_FS
            assert ang >= lip - 1e-15


def test_same_crossing_fewer_evals(setup):
    """Both step rules resolve the same first crossing; the angular rule
    needs no more evaluations."""
    problem, c, dt, dHs, L, fn = setup
    G = mp.joint_gauge(dHs, dt)
    AG = mp.angular_gauge(dHs, dt)
    for d in (np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)):
        L_dir = G.L_dir(d, FT, problem["dim"])
        lip = mp.directional_margin(
            fn, L, FT, d, margin_tol=1e-8, L_dir=L_dir, return_diagnostics=True
        )
        ang = mp.directional_margin(
            fn,
            L,
            FT,
            d,
            margin_tol=1e-8,
            L_dir=L_dir,
            angular_gauge=AG,
            return_diagnostics=True,
        )
        assert lip.reason_plus == "bracketed"
        assert ang.reason_plus == "bracketed"
        # Brackets overlap and margins agree within tolerance scale.
        assert ang.M <= lip.M_upper + 1e-15
        assert lip.M <= ang.M_upper + 1e-15
        assert abs(ang.M - lip.M) <= 1e-6 * max(lip.M, 1e-12)
        assert ang.n_evals <= lip.n_evals


def test_island_safety_with_angular_rule():
    """The certified-promotion logic must remain island-safe under a
    non-Lipschitz (angular-form) radius rule.  The synthetic rule is
    first self-validated against the true distance to the unsafe set."""
    ft = 0.9

    def fid(mu):
        x = abs(mu)
        if x < 0.1:
            return 0.99 - 0.5 * x
        if x < 0.16:
            return 0.94 - 0.5 * (x - 0.1) * 10.0  # crosses FT at 0.108
        if x < 0.24:
            return 0.64 + 0.5 * (x - 0.16) * 10.0  # recovers at 0.212
        return 0.99

    # Angular-form rule with C chosen so that srf(F(x)) never exceeds
    # the true distance to the unsafe set (0.108, 0.212) in |mu|.
    C = 23.0  # binding: srf(1.0) <= 0.02, the narrow-island clearance
    ac_ft = np.arccos(ft)

    def srf(F):
        return (ac_ft - np.arccos(min(F, 1.0))) / C

    def true_dist(x):
        x = abs(x)
        if x <= 0.108:
            return 0.108 - x
        if x >= 0.212:
            return x - 0.212
        return 0.0

    for x in np.linspace(0.0, 0.5, 2001):
        if fid(x) >= ft and true_dist(x) > 0:
            assert srf(fid(x)) <= true_dist(x) + 1e-12, x

    res = iterative_margin(
        fid, 5.0, ft, mu0=0.0, eta=1e-6, margin_tol=1e-6, safe_radius_fn=srf
    )
    assert res.M <= 0.108 + 1e-6
    assert res.M >= 0.108 - 1e-3
    assert res.M_upper <= 0.2


def test_default_rule_bit_identical(setup):
    """Passing safe_radius_fn = (F-FT)/L explicitly reproduces the
    default path exactly."""
    problem, c, dt, dHs, L, fn = setup
    d = np.array([0.0, 1.0, 0.0])
    L_dir = float(np.dot(np.asarray(L), np.abs(d)))
    a = mp.directional_margin(fn, L, FT, d, margin_tol=1e-8, return_diagnostics=True)
    b = mp.directional_margin(
        fn,
        L,
        FT,
        d,
        margin_tol=1e-8,
        return_diagnostics=True,
        safe_radius_fn=lambda F: (F - FT) / L_dir,
    )
    assert a.M == b.M
    assert a.M_upper == b.M_upper
    assert a.M_minus == b.M_minus and a.M_plus == b.M_plus
    assert a.n_evals == b.n_evals

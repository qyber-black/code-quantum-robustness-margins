# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The static and trajectory gauges differ by one constant.

Proposition "one gauge, two scenarios" of the paper: the static Lipschitz
constant and the Choi speed of a structure are the same sum of centred
Frobenius norms, so ``L_j = sqrt(1 - FT^2) * s_j``, and the joint gauges
inherit it. Every ratio of certified radii therefore loses the factor, which
is why the static gauge-over-polytope gain and the trajectory
joint-over-separable gain are one number rather than two.

The two sides are computed by separate code paths -- structure_constants on
one, the Choi speeds on the other -- so this fails if either drifts, and it
is what licenses the paper to present the coinciding rows of the gauge table
as a check rather than as a repetition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qrobustness import dH_structure, load_controllers, load_problem, multiparam as mp
from qrobustness.timevarying import fs_margin, fs_margin_joint

ROOT = Path(__file__).resolve().parents[2]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"

FT = 0.999
#: The identity is per controller, so a few of the frozen ensemble settle
#: it; the drivers exercise all of them.
N_CONTROLLERS = 6
#: Machine-precision agreement: these are the same arithmetic, not two
#: estimates of one quantity.
TOL = 1e-13


@pytest.fixture(scope="module")
def case_study():
    """A slice of the frozen ensemble with the p=3 uncertainty spec."""
    problem = load_problem(CTRL / "problem9.mat")
    controllers = load_controllers(CTRL / "controllers.csv", 1e-4)[:N_CONTROLLERS]
    specs = [
        [
            ("drift", problem["H0"]),
            ("control", problem["H1"], c["u1"]),
            ("control", problem["H2"], c["u2"]),
        ]
        for c in controllers
    ]
    return problem, controllers, specs


def _structures(problem, c):
    return [
        dH_structure(problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], t)
        for t in ("H0", "H1", "H2")
    ]


def test_lipschitz_constant_is_the_choi_speed_times_the_threshold_factor(case_study):
    """Part (i): L_j = sqrt(1 - FT^2) s_j, per structure and controller."""
    problem, controllers, specs = case_study
    kappa = np.sqrt(1.0 - FT**2)
    for c, spec in zip(controllers, specs, strict=True):
        dt = c["tf"] / c["tau"]
        _, L = mp.structure_constants(spec, dt, c["tau"], FT, problem["dim"])
        s = [fs_margin(dH, dt, c["fid"], FT).speed for dH in _structures(problem, c)]
        for Lj, sj in zip(np.asarray(L), s, strict=True):
            assert Lj == pytest.approx(kappa * sj, rel=TOL)


def test_the_joint_gauges_differ_by_the_same_factor(case_study):
    """Part (ii): B_T C_joint(x) = sqrt(1 - FT^2) l(|x|) on these systems.

    The general statement is an inequality; it is an equality when one sign
    vertex maximises every interval Gram form, which holds here because the
    structures are Frobenius-orthogonal per interval. A generic direction is
    included so the test does not pass on sign symmetry alone.
    """
    problem, controllers, _ = case_study
    kappa = np.sqrt(1.0 - FT**2)
    ds = [
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
        np.array([1.0, -1.0, 1.0]) / np.sqrt(3.0),
        np.array([2.0, 1.0, -3.0]) / np.linalg.norm([2.0, 1.0, -3.0]),
    ]
    for c in controllers:
        dt = c["tf"] / c["tau"]
        dHs = _structures(problem, c)
        G = mp.joint_gauge(dHs, dt)
        ell, _ = fs_margin_joint(dHs, dt, c["fid"], FT)
        surplus = c["fid"] - FT
        for d in ds:
            gauge = surplus / G.boundary_radius(d, surplus, FT, problem["dim"])
            assert gauge == pytest.approx(kappa * ell(np.abs(d)), rel=TOL)


def test_the_two_reported_gains_are_one_number(case_study):
    """Part (iii): the factor cancels, so static and trajectory gains agree."""
    problem, controllers, specs = case_study
    diagonals = mp.diagonal_directions(3)
    for c, spec in zip(controllers, specs, strict=True):
        dt = c["tf"] / c["tau"]
        dHs = _structures(problem, c)
        G = mp.joint_gauge(dHs, dt)
        _, L = mp.structure_constants(spec, dt, c["tau"], FT, problem["dim"])
        L = np.asarray(L)
        surplus = c["fid"] - FT
        static = [
            G.boundary_radius(d, surplus, FT, problem["dim"])
            / (surplus / float(np.abs(d) @ L))
            for d in diagonals
        ]
        ell, budget = fs_margin_joint(dHs, dt, c["fid"], FT)
        s = sum(fs_margin(dH, dt, c["fid"], FT).speed for dH in dHs)
        trajectory = (budget / ell(np.ones(3))) / (budget / s)
        # Equal along every diagonal, not only on average.
        for g in static:
            assert g == pytest.approx(trajectory, rel=TOL)

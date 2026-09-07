# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests of the Berberich et al. (arXiv:2509.08481 Thm 2.1) implied
margins."""

from pathlib import Path

import numpy as np
import pytest

from qrobustness import (
    dH_structure,
    load_controllers,
    load_problem,
    perturbed_hamiltonians,
)
from qrobustness import berberich
from qrobustness.core import gate_fidelity, propagator

ROOT = Path(__file__).resolve().parents[2]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"

SZ = np.diag([1.0, -1.0]).astype(complex)
SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

FT = 0.999


def test_single_gate_commuting_analytic():
    """One interval, perturbation commuting with the drift: the depth
    term vanishes (tau = 1) and the bound reduces to the exact rotation
    angle, so the implied margin matches the analytic crossing."""
    dt = 1.0
    H_list = [0.3 * SZ]
    dH_list = [SZ]  # commutes: perturbed gate exp(-i(0.3+m)Z)
    mb = berberich.margin(H_list, dH_list, dt, FT, uncertainty="independent")
    # tau=1: a = 0, b = dt*||Z|| = 1, m = sin(arccos FT) exactly.
    assert mb.m == pytest.approx(np.sin(np.arccos(FT)), rel=1e-12)
    # The certified perturbation indeed keeps F >= FT: F = cos(m*t)
    # for the commuting rotation (trace fidelity of exp(-imZ)).
    U0 = propagator(H_list, dt)
    U = propagator([H_list[0] + mb.m * dH_list[0]], dt)
    assert gate_fidelity(U, U0) >= FT - 1e-12


def test_classes_and_safety_on_ensemble():
    """Both uncertainty classes are positive and ordered on real data, and
    a constant perturbation at the systematic margin still meets FT.

    The ordering is the substance: the constant class may credit coherent
    averaging, so its gamma must come out below the independent one.
    """
    problem = load_problem(CTRL / "problem9.mat")
    c = load_controllers(CTRL / "controllers.csv", 1e-4)[0]
    dt = c["tf"] / c["tau"]
    for tag in ("H0", "H1", "H2"):
        H_list = perturbed_hamiltonians(
            problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag, 0.0
        )
        dH = dH_structure(
            problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag
        )
        mb_i = berberich.margin(
            H_list, dH, dt, FT, nominal_error=c["error"], uncertainty="independent"
        )
        mb_s = berberich.margin(
            H_list, dH, dt, FT, nominal_error=c["error"], uncertainty="systematic"
        )
        assert mb_i.m > 0 and mb_s.m > 0
        assert mb_s.magnus_ok
        # Constant class credits coherent averaging: gamma_sys < gamma_ind.
        assert mb_s.gamma < mb_i.gamma
        # Safety spot check: constant perturbation at the systematic
        # margin keeps the achieved gate within the certified deviation
        # of the nominal product (F >= FT after absorption implies
        # F(U0, U(m)) >= cos(arcsin(s_T)) at least; check threshold
        # directly against the target).
        U = propagator(
            [
                np.asarray(H_list[k]) + mb_s.m * np.asarray(dH[k])
                for k in range(c["tau"])
            ],
            dt,
        )
        assert gate_fidelity(U, problem["Uf"]) >= FT - 1e-12


def test_vacuous_when_budget_exhausted():
    """A nominal error past the threshold leaves no budget: the bound must
    report itself vacuous and return zero, not a small positive margin."""
    mb = berberich.margin([0.3 * SZ], [SX], 1.0, FT, nominal_error=0.01)
    assert mb.vacuous and mb.m == 0.0

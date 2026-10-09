# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Checks of the Berberich et al. (arXiv:2509.08481 Thm 2.1) implied
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
FT_HAND = 0.99


def test_single_gate_commuting_analytic():
    """A single interval, with the perturbation commuting with the drift: the depth
    term vanishes (tau = 1) and the bound reduces to the exact rotation
    angle, so the implied margin matches the analytic crossing.
    """
    dt = 1.0
    H_list = [0.3 * SZ]
    dH_list = [SZ]  # commutes: perturbed gate exp(-i(0.3+m)Z)
    mb = berberich.margin(H_list, dH_list, dt, FT, uncertainty="independent")
    # tau=1: a = 0, b = dt*||Z|| = 1, m = sin(arccos FT) exactly.
    assert mb.m == pytest.approx(np.sin(np.arccos(FT)), rel=1e-12)
    # That certified perturbation does keep F >= FT: F = cos(m*t)
    # for that commuting rotation (trace fidelity of exp(-imZ)).
    U0 = propagator(H_list, dt)
    U = propagator([H_list[0] + mb.m * dH_list[0]], dt)
    assert gate_fidelity(U, U0) >= FT - 1e-12


def test_classes_and_safety_on_ensemble():
    """Both uncertainty classes stay positive and ordered on real data, and
    a constant perturbation at the systematic margin still meets FT.

    That ordering is the substance: the constant class may credit coherent
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
        # The constant class credits coherent averaging: gamma_sys < gamma_ind.
        assert mb_s.gamma < mb_i.gamma
        # Safety spot-check: a constant perturbation at the systematic
        # margin holds the achieved gate inside the certified deviation
        # from the nominal product (F >= FT after absorption implies
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


def test_two_interval_hand_formula_both_classes():
    """Two intervals, norms 1 and 2: both classes match the closed form.

    independent uses the mean spectral norm. systematic uses the supplied
    Kosut rate w_avg and does not recompute it.
    """
    from qrobustness.kosut import UncertaintyRates

    dt = 0.5
    H = [0.2 * SZ, 0.4 * SZ]
    dH = [SZ, 2.0 * SZ]
    w = np.array([1.0, 2.0])
    tau = 2
    w_max = 2.0
    budget = np.arccos(FT_HAND)
    s_T = np.sin(budget)
    a_depth = tau * (tau - 1) / 2.0 * (dt * w_max) ** 2
    b_i = tau * dt * float(np.mean(w))
    m_i = 2.0 * s_T / (b_i + np.sqrt(b_i * b_i + 4.0 * a_depth * s_T))
    rates = UncertaintyRates(w_unc=2.0, w_avg=0.3, w_dev=0.1, T=tau * dt)
    a_s = a_depth + dt**2 * float(np.sum(w**2)) / 2.0
    b_s = rates.T * rates.w_avg
    m_s = 2.0 * s_T / (b_s + np.sqrt(b_s * b_s + 4.0 * a_s * s_T))

    got_i = berberich.margin(H, dH, dt, FT_HAND, uncertainty="independent")
    got_s = berberich.margin(H, dH, dt, FT_HAND, uncertainty="systematic", rates=rates)
    assert got_i.m == pytest.approx(m_i, rel=1e-12)
    assert got_s.m == pytest.approx(m_s, rel=1e-12)
    assert got_s.b == pytest.approx(b_s, rel=1e-12)
    assert got_i.magnus_ok == (got_i.m * dt * got_i.w_max < np.pi)
    assert got_s.magnus_ok


def test_nominal_error_out_of_range_is_rejected():
    """An error outside [0, 1] is rejected rather than turned into a NaN margin."""
    with pytest.raises(ValueError, match="nominal_error"):
        berberich.margin([0.3 * SZ], [SZ], 1.0, FT, nominal_error=1.5)


def test_vacuous_when_budget_exhausted():
    """Nominal error past the threshold leaves no budget: the bound has to
    report itself vacuous and return zero, not a small positive margin."""
    mb = berberich.margin([0.3 * SZ], [SX], 1.0, FT, nominal_error=0.01)
    assert mb.vacuous and mb.m == 0.0


def test_systematic_margin_refuses_a_failed_magnus_condition(monkeypatch):
    monkeypatch.setattr("qrobustness.berberich.np.pi", 1e-30)
    H = [np.eye(2, dtype=complex)]
    dH = [np.eye(2, dtype=complex)]
    with pytest.raises(ValueError, match="Magnus"):
        berberich.margin(H, dH, 0.1, 0.99, uncertainty="systematic")

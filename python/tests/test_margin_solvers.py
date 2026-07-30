# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Selectable margin solvers vs paper Algorithm 1."""

from __future__ import annotations

from pathlib import Path

import pytest
from qrobustness import (
    MARGIN_METHODS,
    dH_structure,
    differential_sensitivity,
    iterative_margin,
    lipschitz_constant,
    load_controllers,
    load_problem,
    make_fidelity_fn,
    perturbed_hamiltonians,
    structure_constant,
)

ROOT = Path(__file__).resolve().parents[2]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"


def test_default_matches_algorithm1_explicit():
    FT = 0.99
    fidelity_fn = lambda mu: max(0.0, 1.0 - 0.05 * abs(mu))
    L = 0.05
    a = iterative_margin(fidelity_fn, L, FT, mu0=0.0, eta=1e-8, k_max=1000)
    b = iterative_margin(fidelity_fn, L, FT, mu0=0.0, eta=1e-8, k_max=1000, method="algorithm1")
    assert abs(a.M - b.M) < 1e-14
    assert abs(a.M_minus - b.M_minus) < 1e-14
    assert abs(a.M_plus - b.M_plus) < 1e-14


@pytest.mark.parametrize("method", ["lipschitz_brent", "lipschitz_toms748"])
def test_lipschitz_polish_matches_algorithm1(method):
    FT = 0.99
    fidelity_fn = lambda mu: max(0.0, 1.0 - 0.05 * abs(mu))
    L = 0.05
    base = iterative_margin(fidelity_fn, L, FT, mu0=0.0, eta=1e-8, k_max=1000, method="algorithm1")
    alt = iterative_margin(fidelity_fn, L, FT, mu0=0.0, eta=1e-8, k_max=1000, method=method)
    assert abs(alt.M - base.M) < 1e-6
    assert abs(alt.M_minus - base.M_minus) < 1e-6
    assert abs(alt.M_plus - base.M_plus) < 1e-6


@pytest.mark.parametrize("method", ["lipschitz_brent", "lipschitz_toms748", "doubling"])
def test_large_L_overshoot_methods(method):
    FT = 0.99
    fidelity_fn = lambda mu: max(0.0, 1.0 - 0.05 * abs(mu))
    base = iterative_margin(
        fidelity_fn, 10.0, FT, mu0=0.0, eta=1e-6, k_max=5000, method="algorithm1"
    )
    alt = iterative_margin(fidelity_fn, 10.0, FT, mu0=0.0, eta=1e-6, k_max=5000, method=method)
    assert abs(alt.M - base.M) < 5e-4


def test_doubling_monotone_agrees():
    FT = 0.99
    fidelity_fn = lambda mu: max(0.0, 1.0 - 0.05 * abs(mu))
    L = 0.05
    base = iterative_margin(fidelity_fn, L, FT, mu0=0.0, eta=1e-8, k_max=1000, method="algorithm1")
    dbl = iterative_margin(
        fidelity_fn,
        L,
        FT,
        mu0=0.0,
        eta=1e-8,
        k_max=1000,
        method="doubling",
        return_diagnostics=True,
    )
    assert abs(dbl.M - base.M) < 1e-5
    assert dbl.n_evals is not None and dbl.n_evals > 0


def test_newton_probe_monotone():
    FT = 0.99
    # F = 1 - 0.05*|mu|, zeta = dF/dmu = -0.05 * sign(mu) (at 0 use -0.05 for + side)
    fidelity_fn = lambda mu: max(0.0, 1.0 - 0.05 * abs(mu))

    def zeta_fn(mu: float) -> float:
        if mu > 0:
            return -0.05
        if mu < 0:
            return 0.05
        return -0.05

    L = 0.05
    base = iterative_margin(fidelity_fn, L, FT, mu0=0.0, eta=1e-8, k_max=1000, method="algorithm1")
    newt = iterative_margin(
        fidelity_fn,
        L,
        FT,
        mu0=0.0,
        eta=1e-8,
        k_max=1000,
        method="newton_probe",
        zeta_fn=zeta_fn,
        return_diagnostics=True,
    )
    assert abs(newt.M - base.M) < 1e-4
    assert newt.n_evals is not None


def test_newton_probe_requires_zeta():
    with pytest.raises(ValueError, match="zeta_fn"):
        iterative_margin(
            lambda mu: 1.0 - 0.05 * abs(mu),
            0.05,
            0.99,
            method="newton_probe",
        )


def test_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        iterative_margin(lambda mu: 1.0 - abs(mu), 1.0, 0.5, method="not_a_method")


def test_diagnostics_populated():
    res = iterative_margin(
        lambda mu: max(0.0, 1.0 - 0.05 * abs(mu)),
        0.05,
        0.99,
        eta=1e-8,
        return_diagnostics=True,
    )
    assert res.method == "algorithm1"
    assert isinstance(res.n_evals, int) and res.n_evals >= 2
    assert isinstance(res.n_steps, int) and res.n_steps >= 0


def test_all_methods_listed():
    assert "algorithm1" in MARGIN_METHODS
    assert "doubling" in MARGIN_METHODS


def test_case_study_lipschitz_methods_match_default():
    FT = 0.999
    eta = 1e-6
    problem = load_problem(CTRL / "problem9.mat")
    controllers = load_controllers(CTRL / "controllers.csv", 1e-4)
    c = controllers[0]
    dt = c["tf"] / c["tau"]
    C = structure_constant("drift", problem["H0"], dt, c["tau"])
    L = lipschitz_constant(FT, problem["dim"], C)
    fid_fn = make_fidelity_fn(
        problem["H0"],
        problem["H1"],
        problem["H2"],
        c["u1"],
        c["u2"],
        problem["Uf"],
        dt,
        "H0",
    )
    base = iterative_margin(fid_fn, L, FT, mu0=0.0, eta=eta, method="algorithm1")
    for method in ("lipschitz_brent", "lipschitz_toms748"):
        alt = iterative_margin(fid_fn, L, FT, mu0=0.0, eta=eta, method=method)
        assert abs(alt.M - base.M) < 1e-8
        assert abs(alt.M_minus - base.M_minus) < 1e-8
        assert abs(alt.M_plus - base.M_plus) < 1e-8


def test_case_study_doubling_and_newton_smoke():
    FT = 0.999
    eta = 1e-6
    problem = load_problem(CTRL / "problem9.mat")
    controllers = load_controllers(CTRL / "controllers.csv", 1e-4)
    c = controllers[0]
    dt = c["tf"] / c["tau"]
    C = structure_constant("drift", problem["H0"], dt, c["tau"])
    L = lipschitz_constant(FT, problem["dim"], C)
    fid_fn = make_fidelity_fn(
        problem["H0"],
        problem["H1"],
        problem["H2"],
        c["u1"],
        c["u2"],
        problem["Uf"],
        dt,
        "H0",
    )

    def zeta_fn(mu: float) -> float:
        H_list = perturbed_hamiltonians(
            problem["H0"],
            problem["H1"],
            problem["H2"],
            c["u1"],
            c["u2"],
            "H0",
            mu,
        )
        dH = dH_structure(problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], "H0")
        return differential_sensitivity(H_list, dH, dt, problem["Uf"], n_quad=16)

    base = iterative_margin(fid_fn, L, FT, mu0=0.0, eta=eta, method="algorithm1")
    dbl = iterative_margin(
        fid_fn, L, FT, mu0=0.0, eta=eta, method="doubling", return_diagnostics=True
    )
    newt = iterative_margin(
        fid_fn,
        L,
        FT,
        mu0=0.0,
        eta=eta,
        method="newton_probe",
        zeta_fn=zeta_fn,
        return_diagnostics=True,
    )
    # Endpoint certificate: F at reported margins should stay above FT.
    assert fid_fn(-dbl.M) >= FT - 1e-10
    assert fid_fn(dbl.M) >= FT - 1e-10
    assert fid_fn(-newt.M) >= FT - 1e-10
    assert fid_fn(newt.M) >= FT - 1e-10
    # On this case study, doubling should not be wildly off Algorithm 1.
    assert abs(dbl.M - base.M) / max(base.M, 1e-30) < 0.05
    assert dbl.n_evals is not None and newt.n_evals is not None

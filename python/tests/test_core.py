# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path

import numpy as np
import pytest
from qrobustness import (
    dH_structure,
    differential_sensitivity,
    gate_fidelity,
    iterative_margin,
    lipschitz_constant,
    load_controllers,
    load_problem,
    make_fidelity_fn,
    perturbed_hamiltonians,
    propagator,
    structure_constant,
)

ROOT = Path(__file__).resolve().parents[2]


def test_propagator_fidelity():
    N = 2
    U = propagator([np.zeros((N, N)), np.zeros((N, N))], 0.1)
    assert np.linalg.norm(U - np.eye(N)) < 1e-12
    assert abs(gate_fidelity(U, np.eye(N)) - 1.0) < 1e-12

    Hz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    dt = 0.3
    U = propagator([Hz], dt)
    Uref = expm_ref(-1j * dt * Hz)
    assert np.linalg.norm(U - Uref) < 1e-12


def expm_ref(A):
    from scipy.linalg import expm

    return expm(A)


def test_lipschitz_structure():
    H0 = np.array([[0, 1], [1, 0]], dtype=complex)
    H1 = np.array([[1, 0], [0, -1]], dtype=complex)
    dt = 0.25
    tau = 4
    tf = tau * dt
    controls = np.array([0.5, -0.2, 0.1, 0.3])

    C0 = structure_constant("drift", H0, dt, tau)
    assert abs(C0 - tf * np.linalg.norm(H0, "fro")) < 1e-14

    C1 = structure_constant("control", H1, dt, tau, controls)
    assert abs(C1 - dt * np.linalg.norm(controls, 1) * np.linalg.norm(H1, "fro")) < 1e-14

    FT = 0.999
    L = lipschitz_constant(FT, 2, C0)
    B = np.sqrt((1 - FT**2) / 2)
    assert abs(L - B * C0) < 1e-14


def test_perfect_fidelity_zeta():
    H = np.array([[0, 1], [1, 0]], dtype=complex)
    dt = 0.2
    from scipy.linalg import expm

    Uf = expm(-1j * dt * H)
    zeta = differential_sensitivity([H], [H], dt, Uf, n_quad=48)
    assert abs(zeta) < 1e-9


def test_iterative_margin_synthetic():
    FT = 0.99
    fidelity_fn = lambda mu: max(0.0, 1.0 - 0.05 * abs(mu))
    L = 0.05
    res = iterative_margin(fidelity_fn, L, FT, mu0=0.0, eta=1e-8, k_max=1000)
    assert abs(res.M_minus - 0.2) < 1e-5
    assert abs(res.M_plus - 0.2) < 1e-5
    assert abs(res.M - 0.2) < 1e-5
    assert res.converged_minus and res.converged_plus

    res_b = iterative_margin(
        fidelity_fn, L, FT, mu0=0.0, eta=1e-8, omega=(-0.05, np.inf), k_max=1000
    )
    assert abs(res_b.M_minus - 0.05) < 1e-8

    res_big = iterative_margin(fidelity_fn, 10.0, FT, mu0=0.0, eta=1e-6, k_max=5000)
    assert abs(res_big.M - 0.2) < 5e-4


def test_threshold_error():
    with pytest.raises(ValueError):
        iterative_margin(lambda mu: 0.95, 1.0, 0.99, mu0=0.0)


def test_load_case_study_smoke():
    CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
    problem = load_problem(CTRL / "problem9.mat")
    assert problem["dim"] == 8
    controllers = load_controllers(CTRL / "controllers.csv", 1e-4)
    assert len(controllers) == 61

    c = controllers[0]
    dt = c["tf"] / c["tau"]
    H_list = perturbed_hamiltonians(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], "H0", 0.0
    )
    U = propagator(H_list, dt)
    F = gate_fidelity(U, problem["Uf"])
    assert abs(F - c["fid"]) < 1e-4

    dH = dH_structure(problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], "H0")
    zeta = differential_sensitivity(H_list, dH, dt, problem["Uf"], n_quad=24)
    assert np.isfinite(zeta)


def test_iterative_margin_case_study_certificate():
    """Algorithm 1 on real case-study fidelity: F(+/-M) >= FT and M = min(M+/-)."""
    CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
    FT = 0.999
    eta = 1e-6
    problem = load_problem(CTRL / "problem9.mat")
    controllers = load_controllers(CTRL / "controllers.csv", 1e-4)
    c = controllers[0]  # controller 1 in 1-based paper indexing
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
    res = iterative_margin(fid_fn, L, FT, mu0=0.0, eta=eta)
    assert res.M == min(res.M_minus, res.M_plus)
    assert res.converged_minus and res.converged_plus
    assert fid_fn(-res.M) >= FT - 1e-12
    assert fid_fn(res.M) >= FT - 1e-12
    # Match published MATLAB table for ctrl1 H0
    table_csv = ROOT / "results/lipschitz-margin-matlab/margins_table_0.999.csv"
    if table_csv.is_file():
        import csv

        with table_csv.open(newline="") as f:
            row = next(csv.DictReader(f))
        assert abs(res.M - float(row["M_H0"])) < 1e-10
        assert abs(res.M_minus - float(row["Mm_H0"])) < 1e-10
        assert abs(res.M_plus - float(row["Mp_H0"])) < 1e-10

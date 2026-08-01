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


def test_certified_bracket_does_not_jump_safe_islands():
    """With a nonmonotone fidelity (an unsafe dip followed by a safe
    island), the certified lower margin must stop at the nominal
    component's boundary rather than promoting pointwise-safe island
    samples (round-2 review, Appendix A semantics)."""
    import numpy as np
    from qrobustness import iterative_margin

    FT = 0.9
    # F: safe plateau to |mu| ~ 0.1, dip below FT on [0.12, 0.2],
    # safe island beyond. Lipschitz constant consistent with slopes.
    def fid(mu):
        x = abs(mu)
        if x < 0.1:
            return 0.99 - 0.5 * x
        if x < 0.16:
            return 0.94 - 0.5 * (x - 0.1) * 10.0   # crosses FT at 0.108
        if x < 0.24:
            return 0.64 + 0.5 * (x - 0.16) * 10.0  # recovers, crosses up at 0.212
        return 0.99

    L = 5.0  # valid Lipschitz constant for the profile above
    res = iterative_margin(fid, L, FT, mu0=0.0, eta=1e-6, margin_tol=1e-6)
    # True first boundary at |mu| = 0.108: certified M must not exceed
    # it, and the upper witness must lie beyond it but well before the
    # island could be mistaken for the component (upper < 0.2).
    assert res.M <= 0.108 + 1e-6
    assert res.M >= 0.108 - 1e-3
    assert res.M_upper <= 0.2


def test_certified_bracket_partial_when_island_masks_first_crossing():
    """Discriminating island case: Algorithm 1 stops early on an
    eta-band plateau, and the outward geometric probe's step grows
    large enough to leap a narrow unsafe dip straight into a wide safe
    island.  Uncertified promotion would report the island's far
    boundary (~0.3) as the margin; the certified-promotion rule must
    keep M at the nominal component and report a rigorous (if wide)
    'partial' bracket that still contains the true first crossing."""
    import numpy as np
    from qrobustness import iterative_margin

    FT = 0.9
    # |mu| profile (L = 300 valid throughout):
    #   descent slope 2 to an eta-band plateau F = FT + 5e-7 from ~0.045,
    #   narrow dip on [0.1, 0.1005] (slope 100) crossing FT at 0.1+5e-9,
    #   recovery slope 280 into a safe island at 0.99 up to 0.3,
    #   final descent slope 300 crossing FT at ~0.3003.
    def fid(mu):
        x = abs(mu)
        if x < 0.1:
            return max(FT + 5e-7, 0.99 - 2.0 * x)
        if x < 0.1005:
            return FT + 5e-7 - 100.0 * (x - 0.1)
        if x < 0.3:
            return min(0.99, FT + 5e-7 - 0.05 + 280.0 * (x - 0.1005))
        return max(0.0, 0.99 - 300.0 * (x - 0.3))

    res = iterative_margin(fid, 300.0, FT, mu0=0.0, eta=1e-6, margin_tol=1e-6)
    # True first boundary at |mu| ~= 0.1: the certified margin must not
    # be inflated to the island's far edge (~0.3003) ...
    assert res.M <= 0.1 + 1e-6
    assert res.M >= 0.04
    # ... the bracket must still contain the true first crossing ...
    assert np.isfinite(res.M_upper)
    assert res.M_upper >= 0.1
    # ... and the unreached tolerance must be reported, not hidden.
    assert res.reason_minus == "partial"
    assert res.reason_plus == "partial"

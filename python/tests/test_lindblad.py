# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Open-system layer: reductions, CPTP structure, Frechet derivatives,
diamond norms against analytic values, and the Lipschitz certificate."""

from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from qrobustness import gate_fidelity, load_controllers, load_problem, propagator
from qrobustness import lindblad as lb

cvxpy = pytest.importorskip("cvxpy")

ROOT = Path(__file__).resolve().parents[2]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


@pytest.fixture(scope="module")
def case():
    problem = load_problem(CTRL / "problem9.mat")
    controllers = load_controllers(CTRL / "controllers.csv")
    return problem, controllers


def _herm(rng, n):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return 0.5 * (A + A.conj().T)


# --------------------------------------------------------------------------
# Reductions to the closed system
# --------------------------------------------------------------------------


def test_gamma_zero_reduces_to_closed_system(case):
    """At gamma = 0, F_pro = (closed gate fidelity)^2 to roundoff."""
    problem, controllers = case
    c = controllers[0]
    dt = c["tf"] / c["tau"]
    H_list = [
        problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
        for k in range(c["tau"])
    ]
    G_list = [lb.generator(H) for H in H_list]
    S = lb.channel(G_list, dt)
    F_pro = lb.process_fidelity(S, problem["Uf"])
    F_closed = gate_fidelity(propagator(H_list, dt), problem["Uf"])
    assert F_pro == pytest.approx(F_closed**2, rel=1e-10)


def test_unitary_superop_consistency():
    rng = np.random.default_rng(0)
    H = _herm(rng, 3)
    U = expm(-1j * 0.7 * H)
    S = lb.channel([lb.generator(H)], 0.7)
    assert np.linalg.norm(S - lb.unitary_superop(U)) < 1e-10


# --------------------------------------------------------------------------
# CPTP structure
# --------------------------------------------------------------------------


def test_channel_is_cptp():
    """Choi PSD and trace preservation for a dissipative channel."""
    rng = np.random.default_rng(1)
    H = _herm(rng, 2)
    G = lb.generator(H, [SZ, SX], [0.3, 0.1])
    S = lb.channel([G], 1.3)
    J = lb.choi_matrix(S)
    evals = np.linalg.eigvalsh(0.5 * (J + J.conj().T))
    assert evals.min() > -1e-10, "Choi not PSD"
    # Trace preservation: applying S to a basis preserves every trace.
    N = 2
    for i in range(N):
        for j in range(N):
            E = np.zeros((N, N), dtype=complex)
            E[i, j] = 1.0
            out = (S @ E.reshape(-1, order="F")).reshape((N, N), order="F")
            assert np.trace(out) == pytest.approx(np.trace(E), abs=1e-10)


def test_dissipator_annihilates_identity():
    """D[V] applied to the identity direction preserves trace (unital check)."""
    D = lb.dissipator(SZ)
    rho = np.eye(2, dtype=complex) / 2
    out = (D @ rho.reshape(-1, order="F")).reshape((2, 2), order="F")
    assert abs(np.trace(out)) < 1e-12


# --------------------------------------------------------------------------
# Frechet derivative
# --------------------------------------------------------------------------


def test_frechet_matches_fd():
    rng = np.random.default_rng(2)
    H = _herm(rng, 2)
    G = lb.generator(H, [SZ], [0.2])
    E = lb.dissipator(SZ)
    dt = 0.9
    dS = lb.frechet_derivative(G, E, dt)
    eps = 1e-6
    fd = (expm(dt * (G + eps * E)) - expm(dt * (G - eps * E))) / (2 * eps)
    assert np.linalg.norm(dS - fd) / np.linalg.norm(dS) < 1e-8


def test_frechet_defective_generator():
    """The block method needs no diagonalisability: use a Jordan block."""
    G = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)  # defective
    E = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    dt = 0.5
    dS = lb.frechet_derivative(G, E, dt)
    eps = 1e-7
    fd = (expm(dt * (G + eps * E)) - expm(dt * (G - eps * E))) / (2 * eps)
    assert np.linalg.norm(dS - fd) / max(np.linalg.norm(dS), 1e-300) < 1e-6


# --------------------------------------------------------------------------
# Diamond norm
# --------------------------------------------------------------------------


def test_diamond_norm_analytic_cases():
    I2 = np.eye(2, dtype=complex)
    assert lb.diamond_norm(lb.unitary_superop(I2)).value == pytest.approx(1.0, abs=1e-5)

    theta = 0.3
    U = expm(-1j * theta * SX)
    D = lb.unitary_superop(U) - lb.unitary_superop(I2)
    # Eigenphases +/- theta: dnorm = 2 sin(theta).
    assert lb.diamond_norm(D).value == pytest.approx(2 * np.sin(theta), abs=1e-5)

    p = 0.37
    tr_row = np.eye(2).reshape(-1, order="F")
    Sdep = (1 - p) * np.eye(4, dtype=complex) + p * np.outer(
        np.eye(2).reshape(-1, order="F") / 2, tr_row
    )
    assert lb.diamond_norm(Sdep).value == pytest.approx(1.0, abs=1e-4)
    # Qubit depolarising distance to identity: p (1 + 1/d) = 3p/2.
    assert lb.diamond_norm(Sdep - np.eye(4)).value == pytest.approx(1.5 * p, abs=1e-4)


def test_diamond_norm_scaling_and_conservatism():
    U = expm(-1j * 0.4 * SZ)
    S = lb.unitary_superop(U)
    a = lb.diamond_norm(2.5 * S).value
    assert a == pytest.approx(2.5, abs=1e-4)
    d = lb.diamond_norm(S)
    assert d.value >= d.raw  # gap added conservatively


# --------------------------------------------------------------------------
# Lipschitz certificate on the case study (small subsystem for speed)
# --------------------------------------------------------------------------


def test_open_lipschitz_certificate_sampled():
    """|F_pro(gamma_a) - F_pro(gamma_b)| <= L |gamma_a - gamma_b| sampled.

    Two-qubit toy: a fixed entangling Hamiltonian with single-qubit
    dephasing at uncertain rate gamma.
    """
    rng = np.random.default_rng(3)
    N = 4
    H = _herm(rng, N)
    V = np.kron(SZ, np.eye(2))
    tau, dt = 4, 0.6
    t_f = tau * dt
    Uf = expm(-1j * t_f * H)

    D = lb.dissipator(V)
    L = 0.5 * t_f * lb.diamond_norm(D).value

    def F_pro(gamma):
        G = lb.generator(H, [V], [float(gamma)])
        return lb.process_fidelity(lb.channel([G] * tau, dt), Uf)

    gammas = rng.uniform(0.0, 0.05, size=12)
    for ga in gammas[:6]:
        for gb in gammas[6:]:
            lhs = abs(F_pro(ga) - F_pro(gb))
            rhs = L * abs(ga - gb)
            assert lhs <= rhs * (1 + 1e-9), (
                f"certificate violated: {lhs:.3e} > {rhs:.3e} at ({ga:.3g},{gb:.3g})"
            )


def test_open_margin_certified_and_bracketed():
    """open_margin certifies gamma below the margin and brackets it."""
    rng = np.random.default_rng(4)
    N = 4
    H = _herm(rng, N)
    V = np.kron(SZ, np.eye(2))
    tau, dt = 4, 0.6
    t_f = tau * dt
    Uf = expm(-1j * t_f * H)
    D = lb.dissipator(V)
    L = 0.5 * t_f * lb.diamond_norm(D).value

    def F_pro(gamma):
        G = lb.generator(H, [V], [float(gamma)])
        return lb.process_fidelity(lb.channel([G] * tau, dt), Uf)

    F0 = F_pro(0.0)
    FT_pro = F0 - 5e-3
    res = lb.open_margin(F_pro, L, FT_pro, margin_tol=1e-6)
    # One-sided domain: the negative direction stops at the gamma >= 0 boundary.
    assert res.reason_minus in ("boundary", "unknown") or res.M_minus == 0.0
    assert F_pro(res.M_plus) >= FT_pro - 1e-12
    assert res.M_upper_plus >= res.M_plus
    if res.reason_plus == "bracketed":
        assert F_pro(res.M_upper_plus * (1 + 1e-6)) < FT_pro + 1e-9


def test_diamond_norm_certified_upper_bound():
    """The verified value is a Rump-repaired primal objective with
    upward-bounded evaluation, so it rigorously upper-bounds the true
    diamond norm on analytic cases and stays tight (<= 1e-5 relative)."""
    import numpy as np
    from qrobustness import lindblad as lb

    I4 = lb.unitary_superop(np.eye(2))
    dn = lb.diamond_norm(I4)
    assert dn.value_certified >= 1.0 - 1e-12
    assert dn.value == dn.value_certified
    assert dn.feas_shift >= 0.0
    th = 0.25
    U = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=complex)
    dn2 = lb.diamond_norm(lb.unitary_superop(U) - I4)
    assert dn2.value_certified >= 2 * np.sin(th) - 1e-12
    assert dn2.value_certified <= 2 * np.sin(th) * (1 + 1e-5)
    # depolarising difference: dnorm = 3p/2
    p_ = 0.2
    dep = (1 - p_) * np.eye(4) + p_ * 0.5 * np.outer(
        np.eye(2).reshape(-1), np.eye(2).reshape(-1)
    )
    dn3 = lb.diamond_norm(dep - np.eye(4))
    assert 1.5 * p_ - 1e-12 <= dn3.value_certified <= 1.5 * p_ * (1 + 1e-5)


def test_common_rate_diamond_norms_are_2n():
    """Lemma (common-rate diamond norms): ||sum_q D[V_q]||_dia = 2n for
    local Pauli-Z and local sigma_- families."""
    import numpy as np
    from qrobustness import lindblad as lb

    Z = np.diag([1.0, -1.0]).astype(complex)
    SM = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def local(op, q, n):
        m = [I2] * n
        m[q] = op
        M = m[0]
        for x in m[1:]:
            M = np.kron(M, x)
        return M

    for n in (1, 2):
        for op in (Z, SM):
            G = sum(lb.dissipator(local(op, q, n)) for q in range(n))
            dn = lb.diamond_norm(G)
            assert dn.value_certified == pytest.approx(2 * n, rel=1e-5)
            assert dn.value_certified >= 2 * n - 1e-9

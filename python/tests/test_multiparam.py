# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-parameter and time-varying margins.

Reduction requirements: p = 1 must reproduce the scalar iteration exactly,
and axis directions must reproduce the paper-1 per-parameter margins.
Certification requirements: sampled points of every certified object
(polytope, union, uniform time-varying radius) must satisfy F >= F_T.
"""

from pathlib import Path

import numpy as np
import pytest

from qrobustness import (
    dH_structure,
    iterative_margin,
    lipschitz_constant,
    load_controllers,
    load_problem,
    make_fidelity_fn,
    structure_constant,
)
from qrobustness import multiparam as mp
from qrobustness import timevarying as tv

ROOT = Path(__file__).resolve().parents[2]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
FT = 0.999
STRUCTURES = ("H0", "H1", "H2")


@pytest.fixture(scope="module")
def case():
    problem = load_problem(CTRL / "problem9.mat")
    controllers = load_controllers(CTRL / "controllers.csv")
    return problem, controllers


def _setup(problem, c):
    dt = c["tf"] / c["tau"]
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
        STRUCTURES,
    )
    return dt, C, L, fn


def _scalar(problem, c, structure):
    dt = c["tf"] / c["tau"]
    if structure == "H0":
        C = structure_constant("drift", problem["H0"], dt, c["tau"])
    else:
        ctl = c["u1"] if structure == "H1" else c["u2"]
        Hm = problem[structure]
        C = structure_constant("control", Hm, dt, c["tau"], ctl)
    L = lipschitz_constant(FT, problem["dim"], C)
    fn = make_fidelity_fn(
        problem["H0"],
        problem["H1"],
        problem["H2"],
        c["u1"],
        c["u2"],
        problem["Uf"],
        dt,
        structure,
    )
    return L, fn


# --------------------------------------------------------------------------
# Reductions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("j,structure", [(0, "H0"), (1, "H1"), (2, "H2")])
def test_axis_direction_reproduces_scalar_margin(case, j, structure):
    """Along e_j the directional margin equals the paper-1 margin."""
    problem, controllers = case
    c = controllers[0]
    _, _, L, fn = _setup(problem, c)
    L1, scalar_fn = _scalar(problem, c, structure)
    assert L[j] == pytest.approx(L1, rel=1e-14)

    ref = iterative_margin(scalar_fn, L1, FT)
    d = np.zeros(3)
    d[j] = 1.0
    got = mp.directional_margin(fn, L, FT, d)
    assert got.M == pytest.approx(ref.M, rel=1e-12)
    assert got.M_minus == pytest.approx(ref.M_minus, rel=1e-12)
    assert got.M_plus == pytest.approx(ref.M_plus, rel=1e-12)


def test_p1_reduces_to_iterative_margin(case):
    """A single-parameter setup is the scalar problem verbatim, brackets included."""
    problem, controllers = case
    c = controllers[0]
    dt = c["tf"] / c["tau"]
    L1, scalar_fn = _scalar(problem, c, "H1")
    fn1 = mp.make_multiparam_fidelity_fn(
        problem["H0"],
        problem["H1"],
        problem["H2"],
        c["u1"],
        c["u2"],
        problem["Uf"],
        dt,
        ("H1",),
    )
    got = mp.directional_margin(
        fn1, np.array([L1]), FT, np.array([1.0]), margin_tol=1e-10
    )
    ref = iterative_margin(scalar_fn, L1, FT, margin_tol=1e-10)
    assert got.M == pytest.approx(ref.M, rel=1e-12)
    assert got.margin_uncertainty / got.M <= 1e-10 * 1.000001
    assert got.certificate == "segment"


# --------------------------------------------------------------------------
# Certification
# --------------------------------------------------------------------------


def test_polytope_is_certified(case):
    """Sampled boundary and interior points satisfy F >= FT."""
    problem, controllers = case
    c = controllers[0]
    _, _, L, fn = _setup(problem, c)
    F0 = fn(np.zeros(3))
    P = mp.safe_polytope(np.zeros(3), L, F0, FT)

    rng = np.random.default_rng(1)
    for _ in range(25):
        d = rng.normal(size=3)
        for scale in (1.0, 0.5):
            mu = P.boundary_point(d) * scale
            assert fn(mu) >= FT - 1e-12, f"violation at {mu}"


def test_polytope_geometry(case):
    """Norm-ball inradii are consistent with the polytope inequality."""
    problem, controllers = case
    c = controllers[0]
    _, _, L, fn = _setup(problem, c)
    F0 = fn(np.zeros(3))
    P = mp.safe_polytope(np.zeros(3), L, F0, FT)

    h = P.inradius_linf
    for corner in mp.diagonal_directions(3) * np.sqrt(3) * h:
        assert np.dot(P.L, np.abs(corner)) <= P.surplus * (1 + 1e-12)
    r2 = P.inradius_l2
    worst = P.L / np.linalg.norm(P.L)
    assert np.dot(P.L, np.abs(worst)) * r2 <= P.surplus * (1 + 1e-12)
    assert np.all(P.axis_radii == pytest.approx(P.surplus / P.L))


def test_safe_union_certified(case):
    """Union of polytopes from safe ray points is certified on samples."""
    problem, controllers = case
    c = controllers[0]
    _, _, L, fn = _setup(problem, c)

    union = mp.SafeUnion(L=L)
    union.add(np.zeros(3), fn(np.zeros(3)), FT, note="centre")
    d = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    res = mp.directional_margin(fn, L, FT, d)
    for s in (0.3 * res.M_plus, 0.7 * res.M_plus):
        mu = s * d
        F = fn(mu)
        if F > FT:
            union.add(mu, F, FT, note=f"ray s={s:.3g}")
    assert len(union.polytopes) >= 2

    rng = np.random.default_rng(2)
    n_checked = 0
    for _ in range(30):
        P = union.polytopes[rng.integers(len(union.polytopes))]
        mu = P.boundary_point(rng.normal(size=3)) * rng.uniform(0.0, 1.0)
        if union.contains(mu):
            assert fn(mu) >= FT - 1e-12
            n_checked += 1
    assert n_checked > 0


def test_direction_designs(case):
    """The three direction sets have the shapes and normalisations the
    directional certificates assume: 2p axes, 2^p unit diagonals, and n unit
    sphere samples."""
    p = 3
    A = mp.axis_directions(p)
    assert A.shape == (2 * p, p)
    D = mp.diagonal_directions(p)
    assert D.shape == (2**p, p)
    assert np.allclose(np.linalg.norm(D, axis=1), 1.0)
    S = mp.sphere_directions(p, 10, seed=0)
    assert S.shape == (10, p)
    assert np.allclose(np.linalg.norm(S, axis=1), 1.0)


# --------------------------------------------------------------------------
# Time-varying
# --------------------------------------------------------------------------


def test_uniform_margin_value_and_ordering(case):
    """r0 = (F-FT)/L and r0 <= iterated margin."""
    problem, controllers = case
    c = controllers[0]
    L1, scalar_fn = _scalar(problem, c, "H1")
    F0 = scalar_fn(0.0)
    r0 = tv.uniform_margin(L1, F0, FT)
    assert r0 == pytest.approx((F0 - FT) / L1, rel=1e-14)
    assert r0 <= iterative_margin(scalar_fn, L1, FT).M


def test_tv_gradient_matches_fd(case):
    """Exact trajectory gradient against central differences."""
    problem, controllers = case
    c = controllers[0]
    dt = c["tf"] / c["tau"]
    tau = c["tau"]
    H_list = [
        problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
        for k in range(tau)
    ]
    Hhat = dH_structure(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], "H1"
    )
    rng = np.random.default_rng(3)
    delta = rng.uniform(-2e-3, 2e-3, size=tau)
    F, g = tv.tv_fidelity_and_gradient(H_list, Hhat, delta, dt, problem["Uf"])
    h = 1e-7
    for k in (0, tau // 2, tau - 1):
        dp = delta.copy()
        dp[k] += h
        dm = delta.copy()
        dm[k] -= h
        Fp, _ = tv.tv_fidelity_and_gradient(H_list, Hhat, dp, dt, problem["Uf"])
        Fm, _ = tv.tv_fidelity_and_gradient(H_list, Hhat, dm, dt, problem["Uf"])
        fd = (Fp - Fm) / (2 * h)
        assert g[k] == pytest.approx(fd, rel=5e-4, abs=1e-10)


def test_certificate_holds_against_adversary(case):
    """Below r0 the adversary must not violate the threshold (Theorem tv)."""
    problem, controllers = case
    c = controllers[0]
    dt = c["tf"] / c["tau"]
    tau = c["tau"]
    L1, scalar_fn = _scalar(problem, c, "H1")
    H_list = [
        problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
        for k in range(tau)
    ]
    Hhat = dH_structure(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], "H1"
    )
    r0 = tv.uniform_margin(L1, scalar_fn(0.0), FT)

    F_min, _ = tv.adversarial_fidelity(
        H_list, Hhat, dt, problem["Uf"], 0.98 * r0, n_starts=3, seed=5
    )
    assert F_min >= FT - 1e-12, "adversary violated the certified radius"


def test_tv_bracket_is_ordered(case):
    """r0 <= m_adv; a found violation certifies the upper bound."""
    problem, controllers = case
    c = controllers[0]
    dt = c["tf"] / c["tau"]
    tau = c["tau"]
    L1, scalar_fn = _scalar(problem, c, "H1")
    H_list = [
        problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
        for k in range(tau)
    ]
    Hhat = dH_structure(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], "H1"
    )
    F0 = scalar_fn(0.0)
    r0 = tv.uniform_margin(L1, F0, FT)
    M = iterative_margin(scalar_fn, L1, FT).M

    br = tv.adversarial_upper_bound(
        H_list, Hhat, dt, problem["Uf"], FT, r0, 2.0 * M, rel_tol=0.1, seed=7
    )
    assert br.r0 == pytest.approx(r0)
    assert br.r0 <= br.m_adv * (1 + 1e-12)
    # M_tv <= M_const: if the adversary found a violation, it must be at or
    # below a budget where the constant perturbation is already unsafe or the
    # trajectory exploited time variation; either way m_adv <= 2 M by search.
    assert br.m_adv <= 2.0 * M * (1 + 1e-12)


def test_joint_gauge_dominates_cross_polytope(case):
    """C_joint(x) <= sum_j C_j |x_j|: the gauge region contains the
    weighted cross-polytope, with equality on the axes."""
    import numpy as np
    from qrobustness import multiparam as mp

    problem, controllers = case
    c = controllers[0]
    dt = c["tf"] / c["tau"]
    from qrobustness import dH_structure

    dHs = [
        dH_structure(problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag)
        for tag in ("H0", "H1", "H2")
    ]
    G = mp.joint_gauge(dHs, dt)
    C = [dt * sum(np.linalg.norm(np.asarray(H), "fro") for H in dH) for dH in dHs]
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.standard_normal(3)
        assert G.C(x) <= float(np.abs(x) @ np.asarray(C)) + 1e-10
    for j in range(3):
        e = np.zeros(3)
        e[j] = 1.0
        assert G.C(e) == pytest.approx(C[j], rel=1e-12)


def test_joint_gauge_region_is_safe(case):
    """Random boundary points of the gauge region keep F >= FT."""
    import numpy as np
    from qrobustness import multiparam as mp
    from qrobustness.core import gate_fidelity, propagator

    problem, controllers = case
    c = controllers[0]
    dt = c["tf"] / c["tau"]
    tau = c["tau"]
    from qrobustness import dH_structure

    dHs = [
        dH_structure(problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag)
        for tag in ("H0", "H1", "H2")
    ]
    G = mp.joint_gauge(dHs, dt)
    H_list = [
        problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
        for k in range(tau)
    ]
    surplus = c["fid"] - FT
    rng = np.random.default_rng(1)
    for _ in range(20):
        d = rng.standard_normal(3)
        r = G.boundary_radius(d, surplus, FT, problem["dim"])
        mu = r * d
        Hp = [H_list[k] + sum(mu[j] * dHs[j][k] for j in range(3)) for k in range(tau)]
        F = gate_fidelity(propagator(Hp, dt), problem["Uf"])
        assert F >= FT - 1e-12


def test_certified_gauge_inradius_bounds_sampled(case):
    """The certified inradius is a valid lower bound: no direction can
    have a smaller boundary radius, and it is <= any sampled estimate."""
    import numpy as np
    from qrobustness import dH_structure
    from qrobustness import multiparam as mp

    problem, controllers = case
    c = controllers[0]
    dt = c["tf"] / c["tau"]
    dHs = [
        dH_structure(problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag)
        for tag in ("H0", "H1", "H2")
    ]
    G = mp.joint_gauge(dHs, dt)
    surplus = c["fid"] - FT
    r_cert = G.inradius_certified(surplus, FT, problem["dim"])
    rng = np.random.default_rng(4)
    for _ in range(200):
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        assert G.boundary_radius(d, surplus, FT, problem["dim"]) >= r_cert - 1e-15


def test_angular_gauge_contains_joint_gauge_region(case):
    """Full-gauge dominance: every point certified by the joint
    Lipschitz gauge is certified by the static angular gauge."""
    import numpy as np
    from qrobustness import dH_structure
    from qrobustness import multiparam as mp

    problem, controllers = case
    c = controllers[0]
    dt = c["tf"] / c["tau"]
    dHs = [
        dH_structure(problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag)
        for tag in ("H0", "H1", "H2")
    ]
    G = mp.joint_gauge(dHs, dt)
    A = mp.angular_gauge(dHs, dt)
    F0 = c["fid"]
    surplus = F0 - FT
    rng = np.random.default_rng(7)
    for _ in range(100):
        d = rng.standard_normal(3)
        r_joint = G.boundary_radius(d, surplus, FT, problem["dim"])
        x = r_joint * d
        assert A.contains(x, F0, FT), "dominance violated"
    # and the angular boundary points are safe (spot check)
    from qrobustness.core import gate_fidelity, propagator

    tau = c["tau"]
    H_list = [
        problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
        for k in range(tau)
    ]
    for _ in range(10):
        d = rng.standard_normal(3)
        r = A.boundary_radius(d, F0, FT)
        mu = r * d
        Hp = [H_list[k] + sum(mu[j] * dHs[j][k] for j in range(3)) for k in range(tau)]
        assert gate_fidelity(propagator(Hp, dt), problem["Uf"]) >= FT - 1e-12

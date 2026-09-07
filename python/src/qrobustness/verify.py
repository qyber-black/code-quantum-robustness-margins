# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Numerical verification harness for the theorems and certificates.

Every theorem shipped with the package can be re-verified numerically
against independent computations.  The harness follows two principles.

*Robust numerics.*  Certified inequalities are checked against slacks,
never equalities, and the allowance is derived from the computation
rather than assumed: fidelities are evaluated through two independent
propagator routes (eigendecomposition and ``scipy.linalg.expm``), the
observed cross-route discrepancy and the unitarity defect of the
product set the tolerance, and a check only counts as a violation when
the slack is negative beyond that allowance.

*Adversarial defaults.*  Certificates are probed where they are most
likely to fail: at the certified boundary, with sign-modulated
sub-interval trajectories (the failure mode of weaker readings), and
with multi-start gradient adversaries, not only with random samples.

Fast, seeded instances of every check run in the test suite
(``tests/test_theorems.py``); ``scripts/run_theorem_verification.py``
sweeps the shipped ensembles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group

from .core import (
    Array,
    HList,
    segment_eig,
    segment_propagator,
    gate_fidelity,
    propagator,
)
from .lengthspace import refine as _refine
from .timevarying import adversarial_fidelity

#: Acceptance tolerances, in three tiers, named so the harness's policy is
#: visible in one place rather than spread over the call sites. The values are
#: unchanged; only their names are new.
#:
#: Identities that hold to roundoff: absorption, the Lipschitz pair bound, and
#: the trajectory slope.
TOL_EXACT = 1e-12
#: Angle and Choi-geometry checks, which accumulate over a propagated path.
TOL_ANGLE = 1e-9
#: The metric triangle inequality, sampled over triples: the loosest, because
#: it compounds three separately computed angles.
TOL_TRIANGLE = 1e-7

__all__ = [
    "CheckReport",
    "unitarity_defect",
    "fidelity_cross_check",
    "check_metric_triangle",
    "check_absorption",
    "check_constant_margin",
    "check_polytope",
    "check_lipschitz_pairs",
    "check_tv_slope",
    "check_fs_angle",
    "check_trajectory_certificate",
]


@dataclass
class CheckReport:
    """Outcome of one certificate check.

    ``passed`` is ``min_slack >= -tol``: the certified inequality held
    on every probe, up to the derived numerical allowance ``tol``.
    ``min_slack`` is the smallest observed slack (certificate quantity
    minus its bound, oriented so that nonnegative means satisfied) and
    ``argmin`` identifies the probe that attained it.
    """

    name: str
    n_checks: int
    min_slack: float
    tol: float
    argmin: object = None
    details: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        # A check that ran no probe has established nothing, so it must not
        # report success. Without the n_checks guard an all-skipped run
        # leaves min_slack at +inf and passes vacuously -- the failure mode
        # where a green report is backed by no evidence at all.
        return self.n_checks > 0 and self.min_slack >= -self.tol


def unitarity_defect(U: Array) -> float:
    """``|| U'U - I ||_2``: roundoff accumulated in a propagator product."""
    N = U.shape[0]
    return float(np.linalg.norm(U.conj().T @ U - np.eye(N), 2))


def fidelity_cross_check(H_list: HList, dt: float, Uf: Array) -> tuple:
    """Fidelity via two independent propagator routes and an allowance.

    Returns ``(F, tol)`` where ``F`` is the ``expm``-route fidelity and
    ``tol`` bounds its numerical uncertainty: the observed discrepancy
    against the eigendecomposition route plus the unitarity defect,
    floored at ``100 * eps``.

    The second route must not be another ``expm`` loop. ``propagator``
    already is one, so building ``U2`` the same way made the discrepancy
    identically zero and the allowance the unitarity defect alone --
    while the docstring, and the paper, claimed a cross-route check.
    The eigendecomposition route below is the one ``differential_sensitivity``
    uses, and is genuinely independent of ``scipy.linalg.expm``.
    """
    U1 = propagator(H_list, dt)
    U2 = np.eye(U1.shape[0], dtype=complex)
    for H in H_list:
        lam, V = segment_eig(H)
        U2 = segment_propagator(lam, V, dt) @ U2
    F1 = gate_fidelity(U1, Uf)
    F2 = gate_fidelity(U2, Uf)
    tol = abs(F1 - F2) + unitarity_defect(U1) + 100 * np.finfo(float).eps
    return F1, float(tol)


def _rand_traj(rng, tau: int, m: float) -> Array:
    """Adversarially-shaped random trajectory: mix of uniform, extreme
    (sign-modulated at full budget) and sparse profiles."""
    kind = rng.integers(3)
    if kind == 0:
        return rng.uniform(-m, m, tau)
    if kind == 1:
        return m * rng.choice([-1.0, 1.0], tau)
    d = np.zeros(tau)
    idx = rng.choice(tau, max(1, tau // 4), replace=False)
    d[idx] = m * rng.choice([-1.0, 1.0], idx.size)
    return d


def check_metric_triangle(N: int, n_triples: int = 200, seed: int = 0) -> CheckReport:
    """Lemma (gate-fidelity angle): triangle inequality on random triples."""
    rng = np.random.default_rng(seed)

    def theta(U, V):
        return np.arccos(min(1.0, abs(np.trace(U.conj().T @ V)) / N))

    worst = np.inf
    arg = None
    for i in range(n_triples):
        U, V, W = (np.asarray(unitary_group.rvs(N, random_state=rng)) for _ in range(3))
        slack = theta(U, V) + theta(V, W) - theta(U, W)
        if slack < worst:
            worst, arg = slack, i
    return CheckReport(
        "metric_triangle", n_triples, float(worst), tol=TOL_TRIANGLE, argmin=arg
    )  # arccos conditioning near 0


def check_absorption(FT: float, N: int, n: int = 200, seed: int = 0) -> CheckReport:
    """Lemma (nominal-error absorption): if the achieved-gate fidelity
    meets the angular threshold, the target fidelity meets ``FT``."""
    from .kosut import effective_threshold

    rng = np.random.default_rng(seed)
    worst = np.inf
    arg = None
    n_done = 0
    for i in range(n):
        Uf = np.asarray(unitary_group.rvs(N, random_state=rng))
        # Achieved gate: a small random rotation of the target.
        A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        Hp = (A + A.conj().T) / 2
        eps_angle = rng.uniform(0.0, 0.9) * np.arccos(FT)
        Hp *= eps_angle / max(np.linalg.norm(Hp, 2), 1e-300)
        US = expm(-1j * Hp) @ Uf
        eps0 = 1.0 - gate_fidelity(US, Uf)
        F_eff = effective_threshold(FT, eps0, "angular")
        if F_eff >= 1.0:
            continue
        # A perturbed evolution exactly at the achieved-gate threshold.
        theta_room = np.arccos(F_eff)
        B = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        Hq = (B + B.conj().T) / 2
        Hq *= theta_room / max(np.linalg.norm(Hq, 2), 1e-300)
        U = expm(-1j * Hq) @ US
        if gate_fidelity(U, US) < F_eff:
            continue  # rotation overshot the threshold; not a test case
        slack = gate_fidelity(U, Uf) - FT
        n_done += 1
        if slack < worst:
            worst, arg = slack, i
    return CheckReport("absorption", n_done, float(worst), tol=TOL_EXACT, argmin=arg)


def check_constant_margin(
    fid_fn: Callable[[float], float],
    M: float,
    FT: float,
    n: int = 201,
    tol: float = 1e-12,
) -> CheckReport:
    """A certified constant margin: ``F(mu) >= FT`` on a dense grid of
    ``[-M, M]`` (both signs; endpoints included)."""
    grid = np.linspace(-M, M, n)
    vals = np.array([fid_fn(float(m)) for m in grid])
    i = int(np.argmin(vals))
    return CheckReport(
        "constant_margin", n, float(vals[i] - FT), tol=tol, argmin=float(grid[i])
    )


def check_polytope(
    fn: Callable[[Array], float],
    L: Array,
    F0: float,
    FT: float,
    n: int = 200,
    seed: int = 0,
    tol: float = 1e-12,
) -> CheckReport:
    """Certified safe polytope: ``F(mu) >= FT`` for random boundary and
    interior points of ``sum_j L_j |mu_j| <= F0 - FT``."""
    rng = np.random.default_rng(seed)
    L = np.asarray(L, dtype=float)
    surplus = F0 - FT
    worst = np.inf
    arg = None
    for i in range(n):
        d = rng.standard_normal(L.size)
        d /= np.abs(d) @ L
        r = surplus * (1.0 if i % 2 == 0 else rng.uniform())
        mu = r * d
        slack = fn(mu) - FT
        if slack < worst:
            worst, arg = slack, mu.copy()
    return CheckReport("polytope", n, float(worst), tol=tol, argmin=arg)


def check_lipschitz_pairs(
    H_list: HList,
    Hhat_lists: Sequence[HList],
    dt: float,
    Uf: Array,
    L: Array,
    FT: float,
    m: float,
    n: int = 100,
    seed: int = 0,
) -> CheckReport:
    """Trajectory Lipschitz lemma: ``|F[a] - F[b]| <= sum_j L_j
    ||a_j - b_j||_inf`` for random trajectory pairs inside the safe set.

    Each structure list must match ``H_list`` interval for interval; without
    the check a short one failed later with a bare IndexError naming no
    argument.
    """
    tau = len(H_list)
    for j, Hh in enumerate(Hhat_lists):
        if len(Hh) != tau:
            raise ValueError(
                f"Hhat_lists[{j}] has length {len(Hh)}, expected {tau} to match H_list"
            )
    if len(L) != len(Hhat_lists):
        raise ValueError(
            f"L has length {len(L)}, expected {len(Hhat_lists)} to match Hhat_lists"
        )
    rng = np.random.default_rng(seed)
    tau = len(H_list)
    p = len(Hhat_lists)

    def F(deltas) -> float:
        Hp = [
            np.asarray(H_list[k])
            + sum(deltas[j][k] * np.asarray(Hhat_lists[j][k]) for j in range(p))
            for k in range(tau)
        ]
        return gate_fidelity(propagator(Hp, dt), Uf)

    worst = np.inf
    arg = None
    n_done = 0
    for i in range(n):
        a = [_rand_traj(rng, tau, m) for _ in range(p)]
        b = [_rand_traj(rng, tau, m) for _ in range(p)]
        Fa, Fb = F(a), F(b)
        if min(Fa, Fb) <= FT:
            continue  # lemma hypothesis requires the safe set
        bound = sum(L[j] * np.max(np.abs(a[j] - b[j])) for j in range(p))
        slack = bound - abs(Fa - Fb)
        n_done += 1
        if slack < worst:
            worst, arg = slack, i
    return CheckReport(
        "lipschitz_pairs", n_done, float(worst), tol=TOL_EXACT, argmin=arg
    )


def check_tv_slope(
    H_list: HList,
    Hhat_lists: Sequence[HList],
    dt: float,
    Uf: Array,
    L: Array,
    FT: float,
    m: float,
    n: int = 120,
    n_homotopy: int = 3,
    seed: int = 0,
) -> CheckReport:
    """Trajectory Lipschitz lemma, slope form: the realised directional
    derivative never exceeds the certified bound.

    The lemma bounds ``|dF/ds|`` along ``s -> delta + s d`` by
    ``sum_j L_j ||d_j||_inf``.  This probes that bound directly, with a
    central difference at several homotopy points along each random
    direction, and reports both the worst slack and -- in ``details`` --
    the largest realised fraction of the bound, which is what says how
    much room the bound leaves in practice.

    ``n_checks`` counts direction-by-homotopy-point probes, so the
    reported count is ``n * n_homotopy``.
    """
    rng = np.random.default_rng(seed)
    tau = len(H_list)
    p = len(Hhat_lists)

    def F(deltas) -> float:
        Hp = [
            np.asarray(H_list[k])
            + sum(deltas[j][k] * np.asarray(Hhat_lists[j][k]) for j in range(p))
            for k in range(tau)
        ]
        return gate_fidelity(propagator(Hp, dt), Uf)

    # Step for the central difference: small against the budget, large
    # against the fidelity's own evaluation noise.
    h = max(m * 1e-4, 1e-9)
    worst = np.inf
    max_frac = 0.0
    arg = None
    n_done = 0
    for i in range(n):
        d = [_rand_traj(rng, tau, 1.0) for _ in range(p)]
        bound = float(sum(L[j] * np.max(np.abs(d[j])) for j in range(p)))
        if bound <= 0.0:
            continue
        base = [np.zeros(tau) for _ in range(p)]
        for hp in range(n_homotopy):
            s0 = m * (hp + 1) / n_homotopy
            lo = [base[j] + (s0 - h) * d[j] for j in range(p)]
            hi = [base[j] + (s0 + h) * d[j] for j in range(p)]
            Flo, Fhi = F(lo), F(hi)
            if min(Flo, Fhi) <= FT:
                continue  # the lemma is a statement on the safe set
            realised = abs(Fhi - Flo) / (2.0 * h)
            n_done += 1
            frac = realised / bound
            if frac > max_frac:
                max_frac = frac
            slack = bound - realised
            if slack < worst:
                worst, arg = slack, (i, hp)
    if n_done == 0:
        return CheckReport("tv_slope", 0, float("inf"), tol=TOL_EXACT)
    return CheckReport(
        "tv_slope",
        n_done,
        float(worst),
        tol=TOL_ANGLE,
        argmin=arg,
        details={"max_fraction_of_bound": float(max_frac)},
    )


def check_fs_angle(
    H_list: HList,
    Hhat_list: HList,
    dt: float,
    m: float,
    s: float,
    n: int = 100,
    refine: int = 8,
    seed: int = 0,
) -> CheckReport:
    """Theorem (FS certificate), inner inequality: the angle between the
    nominal and perturbed propagators is at most ``m * s`` for random
    sub-interval-refined trajectories with ``||delta||_inf <= m``."""
    rng = np.random.default_rng(seed)
    tau = len(H_list)
    N = H_list[0].shape[0]
    Hr, dHr = _refine(H_list, Hhat_list, refine)
    dtr = dt / refine
    US = propagator(Hr, dtr)
    worst = np.inf
    arg = None
    for i in range(n):
        d = _rand_traj(rng, tau * refine, m)
        U = propagator([Hr[k] + d[k] * dHr[k] for k in range(len(Hr))], dtr)
        ang = np.arccos(min(1.0, abs(np.trace(US.conj().T @ U)) / N))
        slack = m * s - ang
        if slack < worst:
            worst, arg = slack, i
    return CheckReport("fs_angle", n, float(worst), tol=TOL_ANGLE, argmin=arg)


def check_trajectory_certificate(
    H_list: HList,
    Hhat_list: HList,
    dt: float,
    Uf: Array,
    FT: float,
    m: float,
    refinements: Sequence[int] = (1, 4),
    n_starts: int = 4,
    maxiter: int = 200,
    n_random: int = 50,
    seed: int = 0,
) -> CheckReport:
    """A certified uniform trajectory margin (``r_0``, ``r_FS`` or
    ``M^K_tv``): gradient adversaries plus random adversarially-shaped
    trajectories at budget ``m`` must not break ``F >= FT``."""
    rng = np.random.default_rng(seed)
    worst = np.inf
    arg = None
    n_total = 0
    for q in refinements:
        Hr, dHr = _refine(H_list, Hhat_list, q)
        dtr = dt / q
        Fmin, _ = adversarial_fidelity(
            Hr,
            dHr,
            dtr,
            Uf,
            m,
            n_starts=n_starts,
            maxiter=maxiter,
            seed=int(rng.integers(2**31)),
        )
        n_total += 1
        if Fmin - FT < worst:
            worst, arg = Fmin - FT, f"adversary_x{q}"
        for i in range(n_random):
            d = _rand_traj(rng, len(Hr), m)
            F = gate_fidelity(
                propagator([Hr[k] + d[k] * dHr[k] for k in range(len(Hr))], dtr), Uf
            )
            n_total += 1
            if F - FT < worst:
                worst, arg = F - FT, f"random_x{q}_{i}"
    _, tol = fidelity_cross_check(H_list, dt, Uf)
    return CheckReport(
        "trajectory_certificate", n_total, float(worst), tol=tol, argmin=arg
    )

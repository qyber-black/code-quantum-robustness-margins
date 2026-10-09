# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Numerical checks of the certificates against independent computations.

Each ``check_*`` function probes one certified inequality: at the boundary,
on sign-modulated sub-interval trajectories, and against gradient
adversaries. It returns a :class:`CheckReport`. The check fails only when
the slack is negative by more than its tolerance. Two propagator routes can
cross-check the fidelities and set that tolerance.
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

#: Tolerance for identities that hold to roundoff.
TOL_EXACT = 1e-12
#: Tolerance for angle checks, which accumulate along a propagated path.
TOL_ANGLE = 1e-9
#: Tolerance for the sampled triangle inequality (three separately computed angles).
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
    """Result of one certificate check.

    Attributes
    ----------
    name :
        Check identifier.
    n_checks :
        Number of probes evaluated.
    min_slack :
        Smallest slack (certified quantity minus bound; nonnegative means
        satisfied).
    tol :
        Numerical allowance on the slack.
    argmin :
        Identifier of the probe attaining ``min_slack``.
    details :
        Check-specific extra values.
    passed :
        ``n_checks > 0 and min_slack >= -tol``; a run with no probes fails.
    """

    name: str
    n_checks: int
    min_slack: float
    tol: float
    argmin: object = None
    details: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        # No probe means no evidence, so an all-skipped run does not pass.
        return self.n_checks > 0 and self.min_slack >= -self.tol


def unitarity_defect(U: Array) -> float:
    """Unitarity defect ``||U^dag U - I||_2`` of a computed propagator.

    Parameters
    ----------
    U :
        ``(N, N)`` matrix.

    Returns
    -------
    float
        Spectral norm of ``U^dag U - I``.
    """
    N = U.shape[0]
    return float(np.linalg.norm(U.conj().T @ U - np.eye(N), 2))


def fidelity_cross_check(H_list: HList, dt: float, Uf: Array) -> tuple:
    """Gate fidelity with a numerical allowance from two propagator routes.

    Parameters
    ----------
    H_list :
        Per-interval Hamiltonians, each ``(N, N)``.
    dt :
        Interval length ``Delta``.
    Uf :
        Target gate.

    Returns
    -------
    F : float
        Fidelity through the ``expm`` route.
    tol : float
        ``|F_expm - F_eig|`` plus the unitarity defect plus ``100 eps``.
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
    """Random trajectory, ``|delta_k| <= m``: uniform, sign-modulated or sparse."""
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
    """Triangle inequality of the gate-fidelity angle on random unitary triples.

    Parameters
    ----------
    N :
        Dimension.
    n_triples :
        Number of Haar-random triples.
    seed :
        Random seed.

    Returns
    -------
    CheckReport
    """
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
    """Angular absorption: achieved-gate fidelity ``>= F_eff`` implies ``F >= F_T``.

    Parameters
    ----------
    FT :
        Threshold ``F_T``.
    N :
        Dimension.
    n :
        Number of random target/nominal/perturbed triples.
    seed :
        Random seed.

    Returns
    -------
    CheckReport
    """
    from .kosut import effective_threshold

    rng = np.random.default_rng(seed)
    worst = np.inf
    arg = None
    n_done = 0
    for i in range(n):
        Uf = np.asarray(unitary_group.rvs(N, random_state=rng))
        # Achieved gate: a slight random rotation of the target.
        A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        Hp = (A + A.conj().T) / 2
        eps_angle = rng.uniform(0.0, 0.9) * np.arccos(FT)
        Hp *= eps_angle / max(np.linalg.norm(Hp, 2), 1e-300)
        US = expm(-1j * Hp) @ Uf
        eps0 = 1.0 - gate_fidelity(US, Uf)
        F_eff = effective_threshold(FT, eps0, "angular")
        if F_eff >= 1.0:
            continue
        # Perturbed evolution placed exactly at the achieved-gate threshold.
        theta_room = np.arccos(F_eff)
        B = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        Hq = (B + B.conj().T) / 2
        Hq *= theta_room / max(np.linalg.norm(Hq, 2), 1e-300)
        U = expm(-1j * Hq) @ US
        if gate_fidelity(U, US) < F_eff:
            continue  # rotation overshot the threshold; not treated as a test case
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
    """Constant margin: ``F(mu) >= F_T`` on a uniform grid over ``[-M, M]``.

    Parameters
    ----------
    fid_fn :
        Fidelity as a function of the scalar perturbation ``mu``.
    M :
        Margin under test.
    FT :
        Threshold ``F_T``.
    n :
        Grid points (endpoints included).
    tol :
        Allowance on the slack.

    Returns
    -------
    CheckReport
    """
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
    """Safe polytope: ``F(mu) >= F_T`` on ``sum_j L_j |mu_j| <= F_0 - F_T``.

    Parameters
    ----------
    fn :
        Fidelity as a function of the parameter vector ``mu`` (length ``p``).
    L :
        Lipschitz constants ``L_j``, shape ``(p,)``.
    F0 :
        Nominal fidelity ``F_0``.
    FT :
        Threshold ``F_T``.
    n :
        Number of random points (alternately on the boundary and inside).
    seed :
        Random seed.

    tol :
        Allowance on the slack.

    Returns
    -------
    CheckReport
    """
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
    """Trajectory Lipschitz bound ``|F[a] - F[b]| <= sum_j L_j ||a_j - b_j||_inf``.

    Parameters
    ----------
    H_list :
        Nominal per-interval Hamiltonians, each ``(N, N)``.
    Hhat_lists :
        One per-interval structure list ``Hhat_j^(k)`` per parameter.
    dt :
        Interval length ``Delta``.
    Uf :
        Target gate.
    L :
        Trajectory Lipschitz constants ``L_j``, shape ``(p,)``.
    FT :
        Threshold ``F_T``; only probes in the safe set count.
    m :
        Sup-norm budget of the random trajectories.
    n :
        Number of random trajectory pairs.
    seed :
        Random seed.

    Returns
    -------
    CheckReport
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
            continue  # the bound holds on the safe set only
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
    """Slope form of the trajectory Lipschitz bound, by central differences.

    ``|dF/ds|`` along ``delta = s d`` must not exceed ``sum_j L_j ||d_j||_inf``.

    Parameters
    ----------
    H_list :
        Nominal per-interval Hamiltonians, each ``(N, N)``.
    Hhat_lists :
        One per-interval structure list ``Hhat_j^(k)`` per parameter.
    dt :
        Interval length ``Delta``.
    Uf :
        Target gate.
    L :
        Trajectory Lipschitz constants ``L_j``, shape ``(p,)``.
    FT :
        Threshold ``F_T``; only probes in the safe set count.
    m :
        Sup-norm budget of the random trajectories.
    n :
        Number of random directions.
    n_homotopy :
        Probe points ``s`` per direction in ``(0, m]``.
    seed :
        Random seed.

    Returns
    -------
    CheckReport
        ``n_checks`` counts direction-point probes; ``details`` holds
        ``max_fraction_of_bound``.
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

    # Small against the budget, large against fidelity evaluation noise.
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
                continue  # the bound holds on the safe set only
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
    """Angle between nominal and perturbed propagators stays at most ``m * s``.

    Probes random sub-interval-refined trajectories with ``||delta||_inf <= m``.

    Parameters
    ----------
    H_list :
        Nominal per-interval Hamiltonians, each ``(N, N)``.
    Hhat_list :
        Per-interval structures ``Hhat^(k)``.
    dt :
        Interval length ``Delta``.
    m :
        Sup-norm budget.
    s :
        Angular rate under test (e.g. the Choi path-length constant ``s_j``).
    n :
        Number of random trajectories.
    refine :
        Sub-intervals per control interval.
    seed :
        Random seed.

    Returns
    -------
    CheckReport
    """
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
    """Trajectory certificate (``r_0``, ``r_FS``, ``M^K_tv``): ``F >= F_T`` at ``m``.

    Probes with gradient adversaries and random trajectories on each grid
    refinement; failure to find a violation is evidence only within this search.

    Parameters
    ----------
    H_list :
        Nominal per-interval Hamiltonians, each ``(N, N)``.
    Hhat_list :
        Per-interval structures ``Hhat^(k)``.
    dt :
        Interval length ``Delta``.
    Uf :
        Target gate.
    FT :
        Threshold ``F_T``.
    m :
        Sup-norm budget (the certificate under test).
    refinements :
        Sub-interval refinement factors to attack on.
    n_starts, maxiter :
        Passed to :func:`timevarying.adversarial_fidelity`.
    n_random :
        Random trajectories per refinement.
    seed :
        Random seed.

    Returns
    -------
    CheckReport
        ``tol`` from :func:`fidelity_cross_check` at the nominal controller.
    """
    rng = np.random.default_rng(seed)
    worst = np.inf
    arg = None
    n_total = 0
    for q in refinements:
        Hr, dHr = _refine(H_list, Hhat_list, q)
        dtr = dt / q
        Fmin, _, _nfev = adversarial_fidelity(
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
    _, tol_nom = fidelity_cross_check(H_list, dt, Uf)
    perturbed = [H_list[k] + m * Hhat_list[k] for k in range(len(H_list))]
    _, tol_pert = fidelity_cross_check(perturbed, dt, Uf)
    tol = max(tol_nom, tol_pert)
    return CheckReport(
        "trajectory_certificate", n_total, float(worst), tol=tol, argmin=arg
    )

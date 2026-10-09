# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Certified and empirical margins for time-varying structured uncertainty.

The Lipschitz radius r_0 is :func:`uniform_margin`. The Choi-Fubini-Study
radius r_FS is :func:`fs_margin` and :func:`fs_margin_joint`. An adversarial
search, :func:`adversarial_upper_bound`, returns the heuristic upper end
m_adv of the bracket on M_tv. r_0 and r_FS certify every measurable
trajectory. The iterated margin M certifies constant perturbations only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .core import (
    Array,
    HList,
    dU_dmu_exact,
    segment_eig,
    segment_propagator,
    gate_fidelity,
)
from .lengthspace import PathGauge, angle_budget, interval_grams

__all__ = [
    "uniform_margin",
    "tv_fidelity_and_gradient",
    "adversarial_upper_bound",
    "TVBracket",
    "tv_bracket",
    "FSMargin",
    "fs_margin",
    "fs_margin_joint",
]


def uniform_margin(L, F: float, FT: float) -> float:
    """Lipschitz radius ``r_0 = (F - F_T) / sum_j L_j`` for time-varying perturbations.

    Every measurable trajectory with ``sum_j L_j ||delta_j||_inf <= F - F_T``
    keeps ``F >= F_T``.

    Parameters
    ----------
    L : array_like, shape (p,)
        Per-structure Lipschitz constants L_j.
    F :
        Nominal fidelity; must exceed ``FT``.
    FT :
        Fidelity threshold F_T.

    Returns
    -------
    float
        r_0.
    """
    L = np.atleast_1d(np.asarray(L, dtype=float))
    if not (F > FT):
        raise ValueError("Require F > FT")
    return float((F - FT) / np.sum(L))


def tv_fidelity_and_gradient(
    H_list: HList,
    Hhat_list: HList,
    delta: Array,
    dt: float,
    Uf: Array,
) -> Tuple[float, Array]:
    """Fidelity and exact gradient for a piecewise-constant trajectory ``delta``.

    Interval Hamiltonians are ``H^(k) + delta_k Hhat^(k)``.

    Parameters
    ----------
    H_list, Hhat_list : lists of tau ndarrays, shape (N, N)
        Nominal interval Hamiltonians and structures.
    delta : array_like, shape (tau,)
        Perturbation value on each interval.
    dt :
        Interval length Delta.
    Uf : ndarray, shape (N, N)
        Target unitary.

    Returns
    -------
    F : float
        Gate fidelity.
    g : ndarray, shape (tau,)
        ``dF/ddelta_k``.
    """
    delta = np.asarray(delta, dtype=float).ravel()
    tau = len(H_list)
    if delta.size != tau or len(Hhat_list) != tau:
        raise ValueError("delta and Hhat_list must have length tau")
    N = H_list[0].shape[0]

    Hp = [H_list[k] + delta[k] * Hhat_list[k] for k in range(tau)]
    eigs = [segment_eig(H) for H in Hp]
    Useg = [segment_propagator(lam, V, dt) for lam, V in eigs]

    Pref: List[Array] = [np.eye(N, dtype=complex)]
    for k in range(tau):
        Pref.append(Useg[k] @ Pref[-1])
    Utot = Pref[tau]
    F = gate_fidelity(Utot, Uf)
    if F <= 0:
        raise ValueError("Fidelity must be positive for phase")
    z = np.trace(Uf.conj().T @ Utot)
    e_minus_i_phi = np.conj(z / np.abs(z))

    Suff: List[Array] = [np.empty((N, N), dtype=complex) for _ in range(tau + 1)]
    Suff[tau] = np.eye(N, dtype=complex)
    for k in range(tau - 1, -1, -1):
        Suff[k] = Suff[k + 1] @ Useg[k]

    g = np.zeros(tau, dtype=float)
    for k in range(tau):
        lam, V = eigs[k]
        dUk = dU_dmu_exact(lam, V, Hhat_list[k], dt)
        Dk = Suff[k + 1] @ dUk @ Pref[k]
        g[k] = float(np.real(np.trace(Uf.conj().T @ Dk * e_minus_i_phi))) / N
    return F, g


def adversarial_fidelity(
    H_list: HList,
    Hhat_list: HList,
    dt: float,
    Uf: Array,
    m: float,
    n_starts: int = 4,
    seed: Optional[int] = None,
    maxiter: int = 200,
    starts: str = "legacy",
) -> Tuple[float, Array, int]:
    """Heuristic minimum of the fidelity over trajectories with ``||delta||_inf <= m``.

    Multi-start L-BFGS-B over piecewise-constant trajectories with exact
    gradients; a local search, so the result is not the true minimum.

    Parameters
    ----------
    H_list, Hhat_list : lists of tau ndarrays, shape (N, N)
        Nominal interval Hamiltonians and structures.
    dt :
        Interval length Delta.
    Uf : ndarray, shape (N, N)
        Target unitary.
    m :
        Sup-norm budget.
    n_starts :
        Number of starts, including the two constant trajectories ``+-m``.
    seed :
        Seed for random starts.
    maxiter :
        L-BFGS-B iteration limit per start.
    starts :
        ``"legacy"``: the two constant trajectories plus uniform random
        interior points. ``"mixed"``: half the random points replaced by
        random sign trajectories ``m * (+-1, ..., +-1)``.

    Returns
    -------
    F_min : float
        Smallest fidelity found.
    delta : ndarray, shape (tau,)
        Trajectory attaining it.
    nfev : int
        Fidelity evaluations inside the search.
    """
    tau = len(H_list)
    rng = np.random.default_rng(seed)

    def objective(delta):
        F, g = tv_fidelity_and_gradient(H_list, Hhat_list, delta, dt, Uf)
        return F, g

    best_F = np.inf
    best_delta = np.zeros(tau)
    nfev = 0
    start_list = [m * np.ones(tau), -m * np.ones(tau)]
    n_rand = max(n_starts - 2, 0)
    if starts == "mixed":
        n_sign = n_rand // 2
        start_list += [m * rng.choice([-1.0, 1.0], size=tau) for _ in range(n_sign)]
        n_rand -= n_sign
    elif starts != "legacy":
        raise ValueError(f"unknown starts scheme: {starts!r}")
    start_list += [rng.uniform(-m, m, size=tau) for _ in range(n_rand)]
    for x0 in start_list:
        res = minimize(
            objective,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-m, m)] * tau,
            options={"maxiter": maxiter},
        )
        nfev += int(getattr(res, "nfev", 0) or 0)
        if res.fun < best_F:
            best_F = float(res.fun)
            best_delta = np.asarray(res.x)
    return best_F, best_delta, nfev


@dataclass
class TVBracket:
    """Bracket ``r0 <= M_tv <= m_adv`` on the uniform time-varying margin.

    Attributes
    ----------
    r0 : float
        Certified lower end.
    m_adv : float
        Sup-norm budget at which the adversary found ``F < F_T``; a heuristic
        search result (an evaluated witness, not a certified bound, and not
        necessarily the least violating budget). If no violation was found it
        is the search ceiling and not a witness.
    F_at_adv : float
        Fidelity of the violating trajectory (or smallest fidelity seen).
    n_evals : int
        Fidelity evaluations inside the adversarial searches.
    delta_adv : ndarray, shape (tau,) or None
        Violating trajectory, for independent re-evaluation; ``None`` when
        no violation was found.
    """

    r0: float
    m_adv: float
    F_at_adv: float
    n_evals: int
    delta_adv: Optional[Array] = None


def adversarial_upper_bound(
    H_list: HList,
    Hhat_list: HList,
    dt: float,
    Uf: Array,
    FT: float,
    r0: float,
    m_hi: float,
    rel_tol: float = 1e-2,
    seed: Optional[int] = None,
    n_starts: int = 4,
    starts: str = "legacy",
    maxiter: int = 200,
) -> TVBracket:
    """Bracket M_tv by bisection on the adversary's sup-norm budget.

    Parameters
    ----------
    H_list, Hhat_list : lists of tau ndarrays, shape (N, N)
        Nominal interval Hamiltonians and structures.
    dt :
        Interval length Delta.
    Uf : ndarray, shape (N, N)
        Target unitary.
    FT :
        Fidelity threshold F_T.
    r0 :
        Certified lower end (e.g. r_0 or r_FS).
    m_hi :
        Search ceiling (heuristic, not a bound).
    rel_tol :
        Stop when ``(hi - lo)/hi <= rel_tol``.
    seed :
        Seed for the adversarial starts.
    n_starts, starts, maxiter :
        Passed to :func:`adversarial_fidelity`.

    Returns
    -------
    TVBracket
        If no violation is found at ``m_hi``, returned unrefined with
        ``m_adv = m_hi`` and ``delta_adv = None`` (not a witness).
    """
    lo, hi = r0, m_hi
    nfev = 0
    F_hi, d_hi, n_hi = adversarial_fidelity(
        H_list,
        Hhat_list,
        dt,
        Uf,
        hi,
        n_starts=n_starts,
        seed=seed,
        maxiter=maxiter,
        starts=starts,
    )
    nfev += n_hi
    if F_hi >= FT:
        return TVBracket(r0=r0, m_adv=hi, F_at_adv=F_hi, n_evals=nfev)
    F_at, d_at = F_hi, d_hi
    while (hi - lo) / max(hi, 1e-300) > rel_tol:
        mid = 0.5 * (lo + hi)
        F_mid, d_mid, n_mid = adversarial_fidelity(
            H_list,
            Hhat_list,
            dt,
            Uf,
            mid,
            n_starts=n_starts,
            seed=seed,
            maxiter=maxiter,
            starts=starts,
        )
        nfev += n_mid
        if F_mid < FT:
            # hi moves only to a budget with an exhibited violation.
            hi, F_at, d_at = mid, F_mid, d_mid
        else:
            lo = mid
    return TVBracket(
        r0=r0, m_adv=hi, F_at_adv=F_at, n_evals=nfev, delta_adv=np.asarray(d_at)
    )


def tv_bracket(*args, **kwargs) -> TVBracket:
    """Alias for :func:`adversarial_upper_bound`."""
    return adversarial_upper_bound(*args, **kwargs)


@dataclass
class FSMargin:
    """Choi-Fubini-Study trajectory certificate.

    Attributes
    ----------
    r_fs : float
        r_FS: every measurable trajectory with ``||delta||_inf <= r_fs``
        keeps ``F >= F_T``.
    r0 : float
        Lipschitz radius r_0 as supplied (0 if not).
    r : float
        ``max(r0, r_fs)``.
    speed : float
        Path-length constant ``s = Delta sum_k ||Hbar^(k)||_F / sqrt(N)``.
    theta_0 : float
        ``arccos F0``.
    theta_T : float
        ``arccos F_T``.
    F0 : float
        Nominal fidelity.
    speed_halfspread : float
        Weaker constant ``Delta sum_k ||Hhat^(k)||_c`` from the half spread
        ||X||_c, for diagnostics.
    """

    r_fs: float
    r0: float
    r: float
    speed: float
    theta_0: float
    theta_T: float
    F0: float
    speed_halfspread: float = float("nan")


def _traceless_fro(H: Array) -> float:
    """``||H - (Tr H / N) I||_F``, the exact Choi-speed constant."""
    H = np.asarray(H)
    N = H.shape[0]
    Hbar = H - (np.trace(H) / N) * np.eye(N)
    return float(np.linalg.norm(Hbar, "fro"))


def fs_margin(
    Hhat_list: HList,
    dt: float,
    F0: float,
    FT: float,
    r0: float = 0.0,
) -> FSMargin:
    """Choi-Fubini-Study radius ``r_FS = (arccos F_T - theta_0) / s`` for one structure.

    ``s = Delta sum_k ||Hbar^(k)||_F / sqrt(N)`` with ``Hbar`` the traceless
    part; ``r_FS = inf`` if ``s = 0``.

    Parameters
    ----------
    Hhat_list : list of tau ndarrays, shape (N, N)
        Structure on each interval.
    dt :
        Interval length Delta.
    F0 :
        Nominal fidelity; ``FT < F0 <= 1``.
    FT :
        Fidelity threshold F_T, ``0 < FT <= 1``.
    r0 :
        Optional Lipschitz radius r_0 to combine into ``r``.

    Returns
    -------
    FSMargin
    """
    if not (0.0 < FT <= 1.0) or not (0.0 < F0 <= 1.0):
        raise ValueError("Require 0 < FT, F0 <= 1")
    if not (F0 > FT):
        raise ValueError("Require F0 > FT")
    N = np.asarray(Hhat_list[0]).shape[0]
    speed = dt * float(sum(_traceless_fro(H) for H in Hhat_list)) / np.sqrt(N)
    E = [0.5 * float(np.ptp(np.linalg.eigvalsh(np.asarray(H)))) for H in Hhat_list]
    speed_hs = dt * float(np.sum(E))
    theta_T = float(np.arccos(FT))
    theta_0 = float(np.arccos(min(F0, 1.0)))
    r_fs = (theta_T - theta_0) / speed if speed > 0 else float("inf")
    return FSMargin(
        r_fs=float(r_fs),
        r0=float(r0),
        r=max(float(r0), float(r_fs)),
        speed=speed,
        theta_0=theta_0,
        theta_T=theta_T,
        F0=float(F0),
        speed_halfspread=speed_hs,
    )


def toggling_frame_integral(H_list: HList, dHhat_list: HList, dt: float) -> Array:
    """Toggling-frame integral ``int_0^{t_f} U_S(t)' Hbar(t) U_S(t) dt`` of a structure.

    Exact for piecewise-constant Hamiltonians (eigenbasis divided differences).

    Parameters
    ----------
    H_list : list of tau ndarrays, shape (N, N)
        Nominal interval Hamiltonians.
    dHhat_list : list of tau ndarrays, shape (N, N)
        Structure on each interval; its traceless part is used.
    dt :
        Interval length Delta.

    Returns
    -------
    ndarray, shape (N, N)
    """
    if len(H_list) != len(dHhat_list):
        raise ValueError("H_list and dHhat_list must have the same length")
    N = np.asarray(H_list[0]).shape[0]
    total = np.zeros((N, N), dtype=complex)
    U = np.eye(N, dtype=complex)  # U_S at the start of the current interval
    for H, dH in zip(H_list, dHhat_list, strict=True):
        lam, V = segment_eig(np.asarray(H))
        dHbar = np.asarray(dH)
        dHbar = dHbar - (np.trace(dHbar) / N) * np.eye(N)
        X = 0.5 * dt * (lam[:, None] - lam[None, :])
        zero = X == 0.0
        sinc = np.where(zero, 1.0, np.sin(X) / np.where(zero, 1.0, X))
        Phi = dt * np.exp(1j * X) * sinc
        seg = V @ ((V.conj().T @ dHbar @ V) * Phi) @ V.conj().T
        total += U.conj().T @ seg @ U
        U = segment_propagator(lam, V, dt) @ U
    return total


def fs_margin_joint(
    Hhat_lists: Sequence[HList],
    dt: float,
    F0: float,
    FT: float,
) -> tuple:
    """Joint Choi-Fubini-Study trajectory certificate for p structures.

    Every measurable trajectory with ``|delta_j(t)| <= m_j`` and
    ``ell(m) <= budget`` keeps ``F >= F_T``, where ``ell`` is the box bound
    of the path gauge ``Delta sum_k sqrt(x^T Q^(k) x)``.

    Parameters
    ----------
    Hhat_lists : sequence of p lists of tau ndarrays, shape (N, N)
        Per-structure, per-interval structures.
    dt :
        Interval length Delta.
    F0 :
        Nominal fidelity.
    FT :
        Fidelity threshold F_T.

    Returns
    -------
    ell : callable
        ``ell(m)`` for box half-widths ``m`` of shape (p,)
        (``PathGauge.C_box``).
    budget : float
        ``arccos F_T - theta_0``.
    """
    gauge = PathGauge(
        grams=interval_grams(Hhat_lists, make_traceless=True, normalise=True), dt=dt
    )
    return gauge.C_box, angle_budget(F0, FT)

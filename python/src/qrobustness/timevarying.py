# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Certified margins against time-varying structured uncertainty.

The trajectory Lipschitz lemma certifies every measurable perturbation
trajectory delta(t) with ``sum_j L_j ||delta_j||_inf <= F - F_T``; the
certified uniform radius therefore equals the *first* Lipschitz step of the
scalar iteration,

    r_0 = (F - F_T) / sum_j L_j,

and is returned by :func:`uniform_margin`.  Its certificate semantics
differ from the iterated margin of :func:`qrobustness.iterative_margin`:
the latter certifies constant (more generally, fixed-direction)
perturbations only, and always satisfies ``M >= r_0``.

:func:`fs_margin` provides a second, geometric uniform certificate
``r_fs`` in closed form (Fubini-Study angle budget over the integrated
half-spread of the structure). The dominance theorem gives ``r_fs >= r_0``
whenever both are built from the same structure, so ``r_fs`` is the
certified uniform radius and ``r_0`` is retained as the baseline that
theorem quantifies. ``FSMargin.r`` still reports ``max(r_0, r_fs)``, which
is the same number under dominance and stays correct if a caller supplies
an ``r_0`` computed from a different structure.

Error control follows the established pattern: the certified radius is a
lower bound on the true uniform time-varying margin ``M_tv``, and
:func:`adversarial_upper_bound` produces an empirical upper bound by
optimising a piecewise-constant trajectory that violates the threshold,
so ``M_tv`` is bracketed by ``[r_fs, m_adv]``.
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
    """Certified uniform margin against time-varying structured uncertainty.

    Every measurable trajectory ``delta_j(t)`` with
    ``sum_j L_j ||delta_j||_inf <= F - F_T`` is certified to keep the
    fidelity at or above ``F_T``; in particular, for a single structure,
    every ``|delta(t)| <= r_0`` with ``r_0`` the returned value.  This is
    the first Lipschitz step of the scalar iteration; the iterated margin
    is larger but certifies constant perturbations only.
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
    """Fidelity and exact gradient for a piecewise-constant trajectory.

    The perturbed interval Hamiltonians are
    ``H^(k) + delta_k Hhat^(k)``; the gradient is ``dF/ddelta_k``,
    evaluated with the closed-form interval derivative (one Hermitian
    eigendecomposition per interval, as in
    :func:`qrobustness.fidelity_and_gradient`).
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
) -> Tuple[float, Array]:
    """Minimise the fidelity over trajectories with ``||delta||_inf <= m``.

    Piecewise-constant trajectories on the control grid suffice for an
    upper bound on the worst case.  Returns the smallest fidelity found and
    the minimising trajectory; multi-start L-BFGS-B with exact gradients.

    ``starts`` selects the initialisation set: ``"legacy"`` uses the two
    sign-saturated trajectories plus uniform random interior points;
    ``"mixed"`` replaces half of the random points with sign-modulated
    boundary trajectories ``m * (+-1, ..., +-1)`` -- the known worst-case
    family for margins derived from coherent time averages, which uniform
    interior starts rarely reach.
    """
    tau = len(H_list)
    rng = np.random.default_rng(seed)

    def objective(delta):
        F, g = tv_fidelity_and_gradient(H_list, Hhat_list, delta, dt, Uf)
        return F, g

    best_F = np.inf
    best_delta = np.zeros(tau)
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
        if res.fun < best_F:
            best_F = float(res.fun)
            best_delta = np.asarray(res.x)
    return best_F, best_delta


@dataclass
class TVBracket:
    """Bracket on the uniform time-varying margin ``M_tv``.

    ``r0 <= M_tv <= m_adv``: ``r0`` is certified (Theorem tv); ``m_adv`` is a
    sup-norm budget at which the adversary produced ``F < F_T`` (empirical,
    an upper bound on the true worst-case margin, and not necessarily the
    least such budget -- the inner minimisation is a heuristic local search
    and the bisection assumes a monotonicity it cannot guarantee).
    ``F_at_adv`` is the violating fidelity found there.

    The name is ``m_adv`` and not ``m_ub`` on purpose, here as in the
    manuscript: ``M_upper`` is the certified threshold bracket of
    Algorithm 1, and an "upper bound" spelling made this heuristic witness
    read as the same kind of object.

    ``delta_adv`` is the violating trajectory itself, one value per
    interval, so the witness can be re-evaluated through an independent
    propagator route rather than trusted on the search's own arithmetic.
    It is ``None`` exactly when no violation was found, which is also the
    case in which the entry is not a witness.
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
) -> TVBracket:
    """Bracket ``M_tv`` by bisection on the adversary's sup-norm budget.

    Starts from ``[r0, m_hi]`` and returns the refined bracket. ``r0`` is a
    certified lower end. ``m_hi`` is a search ceiling and not a bound:
    the theory orders ``r0 <= M_tv <= M_const`` and ``M <= M_const``, which
    says nothing about ``M`` against ``M_tv``, so a multiple of the constant
    margin is a heuristic starting point that becomes an upper witness only
    once a violating trajectory is exhibited at it.

    If the adversary never violates the threshold at ``m_hi``, the bracket
    is returned unrefined with ``m_adv = m_hi`` and ``F_at_adv`` the
    smallest fidelity seen; that entry is not a witness, and the caller is
    expected to treat it as such.
    """
    lo, hi = r0, m_hi
    F_hi, d_hi = adversarial_fidelity(H_list, Hhat_list, dt, Uf, hi, seed=seed)
    n = 1
    if F_hi >= FT:
        # No violation at the ceiling, so there is no witness to carry.
        return TVBracket(r0=r0, m_adv=hi, F_at_adv=F_hi, n_evals=n)
    F_at, d_at = F_hi, d_hi
    while (hi - lo) / max(hi, 1e-300) > rel_tol:
        mid = 0.5 * (lo + hi)
        F_mid, d_mid = adversarial_fidelity(H_list, Hhat_list, dt, Uf, mid, seed=seed)
        n += 1
        if F_mid < FT:
            # The upper end only ever moves to a budget at which a
            # violating trajectory was exhibited, so d_at always belongs
            # to the returned m_adv.
            hi, F_at, d_at = mid, F_mid, d_mid
        else:
            lo = mid
    return TVBracket(
        r0=r0, m_adv=hi, F_at_adv=F_at, n_evals=n, delta_adv=np.asarray(d_at)
    )


def tv_bracket(*args, **kwargs) -> TVBracket:
    """Alias for :func:`adversarial_upper_bound`."""
    return adversarial_upper_bound(*args, **kwargs)


@dataclass
class FSMargin:
    """Fubini-Study trajectory certificate.

    ``r_fs`` certifies every measurable trajectory
    ``||delta||_inf <= r_fs`` (Theorem fs of the paper); ``r`` is
    ``max(r0, r_fs)`` when the first-order radius ``r0`` is supplied
    (both are valid certificates, and ``r_fs >= r0`` holds by the
    dominance theorem whenever both use the same structure).
    ``speed`` is the exact Choi-state Fubini-Study path length per unit
    ``||delta||_inf``: ``dt sum_k ||Hhatbar^(k)||_F / sqrt(N)`` with
    ``Hhatbar`` the traceless part.  ``speed_halfspread`` is the weaker
    half-spread constant retained for diagnostics.
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
    """Closed-form uniform time-varying margin via the Fubini-Study angle.

    The perturbed propagator relative to the nominal one,
    ``W(t) = U_S(t)' U(t)``, obeys ``dW/dt = -i delta(t) Htil(t) W``
    exactly.  The normalised Choi state of ``W`` remains maximally
    entangled, so its Fubini-Study speed is not merely bounded but
    EXACT: the energy variance of ``G = delta Htil`` in that state is
    ``Tr(Gbar^2)/N`` with ``Gbar`` the traceless part, whence

        v_FS(t) = |delta(t)| ||Hhatbar^(k)||_F / sqrt(N)

    on interval ``k`` (Frobenius norm and trace are invariant under the
    isospectral conjugation).  With ``theta(U, V) = arccos(|Tr(U'V)|/N)``
    the Fubini-Study angle (a metric, cf. the nominal-error absorption
    lemma), the path-length bound and the triangle inequality give, for
    every measurable trajectory with ``||delta||_inf <= m``,

        theta(Uf, U(T)) <= theta_0 + m s,
        s = dt sum_k ||Hhatbar^(k)||_F / sqrt(N),

    hence ``F(delta) >= cos(min(theta_0 + m s, pi/2))`` and the
    certified radius

        r_fs = (arccos FT - arccos F0) / s.

    Exact closed form: no sampling, no expansion; identity components
    of the structure (global phase) contribute exactly zero; the
    nominal deficit enters as the angle ``theta_0 = arccos F0``, the
    same mechanism as the angular absorption of the time-bandwidth
    comparison.  Dominance: ``r_fs >= r0`` always, since
    ``theta_T - theta_0 >= (F0 - FT)/sqrt(1 - FT^2)`` and
    ``||Hhatbar||_F <= ||Hhat||_F`` while ``L = B_T C`` charges
    ``sqrt((1 - FT^2)/N) dt sum_k ||Hhat^(k)||_F``.
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
    """``int_0^{t_f} U_S(t)' dHbar(t) U_S(t) dt`` for the nominal evolution.

    The first-order response of the propagator to a *constant* offset in
    one structure is this integral: a coherent sum over the gate in the
    toggling frame, which a well-shaped pulse can make small by
    cancellation between intervals. Its Frobenius norm is therefore the
    quantity a static robustification objective is implicitly driving
    down, and it is not what the free path-length certificates charge --
    they charge ``int ||dHbar(t)|| dt``, which cancellation cannot reduce.
    Measuring both separates "the pulse arranged cancellation" from "the
    pulse used more amplitude".

    Exact for piecewise-constant controls. On interval ``k`` with
    ``H^(k) = V L V'``, the inner integral is a divided difference in the
    eigenbasis: element ``(a, b)`` of ``V' dHbar V`` picks up
    ``int_0^dt exp(1j (L_a - L_b) s) ds``, evaluated as
    ``dt exp(1j X) sinc(X)`` with ``X = dt (L_a - L_b)/2``, which is
    stable because the exponent is purely imaginary and only ``X == 0``
    needs masking.
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
    """Joint trajectory certificate for ``p`` structures via the exact
    Choi speed.

    On interval ``k`` the speed of the combined generator
    ``G = sum_j delta_j Htil_j`` is
    ``sqrt(delta^T Q^(k) delta)`` with the Gram matrix
    ``Q^(k)_ij = Tr(Hhatbar_i^(k) Hhatbar_j^(k)) / N`` (invariant under
    the common conjugation).  For box bounds ``|delta_j(t)| <= m_j``
    the worst case over the box is attained at a sign vertex (convex
    maximisation), enumerated exactly.  Returns
    ``(ell(m), certified)`` as a callable path-length gauge
    ``ell(m) = dt sum_k max_sigma sqrt((sigma m)^T Q^(k) (sigma m))``
    and the certificate: every measurable trajectory with
    ``|delta_j(t)| <= m_j`` and ``ell(m) <= arccos FT - arccos F0``
    keeps ``F >= FT``.
    """
    gauge = PathGauge(
        grams=interval_grams(Hhat_lists, make_traceless=True, normalise=True), dt=dt
    )
    return gauge.C_box, angle_budget(F0, FT)

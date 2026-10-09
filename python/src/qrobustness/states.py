# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""State-fidelity margins for pure states under structured Hamiltonian perturbations.

The module computes the half spread ||X||_c, the state speed C^st and its
joint gauge, and the preparation speed C_prep of a gapped eigenvector. It
propagates a state, returns the fidelity and the Fubini-Study angle, and
evaluates the angular or Lipschitz state margin and the uniform trajectory
radii.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

from .core import Array, HList, MarginResult, iterative_margin

__all__ = [
    "TrajectoryRadius",
    "fs_angle",
    "gapped_preparation_radius",
    "half_spread",
    "nondegenerate_eigenvector",
    "preparation_speed",
    "propagate_state",
    "state_angular_margin",
    "state_fidelity",
    "state_lipschitz_constant",
    "state_speed",
    "state_speed_joint",
    "trajectory_radius",
]


def half_spread(H: Array) -> float:
    """Half spread ||H||_c = (lambda_max - lambda_min)/2 = min_c ||H - c I||_2 of a Hermitian matrix."""
    w = np.linalg.eigvalsh(np.asarray(H))
    return float((w[-1] - w[0]) / 2)


def nondegenerate_eigenvector(H: Array, index: int = 0) -> Tuple[Array, float, float]:
    """Eigenvector of a Hermitian matrix with its eigenvalue and spectral gap.

    Parameters
    ----------
    H : (N, N) array
        Hermitian matrix.
    index : int
        Eigenvalue index in ascending order (0 is the ground state).

    Returns
    -------
    psi : (N,) array
        The eigenvector.
    eigenvalue : float
    gap : float
        Distance gamma to the rest of the spectrum. Raises ValueError if the
        eigenvalue is degenerate to working precision.
    """
    w, V = np.linalg.eigh(np.asarray(H))
    others = np.delete(w, index)
    gap = float(np.min(np.abs(others - w[index]))) if others.size else np.inf
    tol = 64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(w))))
    if gap <= tol:
        raise ValueError("The eigenvalue is degenerate; the gap must be positive")
    return V[:, index], float(w[index]), gap


def preparation_speed(
    dH: Array, psi0: Array, gap: float, bound: str = "sigma"
) -> float:
    """Upper bound on the preparation speed C_prep of a gapped eigenvector.

    Parameters
    ----------
    dH : (N, N) array
        Derivative d H_prep / d mu of the preparation Hamiltonian.
    psi0 : (N,) array
        Prepared nondegenerate eigenvector.
    gap : float
        Spectral gap gamma of psi0; must be positive.
    bound : str
        ``'sigma'``: sigma/gamma with sigma the standard deviation of dH in
        psi0; ``'spread'``: ||dH||_c / gamma (state-independent, not smaller).

    Returns
    -------
    float
        Bound on C_prep. Valid only where the gap is at least ``gap``; an
        interval certificate needs a lower bound on the gap over the interval.
    """
    if not gap > 0:
        raise ValueError("The gap must be positive")
    dH = np.asarray(dH)
    if bound == "spread":
        return float(half_spread(dH) / gap)
    if bound != "sigma":
        raise ValueError("bound must be 'sigma' or 'spread'")
    psi0 = np.asarray(psi0, dtype=complex)
    v = dH @ psi0
    m = np.vdot(psi0, v)
    sigma = float(np.linalg.norm(v - m * psi0))
    return float(sigma / gap)


def gapped_preparation_radius(
    budget: float, gap: float, prep_spread: float, evolution_speed: float = 0.0
) -> float:
    """Certified parameter radius for a gapped, parameter-dependent state preparation.

    Returns the largest r with r (s/(gamma - 2 s r) + C^st) <= budget, where
    the gap is bounded below by gamma - 2 s r over the radius (Weyl); r < gamma/(2 s).

    Parameters
    ----------
    budget : float
        Angle budget arccos F_T - arccos F at the centre.
    gap : float
        Spectral gap gamma of the prepared eigenvector at the centre; positive.
    prep_spread : float
        s = ||d H_prep / d mu||_c.
    evolution_speed : float
        State speed C^st of the subsequent evolution.

    Returns
    -------
    float
        Radius r (0 if budget <= 0, inf if both speeds vanish).
    """
    if not gap > 0:
        raise ValueError("The gap must be positive")
    if budget <= 0:
        return 0.0
    s, C = float(prep_spread), float(evolution_speed)
    if s < 0 or C < 0:
        raise ValueError("Speeds must be non-negative")
    if s == 0 and C == 0:
        return float("inf")
    b = s + C * gap + 2 * s * budget
    disc = max(b * b - 8 * s * C * budget * gap, 0.0)
    return float(2 * budget * gap / (b + np.sqrt(disc)))


def state_speed(Hhat_list: HList, dt: float) -> float:
    """State speed C^st = dt sum_k ||Hhat^(k)||_c of a structure (per unit parameter).

    Parameters
    ----------
    Hhat_list : sequence of (N, N) arrays
        Structure Hhat^(k) per interval.
    dt : float
        Interval length Delta.

    Returns
    -------
    float
        Upper bound on the Fubini-Study speed of any pure state.
    """
    return float(dt * sum(half_spread(H) for H in Hhat_list))


def state_speed_joint(
    Hhat_lists: Sequence[HList], dt: float, x: Sequence[float]
) -> float:
    """Joint state gauge dt sum_k ||sum_j x_j Hhat_j^(k)||_c in direction x.

    Parameters
    ----------
    Hhat_lists : sequence of HList
        One structure list per parameter j.
    dt : float
        Interval length Delta.
    x : (J,) array
        Parameter direction.

    Returns
    -------
    float
        Gauge value, at most sum_j |x_j| C^st_j.
    """
    x = np.asarray(x, dtype=float)
    if len(Hhat_lists) != x.size:
        raise ValueError("Need one direction component per structure")
    total = 0.0
    for k in range(len(Hhat_lists[0])):
        G = sum(x[j] * np.asarray(Hhat_lists[j][k]) for j in range(x.size))
        total += half_spread(G)
    return float(dt * total)


def propagate_state(H_list: HList, dt: float, psi0: Array) -> Array:
    """State after the piecewise-constant evolution prod_k exp(-i H_k dt).

    Parameters
    ----------
    H_list : sequence of (N, N) arrays
        Hermitian Hamiltonians per interval.
    dt : float
        Interval length.
    psi0 : (N,) array
        Initial state.

    Returns
    -------
    (N,) complex array
        Final state.
    """
    psi = np.asarray(psi0, dtype=complex)
    for H in H_list:
        w, V = np.linalg.eigh(np.asarray(H))
        psi = V @ (np.exp(-1j * w * dt) * (V.conj().T @ psi))
    return psi


def state_fidelity(chi: Array, psi: Array) -> float:
    """State fidelity |<chi|psi>| of two normalised states, clipped to at most 1."""
    return float(min(1.0, abs(np.vdot(chi, psi))))


def fs_angle(psi: Array, phi: Array) -> float:
    """Fubini-Study angle arccos |<psi|phi>| between two normalised states.

    Computed with atan2, which keeps full relative accuracy for small angles.
    """
    psi, phi = np.asarray(psi), np.asarray(phi)
    overlap = np.vdot(psi, phi)
    return float(np.arctan2(np.linalg.norm(phi - overlap * psi), abs(overlap)))


def state_lipschitz_constant(FT: float, C: float) -> float:
    """Lipschitz constant sqrt(1 - F_T^2) C of F = |<chi|psi(mu)>| on the safe set F > F_T.

    Parameters
    ----------
    FT : float
        Threshold F_T, 0 < F_T < 1.
    C : float
        State speed C^st.

    Returns
    -------
    float
    """
    if not 0.0 < FT < 1.0:
        raise ValueError("Require 0 < FT < 1")
    return float(np.sqrt(1.0 - FT * FT) * C)


def state_angular_margin(
    fidelity_fn: Callable[[float], float],
    C: float,
    FT: float,
    mu0: float = 0.0,
    angular: bool = True,
    **kwargs,
) -> MarginResult:
    """Certified margin of a state fidelity F(mu) = |<chi|psi(mu)>|.

    Parameters
    ----------
    fidelity_fn : callable
        mu -> F(mu).
    C : float
        State speed C^st (plus C_prep if the preparation depends on mu); positive.
    FT : float
        Threshold F_T.
    mu0 : float
        Nominal parameter value.
    angular : bool
        True: safe radius (arccos F_T - arccos F)/C; False: the Lipschitz
        radius (F - F_T)/(sqrt(1 - F_T^2) C), which it dominates.
    **kwargs
        Passed to ``iterative_margin``.

    Returns
    -------
    MarginResult
        As returned by ``iterative_margin``.
    """
    if C <= 0:
        raise ValueError("Speed constant C must be positive")
    L = state_lipschitz_constant(FT, C)
    radius = None
    if angular:
        theta_T = float(np.arctan2(np.sqrt(max(0.0, 1.0 - FT * FT)), FT))

        def radius(F: float) -> float:
            Fc = min(max(F, 0.0), 1.0)
            theta = float(np.arctan2(np.sqrt(max(0.0, 1.0 - Fc * Fc)), Fc))
            return max(0.0, (theta_T - theta) / C)

    return iterative_margin(
        fidelity_fn, L, FT, mu0=mu0, safe_radius_fn=radius, **kwargs
    )


@dataclass
class TrajectoryRadius:
    """Uniform radius r certified for every measurable trajectory with |mu(t)| <= r.

    Attributes
    ----------
    r : float
        Radius budget/speed.
    budget : float
        Angle budget spent.
    speed : float
        Path length per unit ||mu||_inf (C^st for a state, C_a + C_b for a pair).
    """

    r: float
    budget: float
    speed: float


def trajectory_radius(budget: float, speed: float) -> TrajectoryRadius:
    """Radius budget/speed certified for time-varying perturbations.

    Parameters
    ----------
    budget : float
        Positive angle budget: arccos F_T - arccos F_0 for a state fidelity,
        theta_0 - theta_T for distinguishability.
    speed : float
        Path length per unit ||mu||_inf.

    Returns
    -------
    TrajectoryRadius
        ``r`` is inf when ``speed`` is 0.
    """
    if budget <= 0:
        raise ValueError(
            "The budget must be positive (nominal value beyond the threshold)"
        )
    r = budget / speed if speed > 0 else float("inf")
    return TrajectoryRadius(r=float(r), budget=float(budget), speed=float(speed))

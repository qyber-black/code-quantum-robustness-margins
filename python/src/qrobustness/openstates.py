# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Open-system state-fidelity margins under piecewise-constant Lindblad generators.

The speed constant is D = sum_k Delta_k ||Ghat_k||_diamond, taken from
verified diamond-norm upper bounds. The module evolves a density operator,
computes the trace distance, and certifies the margin of
F(mu) = <chi|rho(mu)|chi> with Lipschitz constant D/2. Superoperators use
the column-stacking convention of ``qrobustness.lindblad``.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .core import Array, MarginResult, iterative_margin
from .lindblad import channel, diamond_norm_free, hamiltonian_dnorm, hamiltonian_part

__all__ = [
    "evolve_density",
    "open_speed",
    "open_state_fidelity_margin",
    "trace_distance",
]


def open_speed(
    Ghat_list: Sequence[Array],
    dt: float | Sequence[float],
    norm: Optional[Callable[[Array], float]] = None,
    exact_hamiltonian: bool = False,
) -> float:
    """Speed constant D = sum_k dt_k ||Ghat_k||_diamond from verified upper bounds.

    Parameters
    ----------
    Ghat_list : sequence of (N^2, N^2) arrays
        Perturbation superoperators Ghat_k, one per interval.
    dt : float or sequence of float
        Interval length(s) dt_k.
    norm : callable, optional
        Maps a superoperator to a certified diamond-norm upper bound; default
        ``diamond_norm_free(G).value_certified``. Identical objects are
        evaluated once.
    exact_hamiltonian : bool
        If True, structures of the form -i[B, .] use the verified closed form
        lambda_max(B) - lambda_min(B) instead of ``norm``.

    Returns
    -------
    float
        Upper bound on D.
    """
    if norm is None:

        def norm(G):
            return float(diamond_norm_free(G).value_certified)

    if exact_hamiltonian:
        inner = norm

        def norm(G):
            if hamiltonian_part(G) is not None:
                return float(hamiltonian_dnorm(G).value_certified)
            return inner(G)

    dts = np.broadcast_to(np.asarray(dt, dtype=float), (len(Ghat_list),))
    cache = {}
    total = 0.0
    for G, step in zip(Ghat_list, dts, strict=True):
        key = id(G)
        if key not in cache:
            cache[key] = norm(np.asarray(G))
        total += step * cache[key]
    return float(total)


def evolve_density(G_list: Sequence[Array], dt: float, rho0: Array) -> Array:
    """Evolve a density operator through the channel prod_k expm(dt G_k).

    Parameters
    ----------
    G_list : sequence of (N^2, N^2) arrays
        Lindblad generators G_k (column stacking), one per interval.
    dt : float
        Interval length.
    rho0 : (N, N) array
        Initial density operator.

    Returns
    -------
    (N, N) complex array
        Final density operator.
    """
    rho0 = np.asarray(rho0, dtype=complex)
    N = rho0.shape[0]
    vec = channel(G_list, dt) @ rho0.reshape(-1, order="F")
    return vec.reshape(N, N, order="F")


def trace_distance(rho: Array, sigma: Array) -> float:
    """Trace distance ||rho - sigma||_1 / 2 of two Hermitian (N, N) arrays, as a float."""
    D = np.asarray(rho) - np.asarray(sigma)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh((D + D.conj().T) / 2))))


def open_state_fidelity_margin(
    fidelity_fn: Callable[[float], float],
    D: float,
    FT: float,
    mu0: float = 0.0,
    omega: Tuple[float, float] = (-np.inf, np.inf),
    **kwargs,
) -> MarginResult:
    """Certified margin of F(mu) = <chi|rho(mu)|chi> via ``iterative_margin`` with L = D/2.

    Parameters
    ----------
    fidelity_fn : callable
        mu -> state fidelity F(mu).
    D : float
        Speed constant from ``open_speed``; must be positive.
    FT : float
        Fidelity threshold F_T.
    mu0 : float
        Nominal parameter value.
    omega : (float, float)
        Parameter domain; must keep every generator a Lindblad generator
        (for a rate offset, mu >= -rate).
    **kwargs
        Passed to ``iterative_margin``.

    Returns
    -------
    MarginResult
        As returned by ``iterative_margin``.
    """
    if D <= 0:
        raise ValueError("Speed constant D must be positive")
    return iterative_margin(fidelity_fn, 0.5 * D, FT, mu0=mu0, omega=omega, **kwargs)

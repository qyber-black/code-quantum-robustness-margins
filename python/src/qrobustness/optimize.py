# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gate-fidelity maximisation for two-control piecewise-constant controllers.

The module packs the control vector, returns the gate fidelity with its
GRAPE gradient, and synthesises a controller by L-BFGS-B from a random or
a given start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from .core import (
    DU_METHODS,
    dU_dmu_exact,
    dU_dmu_integral,
    gauss_legendre_01,
    segment_eig,
    segment_propagator,
    gate_fidelity,
    propagator,
)

Array = np.ndarray


def pack_controls(u1: Array, u2: Array) -> Array:
    """Interleave two control arrays into one vector (u1_1, u2_1, u1_2, u2_2, ...).

    Parameters
    ----------
    u1, u2 : (tau,) arrays
        Control amplitudes per interval.

    Returns
    -------
    (2 tau,) array
        Packed vector, column-major as read by ``load_controllers``.
    """
    u1 = np.asarray(u1, dtype=float).ravel()
    u2 = np.asarray(u2, dtype=float).ravel()
    if u1.size != u2.size:
        raise ValueError("u1 and u2 must have equal length")
    return np.vstack([u1, u2]).reshape(-1, order="F")


def unpack_controls(x: Array, tau: int) -> Tuple[Array, Array]:
    """Split a packed control vector into the two amplitude arrays.

    Parameters
    ----------
    x : (2 tau,) array
        Packed vector as produced by ``pack_controls``.
    tau : int
        Number of intervals.

    Returns
    -------
    u1, u2 : (tau,) arrays
        Control amplitudes per interval.
    """
    u = np.asarray(x, dtype=float).ravel().reshape((2, tau), order="F")
    return u[0].copy(), u[1].copy()


def fidelity_and_gradient(
    H0: Array,
    H1: Array,
    H2: Array,
    u1: Array,
    u2: Array,
    Uf: Array,
    dt: float,
    n_quad: int = 32,
    *,
    method: str = "exact",
) -> Tuple[float, Array, Array]:
    """Gate fidelity F and its gradients with respect to both control arrays.

    Parameters
    ----------
    H0, H1, H2 : (N, N) arrays
        Drift and the two control Hamiltonians.
    u1, u2 : (tau,) arrays
        Control amplitudes per interval.
    Uf : (N, N) array
        Target gate.
    dt : float
        Interval length.
    n_quad : int
        Gauss-Legendre nodes for ``method='quadrature'``; unused for ``'exact'``.
    method : str
        ``'exact'`` (one eigendecomposition per interval) or ``'quadrature'``.

    Returns
    -------
    F : float
        Gate fidelity |tr(Uf^dag U)|/N; must be positive.
    g1, g2 : (tau,) arrays
        dF/du1_k and dF/du2_k.
    """
    if method not in DU_METHODS:
        raise ValueError(f"Unknown method={method!r}; expected one of {DU_METHODS}")
    use_exact = method == "exact"

    u1 = np.asarray(u1, dtype=float).ravel()
    u2 = np.asarray(u2, dtype=float).ravel()
    tau = u1.size
    N = H0.shape[0]

    H_list = [H0 + u1[k] * H1 + u2[k] * H2 for k in range(tau)]
    if use_exact:
        eigs = [segment_eig(H) for H in H_list]
        Useg = [segment_propagator(lam, V, dt) for lam, V in eigs]
    else:
        eigs = []
        Useg = [expm(-1j * dt * H) for H in H_list]

    Pref: list[Array] = [np.eye(N, dtype=complex)]
    for k in range(tau):
        Pref.append(Useg[k] @ Pref[-1])
    Utot = Pref[tau]
    F = gate_fidelity(Utot, Uf)
    if F <= 0:
        raise ValueError("Fidelity must be positive for phase")

    z = np.trace(Uf.conj().T @ Utot)
    e_minus_i_phi = np.conj(z / np.abs(z))

    Suff: list[Array] = [np.empty((N, N), dtype=complex) for _ in range(tau + 1)]
    Suff[tau] = np.eye(N, dtype=complex)
    for k in range(tau - 1, -1, -1):
        Suff[k] = Suff[k + 1] @ Useg[k]

    if not use_exact:
        nodes, weights = gauss_legendre_01(n_quad)

    g1 = np.zeros(tau, dtype=float)
    g2 = np.zeros(tau, dtype=float)
    for k in range(tau):
        if use_exact:
            lam, V = eigs[k]
            dUk1 = dU_dmu_exact(lam, V, H1, dt)
            dUk2 = dU_dmu_exact(lam, V, H2, dt)
        else:
            dUk1 = dU_dmu_integral(H_list[k], H1, dt, nodes, weights)
            dUk2 = dU_dmu_integral(H_list[k], H2, dt, nodes, weights)
        D1 = Suff[k + 1] @ dUk1 @ Pref[k]
        D2 = Suff[k + 1] @ dUk2 @ Pref[k]
        g1[k] = float(np.real(np.trace(Uf.conj().T @ D1 * e_minus_i_phi))) / N
        g2[k] = float(np.real(np.trace(Uf.conj().T @ D2 * e_minus_i_phi))) / N
    return F, g1, g2


@dataclass
class OptimizeResult:
    """Result of one synthesis run.

    Attributes
    ----------
    u1, u2 : (tau,) arrays
        Optimised controls.
    fid, error : float
        Final gate fidelity (clamped to [0, 1]) and 1 - fid.
    fid_init : float
        Fidelity at the starting controls.
    n_iter : int
        Optimiser iterations.
    success, message :
        The optimiser's own termination status and message.
    """

    u1: Array
    u2: Array
    fid: float
    error: float
    fid_init: float
    n_iter: int
    success: bool
    message: str


def optimize_controller(
    H0: Array,
    H1: Array,
    H2: Array,
    Uf: Array,
    tf: float,
    tau: int,
    u1_init: Optional[Array] = None,
    u2_init: Optional[Array] = None,
    sigma: float = 1.0,
    seed: Optional[int] = None,
    n_quad: int = 32,
    method: str = "exact",
    maxiter: int = 500,
    ftol: float = 1e-12,
) -> OptimizeResult:
    """Maximise the gate fidelity by L-BFGS-B, using the analytic GRAPE gradient.

    Parameters
    ----------
    H0, H1, H2 : (N, N) arrays
        Drift and the two control Hamiltonians.
    Uf : (N, N) array
        Target gate.
    tf : float
        Gate time T; the interval length is tf/tau.
    tau : int
        Number of intervals.
    u1_init, u2_init : (tau,) arrays, optional
        Starting controls; drawn from N(0, sigma^2) when omitted.
    sigma : float
        Standard deviation of random starting controls.
    seed : int, optional
        Seed for the random start.
    n_quad, method :
        Passed to ``fidelity_and_gradient``.
    maxiter, ftol :
        L-BFGS-B options.

    Returns
    -------
    OptimizeResult
    """
    dt = tf / tau
    rng = np.random.default_rng(seed)
    if u1_init is None:
        u1_init = rng.normal(0.0, sigma, size=tau)
    if u2_init is None:
        u2_init = rng.normal(0.0, sigma, size=tau)
    u1_init = np.asarray(u1_init, dtype=float).ravel()
    u2_init = np.asarray(u2_init, dtype=float).ravel()
    if u1_init.size != tau or u2_init.size != tau:
        raise ValueError("u1_init/u2_init length must equal tau")

    H_list0 = [H0 + u1_init[k] * H1 + u2_init[k] * H2 for k in range(tau)]
    fid_init = gate_fidelity(propagator(H_list0, dt), Uf)

    x0 = pack_controls(u1_init, u2_init)

    def fun(x: Array) -> Tuple[float, Array]:
        u1, u2 = unpack_controls(x, tau)
        F, g1, g2 = fidelity_and_gradient(
            H0, H1, H2, u1, u2, Uf, dt, n_quad=n_quad, method=method
        )
        return 1.0 - F, -pack_controls(g1, g2)

    res = minimize(
        fun,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": ftol},
    )
    u1, u2 = unpack_controls(res.x, tau)
    # Clamp: roundoff can drive F slightly above 1 (negative error).
    fid = float(min(1.0, max(0.0, 1.0 - float(res.fun))))
    return OptimizeResult(
        u1=u1,
        u2=u2,
        fid=fid,
        error=max(0.0, 1.0 - fid),
        fid_init=fid_init,
        n_iter=int(res.nit),
        success=bool(res.success),
        message=str(res.message),
    )

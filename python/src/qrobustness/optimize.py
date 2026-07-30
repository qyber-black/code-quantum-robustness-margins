# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fidelity-maximising controller synthesis (GRAPE + quasi-Newton)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from .core import (
    DU_METHODS,
    _dU_dmu_exact,
    _dU_dmu_integral,
    _gauss_legendre_01,
    _segment_eig,
    _segment_propagator,
    gate_fidelity,
    propagator,
)

Array = np.ndarray


def pack_controls(u1: Array, u2: Array) -> Array:
    """Interleave (u1_k, u2_k) to match CSV / load_controllers Fortran reshape."""
    u1 = np.asarray(u1, dtype=float).ravel()
    u2 = np.asarray(u2, dtype=float).ravel()
    if u1.size != u2.size:
        raise ValueError("u1 and u2 must have equal length")
    return np.vstack([u1, u2]).reshape(-1, order="F")


def unpack_controls(x: Array, tau: int) -> Tuple[Array, Array]:
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
    r"""Gate fidelity and GRAPE gradients \partial F/\partial u_1, \partial F/\partial u_2.

    method='exact' (default) uses one eigendecomposition per interval for the
    propagator and both control derivatives; 'quadrature' uses Gauss-Legendre
    with n_quad nodes.  n_quad is inert under 'exact'.
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
        eigs = [_segment_eig(H) for H in H_list]
        Useg = [_segment_propagator(lam, V, dt) for lam, V in eigs]
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
        nodes, weights = _gauss_legendre_01(n_quad)

    g1 = np.zeros(tau, dtype=float)
    g2 = np.zeros(tau, dtype=float)
    for k in range(tau):
        if use_exact:
            lam, V = eigs[k]
            dUk1 = _dU_dmu_exact(lam, V, H1, dt)
            dUk2 = _dU_dmu_exact(lam, V, H2, dt)
        else:
            dUk1 = _dU_dmu_integral(H_list[k], H1, dt, nodes, weights)
            dUk2 = _dU_dmu_integral(H_list[k], H2, dt, nodes, weights)
        D1 = Suff[k + 1] @ dUk1 @ Pref[k]
        D2 = Suff[k + 1] @ dUk2 @ Pref[k]
        g1[k] = float(np.real(np.trace(Uf.conj().T @ D1 * e_minus_i_phi))) / N
        g2[k] = float(np.real(np.trace(Uf.conj().T @ D2 * e_minus_i_phi))) / N
    return F, g1, g2


@dataclass
class OptimizeResult:
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
    """Maximize gate fidelity via L-BFGS-B with analytic GRAPE gradient."""
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
        # minimise error = 1 - F
        return 1.0 - F, -pack_controls(g1, g2)

    res = minimize(
        fun,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": ftol},
    )
    u1, u2 = unpack_controls(res.x, tau)
    # Clamp: roundoff can push F slightly above 1 (negative error).
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

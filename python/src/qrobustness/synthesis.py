# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GRAPE controller synthesis with exact interval derivatives.

The module returns the gate fidelity and its exact control gradient for any
number of controls. Single-controller and ensemble synthesis are seeded
L-BFGS-B minimisations of 1 - F. Ensemble-robust synthesis samples
multiplicative perturbations. The seeds reproduce the results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .core import (
    Array,
    dU_dmu_exact,
    segment_eig,
    segment_propagator,
    gate_fidelity,
)

#: Gradient tolerance of the L-BFGS-B stopping rule (fixed for reproducibility).
GTOL = 1e-12

__all__ = [
    "fidelity_and_control_gradient",
    "GrapeResult",
    "grape",
    "grape_ensemble",
    "grape_robust",
]


def fidelity_and_control_gradient(
    H0: Array,
    H_ctrls: Sequence[Array],
    u: Array,
    dt: float,
    Uf: Array,
) -> Tuple[float, Array]:
    """Gate fidelity and its exact gradient in the controls.

    Parameters
    ----------
    H0 : (N, N) array
        Drift Hamiltonian.
    H_ctrls : sequence of (N, N) arrays
        Control Hamiltonians, one per control j.
    u : (n_ctrl, tau) array
        Controls; interval k evolves under H0 + sum_j u[j, k] H_ctrls[j].
    dt : float
        Interval length.
    Uf : (N, N) array
        Target gate.

    Returns
    -------
    F : float
        Gate fidelity |tr(Uf^dag U)|/N.
    dF_du : (n_ctrl, tau) array
        Gradient; zero where tr(Uf^dag U) = 0 (gradient undefined).
    """
    u = np.asarray(u, dtype=float)
    n_ctrl, tau = u.shape
    if len(H_ctrls) != n_ctrl:
        raise ValueError("u rows must match H_ctrls")
    N = H0.shape[0]

    eigs = []
    Pref: List[Array] = [np.eye(N, dtype=complex)]
    for k in range(tau):
        H = H0 + sum(u[j, k] * H_ctrls[j] for j in range(n_ctrl))
        lam, V = segment_eig(H)
        eigs.append((lam, V))
        Pref.append(segment_propagator(lam, V, dt) @ Pref[-1])
    Utot = Pref[tau]
    F = gate_fidelity(Utot, Uf)
    z = np.trace(Uf.conj().T @ Utot)
    if np.abs(z) == 0.0:
        # |z| is not differentiable at z = 0.
        return F, np.zeros_like(u)
    e_minus_i_phi = np.conj(z / np.abs(z))

    Suff: List[Array] = [np.eye(N, dtype=complex) for _ in range(tau + 1)]
    for k in range(tau - 1, -1, -1):
        lam, V = eigs[k]
        Suff[k] = Suff[k + 1] @ segment_propagator(lam, V, dt)

    g = np.zeros_like(u)
    for k in range(tau):
        lam, V = eigs[k]
        for j in range(n_ctrl):
            dUk = dU_dmu_exact(lam, V, H_ctrls[j], dt)
            Dk = Suff[k + 1] @ dUk @ Pref[k]
            g[j, k] = np.real(np.trace(Uf.conj().T @ Dk) * e_minus_i_phi) / N
    return F, g


@dataclass
class GrapeResult:
    """Synthesised controller.

    Attributes
    ----------
    u : (n_ctrl, tau) array
        Controls.
    fidelity, error : float
        Nominal gate fidelity F_0 and eps_0 = 1 - F_0.
    n_iter : int
        Optimiser iterations.
    seed : int
        Seed of the random initialisation.
    converged : bool
        The optimiser's success flag.
    """

    u: Array
    fidelity: float
    error: float
    n_iter: int
    seed: int
    converged: bool


def grape(
    H0: Array,
    H_ctrls: Sequence[Array],
    Uf: Array,
    tf: float,
    tau: int,
    seed: int,
    maxiter: int = 500,
    ftol: float = 1e-12,
    u0_scale: float = 1.0,
) -> GrapeResult:
    """Synthesise one controller by L-BFGS-B from a seeded random start.

    Parameters
    ----------
    H0 : (N, N) array
        Drift Hamiltonian.
    H_ctrls : sequence of (N, N) arrays
        Control Hamiltonians.
    Uf : (N, N) array
        Target gate.
    tf : float
        Gate time T; the interval length is tf/tau.
    tau : int
        Number of intervals.
    seed : int
        Seed for the standard-normal initial controls.
    maxiter, ftol :
        L-BFGS-B options.
    u0_scale : float
        Scale of the initial controls.

    Returns
    -------
    GrapeResult
    """
    rng = np.random.default_rng(seed)
    dt = tf / tau
    n_ctrl = len(H_ctrls)
    u0 = u0_scale * rng.standard_normal((n_ctrl, tau))

    def objective(x):
        F, g = fidelity_and_control_gradient(
            H0, H_ctrls, x.reshape(n_ctrl, tau), dt, Uf
        )
        return 1.0 - F, -g.ravel()

    res = minimize(
        objective,
        u0.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": ftol, "gtol": GTOL},
    )
    u = res.x.reshape(n_ctrl, tau)
    F = 1.0 - float(res.fun)
    return GrapeResult(
        u=u,
        fidelity=F,
        error=1.0 - F,
        n_iter=int(res.nit),
        seed=seed,
        converged=bool(res.success),
    )


def grape_ensemble(
    H0: Array,
    H_ctrls: Sequence[Array],
    Uf: Array,
    tf: float,
    tau: int,
    n_attempts: int = 100,
    max_error: float = 1e-4,
    seed0: int = 0,
    maxiter: int = 500,
    verbose: bool = False,
) -> List[GrapeResult]:
    """Synthesise an ensemble, keeping controllers with eps_0 <= max_error.

    Parameters
    ----------
    H0, H_ctrls, Uf, tf, tau, maxiter :
        As for ``grape``.
    n_attempts : int
        Number of runs; run i uses seed seed0 + i.
    max_error : float
        Largest nominal error eps_0 kept.
    seed0 : int
        First seed.
    verbose : bool
        Print one line per attempt.

    Returns
    -------
    list of GrapeResult
        Kept controllers in seed order.
    """
    kept: List[GrapeResult] = []
    for i in range(n_attempts):
        r = grape(H0, H_ctrls, Uf, tf, tau, seed=seed0 + i, maxiter=maxiter)
        if r.error <= max_error:
            kept.append(r)
        if verbose:
            print(
                f"attempt {i + 1}/{n_attempts}: eps0={r.error:.3e} "
                f"{'kept' if r.error <= max_error else 'dropped'}",
                flush=True,
            )
    return kept


def grape_robust(
    H0: Array,
    H_ctrls: Sequence[Array],
    Uf: Array,
    tf: float,
    tau: int,
    seed: int,
    sample_deltas: Sequence[Sequence[float]],
    maxiter: int = 500,
    ftol: float = 1e-12,
    u0_scale: float = 1.0,
) -> GrapeResult:
    """Synthesise one controller maximising the mean fidelity over perturbation samples.

    Parameters
    ----------
    H0, H_ctrls, Uf, tf, tau, seed, maxiter, ftol, u0_scale :
        As for ``grape``.
    sample_deltas : sequence of (1 + n_ctrl,) arrays
        Samples (d_0, d_1, ..., d_nc) applied as H0 (1 + d_0) and
        H_ctrls[j] (1 + d_j).

    Returns
    -------
    GrapeResult
        ``fidelity`` and ``error`` are for the nominal (unperturbed) system.
    """
    rng = np.random.default_rng(seed)
    dt = tf / tau
    n_ctrl = len(H_ctrls)
    u0 = u0_scale * rng.standard_normal((n_ctrl, tau))
    samples = [np.asarray(s, dtype=float) for s in sample_deltas]

    def objective(x):
        u = x.reshape(n_ctrl, tau)
        tot, g = 0.0, np.zeros_like(u)
        for d in samples:
            H0s = H0 * (1.0 + d[0])
            scales = 1.0 + d[1 : 1 + n_ctrl]
            Hs = [scales[j] * H_ctrls[j] for j in range(n_ctrl)]
            F, gs = fidelity_and_control_gradient(H0s, Hs, u, dt, Uf)
            tot += F
            g += gs
        n = len(samples)
        return 1.0 - tot / n, -(g / n).ravel()

    res = minimize(
        objective,
        u0.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": ftol, "gtol": GTOL},
    )
    u = res.x.reshape(n_ctrl, tau)
    F_nom, _ = fidelity_and_control_gradient(H0, H_ctrls, u, dt, Uf)
    return GrapeResult(
        u=u,
        fidelity=float(F_nom),
        error=1.0 - float(F_nom),
        n_iter=int(res.nit),
        seed=seed,
        converged=bool(res.success),
    )

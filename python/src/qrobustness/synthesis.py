# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Controller synthesis by GRAPE with exact gradients.

Synthesises piecewise-constant controllers for the gate-fidelity
objective of this package, so that margin case studies on systems other
than the shipped controller set are reproducible from the repository
alone.  The gradient of the fidelity with respect to each control
amplitude is evaluated with the same closed-form interval derivative
used throughout the package (one Hermitian eigendecomposition per
interval; see ``core.dU_dmu_exact``), so synthesis introduces no new
numerics.

The optimiser is L-BFGS-B on ``1 - F`` from independent standard-normal
initialisations with deterministic seeds; ensembles are therefore
byte-reproducible.  This mirrors the protocol used to produce the
shipped controller set (500 iterations, fidelity tolerance ``1e-12``,
keep controllers with nominal error ``eps_0 <= 1e-4``).
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

#: Gradient tolerance for the L-BFGS-B stopping rule. Not a parameter, unlike
#: maxiter and ftol, because the synthesised ensembles must be reproducible
#: from the seed alone; it is named here so the stopping rule is visible
#: rather than sitting twice as a literal in an options dict.
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
    """Gate fidelity and its exact gradient with respect to the controls.

    ``u`` has shape ``(n_ctrl, tau)``; interval ``k`` evolves under
    ``H0 + sum_j u[j, k] H_ctrls[j]`` for time ``dt``.  Returns
    ``(F, dF/du)`` with ``dF/du`` of the same shape as ``u``.
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
        # Gradient of |z| is undefined at z = 0; return a zero gradient
        # (a measure-zero event under random initialisation).
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
    """A synthesised controller: controls ``u``, fidelity and metadata."""

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
    """Synthesise one controller from a seeded random initialisation."""
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
    """Synthesise an ensemble; keep controllers with ``eps_0 <= max_error``.

    Attempt ``i`` uses seed ``seed0 + i``, so the ensemble is
    reproducible from ``(seed0, n_attempts)`` alone.
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
    """Ensemble-robust GRAPE: maximise the average fidelity over sampled
    structured perturbations.

    ``sample_deltas`` is a list of perturbation samples; each sample is a
    vector ``(d_0, d_1, ..., d_nc)`` of multiplicative perturbations
    applied as ``H0 (1 + d_0)`` and control amplitudes ``u_j (1 + d_j)``
    (the structures of the margin analysis). The objective is
    ``1 - mean_s F_s``; gradients are averaged sample gradients, exact.
    The reported ``fidelity``/``error`` refer to the NOMINAL (unperturbed)
    controller, so results are directly comparable to :func:`grape`.
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

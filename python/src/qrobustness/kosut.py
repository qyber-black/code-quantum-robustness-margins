# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kosut-Lidar-Rabitz time-bandwidth bound (arXiv:2507.01215, Theorem 1).

The bound is specialised to the closed-system scalar structured model
H_unc = delta Hhat. The module returns the per-unit rates w_unc, w_avg and
w_dev, the fidelity lower bound F_lb, and the implied margin M^K for a
constant delta, or M^K_tv for trajectories with |delta(t)| <= m. The nominal
error is absorbed angularly. M^K is not a sup-norm time-varying margin.
Use ``uncertainty='trajectory'`` for that class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

Array = np.ndarray
HList = Sequence[Array]

#: Largest ``T*Omega_bnd`` for which ``F_lb > 0``.
T_OMEGA_MAX = 2.0 * np.sqrt(np.log(1.0 + np.sqrt(2.0)))


@dataclass
class UncertaintyRates:
    """Per-unit-delta uncertainty measures w_unc, w_avg, w_dev of a controller.

    Attributes
    ----------
    w_unc, w_avg, w_dev :
        ``Omega_unc``, ``Omega_avg``, ``Omega_avg^dev`` at constant ``delta = 1``.
    T :
        Gate duration ``t_f``.
    w_avg_traj, w_dev_traj :
        Upper bounds on ``Omega_avg``, ``Omega_avg^dev`` over trajectories
        ``|delta(t)| <= 1``: ``mean_k ||Hhat^(k)||`` and ``w_unc + w_avg_traj``.
    w_dev_certified :
        Rigorous upper bound on ``Omega_avg^dev``; the true value lies in
        ``[w_dev, w_dev_certified]`` (sampling under-estimates a supremum).
    w_dev_refinement :
        Relative increase of ``w_dev`` from refinement over the seed grid.
    w_dev_bracket_lo, w_dev_bracket_hi :
        Independent rigorous bracket on ``Omega_avg^dev`` from isospectrality.
    n_dev_used :
        Grid points per interval at the finest sweep.
    dev_converged :
        Whether successive refined sweeps agreed to ``dev_tol``.
    dev_cycles_max :
        Largest number of cycles of the fastest Bohr frequency per interval.
    dev_samples_per_cycle :
        Samples per cycle achieved at the finest sweep.
    dev_resolved :
        False when ``n_dev_max`` caps the grid that was actually used
        below the requested samples per cycle; ``w_dev`` may then be
        under-estimated.
    """

    w_unc: float
    w_avg: float
    w_dev: float
    T: float
    w_avg_traj: float = float("nan")
    w_dev_traj: float = float("nan")
    w_dev_certified: float = float("nan")
    w_dev_refinement: float = float("nan")
    w_dev_bracket_lo: float = float("nan")
    w_dev_bracket_hi: float = float("nan")
    n_dev_used: int = 0
    dev_converged: bool = False
    dev_cycles_max: float = float("nan")
    dev_samples_per_cycle: float = float("nan")
    dev_resolved: bool = False

    def time_bandwidth(self, delta: float) -> float:
        """``T*Omega_bnd`` at constant perturbation ``delta``."""
        return time_bandwidth(self, delta)


def _spectral_norm(A: Array) -> float:
    return float(np.linalg.norm(A, 2))


def _time_average_htil(lam: Array, V: Array, dH: Array, dt: float) -> Array:
    """Exact ``int_0^dt exp(+i H s) dH exp(-i H s) ds`` from the eigenpairs of ``H``.

    Entry ``(m, n)`` is scaled by ``dt exp(i Y) sin(Y)/Y``,
    ``Y = dt (lam_m - lam_n)/2``; the form has no cancellation, so only
    ``Y == 0`` is masked.
    """
    Y = 0.5 * dt * (lam[:, None] - lam[None, :])
    zero = Y == 0.0
    S = np.where(zero, 1.0, np.sin(Y) / np.where(zero, 1.0, Y))
    W = dt * np.exp(1j * Y) * S
    inner = (V.conj().T @ dH @ V) * W
    return V @ inner @ V.conj().T


def uncertainty_rates(
    H_list: HList,
    dH_list: HList,
    dt: float,
    n_quad: int | None = None,
    n_dev: int = 17,
    dev_tol: float = 1e-9,
    n_dev_max: int = 4097,
    adaptive_dev: bool = True,
    dev_samples_per_cycle: float = 16.0,
) -> UncertaintyRates:
    """Uncertainty rates w_unc, w_avg, w_dev of a piecewise-constant controller.

    ``w_unc`` is exact and ``w_avg`` exact to roundoff (closed-form time
    average). ``w_dev`` is a sampled supremum and can only be under-estimated;
    ``w_dev_certified`` and ``w_dev_bracket_*`` bound it rigorously.

    Parameters
    ----------
    H_list :
        Nominal per-interval Hamiltonians ``H^(k)``, each ``(N, N)``.
    dH_list :
        Per-interval structures ``Hhat^(k)``, so ``H_unc^(k) = delta * dH_list[k]``.
    dt :
        Interval length ``Delta``.
    n_quad :
        Accepted and ignored.
    n_dev :
        Minimum grid points per interval (endpoints included) for the
        supremum defining ``Omega_avg^dev``.
    dev_tol :
        Relative tolerance between successive refined sweeps.
    n_dev_max :
        Cap on grid points per interval.
    adaptive_dev :
        If False, sample once on the seed grid without refinement or polish.
    dev_samples_per_cycle :
        Requested samples per cycle of the fastest Bohr frequency of ``H^(k)``.

    Returns
    -------
    UncertaintyRates
        The rates, ``T = len(H_list) * dt``, and the error-control fields.
    """
    tau = len(H_list)
    if tau == 0:
        raise ValueError("H_list must be non-empty")
    if len(dH_list) != tau:
        raise ValueError("H_list and dH_list must have equal length")
    if dt <= 0:
        raise ValueError("dt must be positive")

    N = H_list[0].shape[0]
    T = tau * dt

    # Per-interval eigendecomposition of the nominal Hamiltonian and the
    # left-accumulated propagator P_{k-1}.
    eigs: list[tuple[Array, Array]] = []
    Pref: list[Array] = [np.eye(N, dtype=complex)]
    for k in range(tau):
        lam, V = np.linalg.eigh(H_list[k])
        eigs.append((lam, V))
        phase = np.exp(-1j * dt * lam)
        Useg = (V * phase) @ V.conj().T
        Pref.append(Useg @ Pref[-1])

    def Htil(k: int, s: float) -> Array:
        """Interaction-picture ``Htil`` at time ``t_{k-1} + s``, per unit delta."""
        lam, V = eigs[k]
        phase = np.exp(1j * s * lam)
        E = (V * phase) @ V.conj().T  # exp(+i H^(k) s)
        inner = E @ dH_list[k] @ E.conj().T
        P = Pref[k]
        return P.conj().T @ inner @ P

    # Omega_unc: H_unc is piecewise constant, so the sup is over intervals.
    norms_dH = [_spectral_norm(np.asarray(dH)) for dH in dH_list]
    w_unc = max(norms_dH)

    # <Htil> = (1/T) sum_k int_0^dt Htil(k, s) ds, in closed form per interval.
    acc = np.zeros((N, N), dtype=complex)
    for k in range(tau):
        lam, V = eigs[k]
        M = _time_average_htil(lam, V, np.asarray(dH_list[k]), dt)
        P = Pref[k]
        acc += P.conj().T @ M @ P
    Havg = acc / T
    w_avg = _spectral_norm(Havg)

    # Omega_avg^dev = sup_t ||Htil(t) - <Htil>||: grid search, then Brent polish.
    def f(k: int, s: float) -> float:
        return _spectral_norm(Htil(k, s) - Havg)

    # Seed grid from the fastest Bohr frequency of H^(k) (cycles per interval).
    cycles = [float(np.ptp(lam)) * dt / (2.0 * np.pi) for lam, _ in eigs]
    n_seed = [
        min(
            max(int(n_dev), 3, int(np.ceil(dev_samples_per_cycle * ck)) + 1),
            int(n_dev_max),
        )
        for ck in cycles
    ]

    def sweep(scale: int, polish: bool) -> tuple[float, float]:
        """Return (best sampled or polished value, Lipschitz shortfall of the grid)."""
        best = 0.0
        gap = 0.0
        for k in range(tau):
            n_grid = min(scale * (n_seed[k] - 1) + 1, int(n_dev_max))
            grid = np.linspace(0.0, dt, n_grid)
            vals = [f(k, s) for s in grid]
            local_best = max(vals)

            # f is Lipschitz in s with L_s = 2 ||H|| ||Hhat||; shortfall <= L_s h / 2.
            L_s = 2.0 * _spectral_norm(np.asarray(H_list[k])) * norms_dH[k]
            gap = max(gap, 0.5 * L_s * dt / (n_grid - 1))

            best_k = local_best
            if polish:
                for i, v in enumerate(vals):
                    interior = (
                        0 < i < n_grid - 1 and v >= vals[i - 1] and v >= vals[i + 1]
                    )
                    if not (interior or v >= local_best):
                        continue
                    a = grid[max(i - 1, 0)]
                    b = grid[min(i + 1, n_grid - 1)]
                    if b <= a:
                        continue
                    res = minimize_scalar(
                        lambda s, k=k: -f(k, s),
                        bounds=(a, b),
                        method="bounded",
                        options={"xatol": 1e-15},
                    )
                    best_k = max(best_k, float(-res.fun))
            best = max(best, best_k)
        return best, gap

    w_dev_sampled, lipschitz_gap = sweep(1, polish=False)

    if not adaptive_dev:
        w_dev = w_dev_sampled
        refinement = 0.0
        dev_converged = False
        scale = 1
    else:
        # Double the grid until successive polished sweeps agree to dev_tol.
        scale = 1
        w_dev, _ = sweep(scale, polish=True)
        dev_converged = False
        while max(n_seed) * scale < n_dev_max:
            scale *= 2
            w_next, _ = sweep(scale, polish=True)
            change = abs(w_next - w_dev) / max(w_next, 1e-300)
            w_dev = max(w_dev, w_next)
            if change <= dev_tol:
                dev_converged = True
                break
        refinement = (w_dev - w_dev_sampled) / w_dev if w_dev > 0 else 0.0
    n_used = min(scale * (max(n_seed) - 1) + 1, int(n_dev_max))
    # Samples per cycle on the grid that was actually used, after the cap.
    final_achieved = [
        (min(scale * (n - 1) + 1, int(n_dev_max)) - 1) / ck if ck > 0 else float("inf")
        for n, ck in zip(n_seed, cycles, strict=True)
    ]
    samples_used = float(min(final_achieved)) if final_achieved else float("inf")
    dev_resolved = bool(samples_used >= dev_samples_per_cycle)

    # Isospectrality: ||Htil(t)|| = ||Hhat^(k)||, which brackets the deviation norm.
    bracket_lo = max(0.0, max(n - w_avg for n in norms_dH))
    bracket_hi = w_unc + w_avg

    # Trajectory class |delta(t)| <= 1: ||<delta Htil>|| <= mean_k ||Hhat^(k)||.
    w_avg_traj = float(np.mean(norms_dH))

    return UncertaintyRates(
        w_unc=float(w_unc),
        w_avg=float(w_avg),
        w_dev=float(w_dev),
        T=float(T),
        w_avg_traj=w_avg_traj,
        w_dev_traj=float(w_unc) + w_avg_traj,
        w_dev_certified=float(w_dev_sampled + lipschitz_gap),
        w_dev_refinement=float(refinement),
        w_dev_bracket_lo=float(bracket_lo),
        w_dev_bracket_hi=float(bracket_hi),
        n_dev_used=int(n_used),
        dev_converged=dev_converged,
        dev_cycles_max=float(max(cycles)) if cycles else 0.0,
        dev_samples_per_cycle=samples_used,
        dev_resolved=dev_resolved,
    )


UNCERTAINTIES = ("constant", "trajectory")


def _select_rates(rates: UncertaintyRates, uncertainty: str) -> tuple[float, float]:
    """``(w_avg, w_dev)`` for ``'constant'`` or ``'trajectory'`` uncertainty."""
    if uncertainty not in UNCERTAINTIES:
        raise ValueError(f"uncertainty must be one of {UNCERTAINTIES}")
    if uncertainty == "constant":
        return rates.w_avg, rates.w_dev
    if not (np.isfinite(rates.w_avg_traj) and np.isfinite(rates.w_dev_traj)):
        raise ValueError("trajectory rates unavailable; recompute uncertainty_rates()")
    return rates.w_avg_traj, rates.w_dev_traj


def time_bandwidth(
    rates: UncertaintyRates, delta: float, uncertainty: str = "constant"
) -> float:
    """Time-bandwidth product ``T*Omega_bnd`` at perturbation size ``delta``.

    Parameters
    ----------
    rates :
        Output of :func:`uncertainty_rates`.
    delta :
        Perturbation size (constant value, or sup-norm for ``'trajectory'``).
    uncertainty :
        ``'constant'`` or ``'trajectory'``.

    Returns
    -------
    float
        ``sqrt(T^2 w_unc w_dev delta^2 + 4 T w_avg |delta|)``.
    """
    w_avg, w_dev = _select_rates(rates, uncertainty)
    d = abs(float(delta))
    a = rates.T**2 * rates.w_unc * w_dev
    b = 4.0 * rates.T * w_avg
    return float(np.sqrt(a * d * d + b * d))


def fidelity_bound(T_omega_bnd: float) -> float:
    """Fidelity lower bound ``F_lb`` for a given ``T*Omega_bnd``.

    Parameters
    ----------
    T_omega_bnd :
        Non-negative time-bandwidth product.

    Returns
    -------
    float
        ``F_lb`` in ``[0, 1]``; ``0`` from :data:`T_OMEGA_MAX` on.
    """
    y = float(T_omega_bnd)
    if y < 0:
        raise ValueError("T_omega_bnd must be non-negative")
    if y >= T_OMEGA_MAX:
        return 0.0
    return float(max(1.0 - 0.5 * (np.exp((y / 2.0) ** 2) - 1.0) ** 2, 0.0))


def fidelity_bound_at(
    rates: UncertaintyRates, delta: float, uncertainty: str = "constant"
) -> float:
    """Fidelity lower bound ``F_lb`` at perturbation size ``delta``.

    Parameters and ``uncertainty`` as in :func:`time_bandwidth`.
    """
    return fidelity_bound(time_bandwidth(rates, delta, uncertainty))


ABSORPTIONS = ("angular", "additive")


def effective_threshold(
    FT: float, nominal_error: float = 0.0, absorption: str = "angular"
) -> float:
    """Threshold on the achieved-gate fidelity implied by ``F_T`` on the target.

    The bound refers to the achieved nominal gate, the margin to the target.
    ``'angular'`` gives ``cos(arccos F_T - arccos(1 - eps_0))``, which is
    sufficient; ``'additive'`` gives ``F_T + eps_0``, which is not.

    Parameters
    ----------
    FT :
        Fidelity threshold ``F_T`` in ``(0, 1)``.
    nominal_error :
        Nominal error ``eps_0 = 1 - F_0`` in ``[0, 1]``.
    absorption :
        ``'angular'`` (default) or ``'additive'``.

    Returns
    -------
    float
        Effective threshold ``F_eff``; ``1.0`` when the nominal angle
        exhausts the budget (nothing is certifiable).
    """
    if not (0.0 < FT < 1.0):
        raise ValueError("FT must satisfy 0 < FT < 1")
    if not (0.0 <= nominal_error <= 1.0):
        raise ValueError("nominal_error must satisfy 0 <= nominal_error <= 1")
    if absorption not in ABSORPTIONS:
        raise ValueError(
            f"Unknown absorption={absorption!r}; expected one of {ABSORPTIONS}"
        )
    if absorption == "additive":
        return float(min(FT + nominal_error, 1.0))
    theta_T = np.arccos(FT)
    theta_nom = np.arccos(1.0 - nominal_error)
    if theta_nom >= theta_T:
        return 1.0
    return float(np.cos(theta_T - theta_nom))


def threshold_time_bandwidth(
    FT: float, nominal_error: float = 0.0, absorption: str = "angular"
) -> float:
    """``T*Omega_bnd`` at which ``F_lb`` equals ``F_eff``.

    Parameters
    ----------
    FT, nominal_error, absorption :
        As in :func:`effective_threshold`.

    Returns
    -------
    float
        ``2 sqrt(ln(1 + sqrt(2 (1 - F_eff))))``; ``0.0`` when ``F_eff >= 1``.
    """
    F_eff = effective_threshold(FT, nominal_error, absorption)
    eps = 1.0 - F_eff
    if eps <= 0.0:
        return 0.0
    return float(2.0 * np.sqrt(np.log(1.0 + np.sqrt(2.0 * eps))))


def margin(
    rates: UncertaintyRates,
    FT: float,
    nominal_error: float = 0.0,
    absorption: str = "angular",
    uncertainty: str = "constant",
) -> float:
    """Perturbation margin M^K (or M^K_tv) implied by the bound.

    ``'constant'`` certifies constant ``|delta| <= M^K`` only; it is not a
    sup-norm time-varying margin. ``'trajectory'`` certifies every measurable
    ``|delta(t)| <= M^K_tv``.

    Parameters
    ----------
    rates :
        Output of :func:`uncertainty_rates`.
    FT, nominal_error, absorption :
        As in :func:`effective_threshold`.
    uncertainty :
        ``'constant'`` (default) or ``'trajectory'``.

    Returns
    -------
    float
        Largest ``|delta|`` with ``F_lb >= F_eff``; ``0.0`` if none, ``inf``
        if the structure does not enter the bound.
    """
    w_avg, w_dev = _select_rates(rates, uncertainty)
    y = threshold_time_bandwidth(FT, nominal_error, absorption)
    if y <= 0.0:
        return 0.0
    a = rates.T**2 * rates.w_unc * w_dev
    b = 4.0 * rates.T * w_avg
    y2 = y * y
    if a <= 0.0 and b <= 0.0:
        return float("inf")
    # Rationalised positive root of a m^2 + b m = y2; stable also when a ~ 0.
    return float(2.0 * y2 / (b + np.sqrt(b * b + 4.0 * a * y2)))

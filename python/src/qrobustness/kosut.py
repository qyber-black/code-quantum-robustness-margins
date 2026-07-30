# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kosut-Lidar-Rabitz (arXiv:2507.01215) time-bandwidth bound, specialised.

Implementation of Theorem 1 of [Kosut, Lidar, Rabitz, *A Fundamental Bound for
Robust Quantum Gate Control*, arXiv:2507.01215v2], specialised to the
closed-system, purely coherent, scalar structured perturbation model of this
package, so that the perturbation margin it implies can be compared with the
Lipschitz margin of :func:`qrobustness.core.iterative_margin`.

Supplementary and experimental: outside the reproduction gate
``make check-margins``, and no claim in the accompanying paper depends on it.
``docs/time-bandwidth-bound.md`` gives the specialisation, the caveats, the
numerical accuracy and the results.

With ``U_S(t)`` the nominal propagator, ``Htil(t) = U_S(t)^dag H_unc(t) U_S(t)``
the interaction-picture uncertainty, ``<A> = (1/T) int_0^T A(t) dt`` the time
average and ``H_unc^(k) = delta * Hhat^(k)``, the three measures of Eq. 28 of
the reference are

    Omega_unc     = max_t ||H_unc(t)||         = |delta| * w_unc
    Omega_avg     = ||<Htil>||                 = |delta| * w_avg
    Omega_avg^dev = max_t ||Htil(t) - <Htil>|| = |delta| * w_dev

in the induced 2-norm, all linear in ``|delta|`` because a scalar perturbation
enters ``H_unc`` linearly.  The per-unit rates ``w_*`` are therefore properties
of the controller and the structure, computed once by :func:`uncertainty_rates`.
The bound (Eqs. 29-30 of the reference) is

    T*Omega_bnd(delta) = sqrt( (T Omega_unc)(T Omega_avg^dev) + 4 T Omega_avg )
    F_lb               = max( 1 - (1/2) (exp((T Omega_bnd/2)^2) - 1)^2, 0 )

and is non-trivial only for ``T Omega_bnd <= 2 sqrt(ln(1+sqrt(2)))``
(:data:`T_OMEGA_MAX`, Eq. 32 of the reference); beyond that it is vacuous.

``F_avg^low`` (Eq. 24 of the reference) lower-bounds ``|Tr Utilde(T)|/d``, which
is the normalised gate fidelity of this package up to the global phase, so
``F_lb`` is directly comparable to ``F_mu``.  Theorem 1 assumes exact nominal
fidelity ``F_nom = 1``; pass ``nominal_error=eps_0`` to :func:`margin` or
:func:`threshold_time_bandwidth` to absorb the nominal deficit into the
threshold, which is conservative, or leave it at 0 to evaluate the bound as
stated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

Array = np.ndarray
HList = Sequence[Array]

#: Largest ``T*Omega_bnd`` for which ``F_lb > 0`` (their Eq. 32), in radians.
T_OMEGA_MAX = 2.0 * np.sqrt(np.log(1.0 + np.sqrt(2.0)))


@dataclass
class UncertaintyRates:
    """Per-unit-``delta`` uncertainty measures of their Eq. 28, plus ``T``.

    Attributes
    ----------
    w_unc, w_avg, w_dev :
        ``Omega_unc``, ``Omega_avg``, ``Omega_avg^dev`` at ``delta = 1``.
    T :
        Gate duration ``t_f``.
    """

    w_unc: float
    w_avg: float
    w_dev: float
    T: float
    # --- error control -------------------------------------------------
    # w_unc and w_avg are exact to roundoff (see uncertainty_rates).  w_dev
    # is a supremum recovered from samples, so it carries an uncertainty.
    w_dev_certified: float = float("nan")
    w_dev_refinement: float = float("nan")
    w_dev_bracket_lo: float = float("nan")
    w_dev_bracket_hi: float = float("nan")
    n_dev_used: int = 0
    dev_converged: bool = False
    #: Cycles the fastest Bohr frequency completes over one interval, i.e. the
    #: exactly-known bandwidth of ``Htil(k, .)``.  The seed grid is derived from
    #: this, so long intervals or wide spectra are resolved automatically.
    dev_cycles_max: float = float("nan")
    #: Samples per cycle actually achieved at the finest sweep.
    dev_samples_per_cycle: float = float("nan")
    #: Whether the requested samples-per-cycle was met on every interval
    #: (False means ``n_dev_max`` capped the grid before the bandwidth was
    #: resolved -- the supremum may then be under-estimated).
    dev_resolved: bool = False

    def time_bandwidth(self, delta: float) -> float:
        """``T*Omega_bnd`` of their Eq. 29 at perturbation ``delta``."""
        return time_bandwidth(self, delta)


def _spectral_norm(A: Array) -> float:
    return float(np.linalg.norm(A, 2))


def _time_average_htil(lam: Array, V: Array, dH: Array, dt: float) -> Array:
    """Exact ``int_0^dt exp(+i H s) dH exp(-i H s) ds`` for constant Hermitian ``H``.

    ``H`` is constant on the interval, so in its eigenbasis the integral is a
    divided difference: the ``(m, n)`` entry picks up
    ``int_0^dt exp(i s (lam_m - lam_n)) ds = dt * exp(i Y) * sin(Y)/Y`` with
    ``Y = dt (lam_m - lam_n)/2``.  As with ``core._dU_dmu_exact`` the exponent
    is purely imaginary, so this form has no cancellation and needs no
    magnitude threshold -- only the literal ``Y == 0`` entries are masked.
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
    """Uncertainty measures of their Eq. 28 for a piecewise-constant controller.

    Parameters
    ----------
    H_list :
        Nominal per-interval Hamiltonians ``H^(k)`` (as from
        ``perturbed_hamiltonians(..., delta=0.0)``).
    dH_list :
        Per-interval perturbation structures ``Hhat^(k)`` (as from
        ``dH_structure``), i.e. ``H_unc^(k) = delta * dH_list[k]``.
    dt :
        Interval length ``Delta``.
    n_quad :
        Accepted and unused.  ``<Htil>`` is evaluated in closed form, so
        there is no quadrature to tune.
    n_dev :
        Starting number of uniform grid points per interval (endpoints
        included) for the supremum defining ``Omega_avg^dev``.
    dev_tol :
        Relative tolerance for the adaptive refinement of that supremum.
    n_dev_max :
        Cap on the refined grid size per interval.
    adaptive_dev :
        If ``False``, sample once on the ``n_dev`` grid and report the
        certificate without refining.

    Notes
    -----
    Within interval ``k`` the nominal propagator is
    ``U_S(t_{k-1}+s) = exp(-i H^(k) s) P_{k-1}``, so
    ``Htil = P_{k-1}^dag exp(i H^(k) s) Hhat^(k) exp(-i H^(k) s) P_{k-1}``.
    Each ``H^(k)`` is diagonalised once and the conjugations are formed from
    its eigenbasis, avoiding repeated matrix exponentials.

    Error control
    -------------
    ``w_unc`` is exact: ``H_unc`` is piecewise constant, so its supremum over
    ``t`` is a maximum over intervals.

    ``w_avg`` is exact to roundoff: the time average is a closed-form divided
    difference (see ``_time_average_htil``), not a quadrature.

    ``w_dev`` is a supremum of a smooth function of ``s`` recovered from
    samples, and is the only approximated quantity here.  Sampling can only
    *under*-estimate a supremum, and a smaller ``w_dev`` yields a larger
    ``margin`` -- i.e. the error is biased in the optimistic direction.  Two
    certificates are therefore returned:

    ``w_dev_certified``
        Rigorous upper bound.  ``d(Htil)/ds = i[H^(k), Htil]`` gives
        ``||d(Htil)/ds|| <= 2 ||H^(k)|| ||Hhat^(k)||``, so on a grid of
        spacing ``h`` the sampled maximum falls short of the true supremum by
        at most ``L h / 2``.  The true ``Omega_avg^dev`` lies in
        ``[w_dev, w_dev_certified]``.
    ``w_dev_bracket_lo/hi``
        A second, independent rigorous bracket.  Unitary conjugation is
        isospectral, so ``||Htil(t)|| = ||Hhat^(k)||`` exactly, whence
        ``| ||Hhat^(k)|| - w_avg | <= ||Htil - <Htil>|| <= ||Hhat^(k)|| + w_avg``.
        Provides an inexpensive independent check on the sampling.
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
        """Interaction-picture uncertainty at time ``t_{k-1} + s``, per unit delta."""
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

    # Omega_avg^dev = sup_t ||Htil(t) - <Htil>||.  Locate candidate maxima on a
    # coarse grid, then polish each with Brent.  f is smooth in s, so the
    # polished value is accurate to ~eps rather than to the grid spacing.
    def f(k: int, s: float) -> float:
        return _spectral_norm(Htil(k, s) - Havg)

    # Htil(k, s) = P^dag exp(+i H s) Hhat exp(-i H s) P is a trigonometric
    # polynomial in s whose frequencies are exactly the Bohr frequencies
    # lam_m - lam_n of H^(k).  Its bandwidth is therefore known in closed form,
    # and the sampling density needed to resolve it can be *derived* rather than
    # assumed: over an interval of length dt the fastest component completes
    # cycles_k = ptp(lam_k) * dt / (2 pi) cycles.  Seeding each interval with
    # dev_samples_per_cycle samples per cycle keeps the supremum search honest
    # for controllers with long intervals or wide spectra, where a fixed grid
    # would under-resolve without indication.
    cycles = [float(np.ptp(lam)) * dt / (2.0 * np.pi) for lam, _ in eigs]
    n_seed = [
        min(max(int(n_dev), 3, int(np.ceil(dev_samples_per_cycle * ck)) + 1), int(n_dev_max))
        for ck in cycles
    ]
    achieved = [
        (n - 1) / ck if ck > 0 else float("inf") for n, ck in zip(n_seed, cycles)
    ]
    min_samples_per_cycle = float(min(achieved)) if achieved else float("inf")
    dev_resolved = bool(min_samples_per_cycle >= dev_samples_per_cycle)

    def sweep(scale: int, polish: bool) -> tuple[float, float]:
        """Return (best over grid+polish, rigorous Lipschitz shortfall of the grid).

        ``scale`` multiplies each interval's bandwidth-derived seed grid.
        """
        best = 0.0
        gap = 0.0
        for k in range(tau):
            n_grid = min(scale * (n_seed[k] - 1) + 1, int(n_dev_max))
            grid = np.linspace(0.0, dt, n_grid)
            vals = [f(k, s) for s in grid]
            local_best = max(vals)

            # Rigorous shortfall of a sampled maximum: d(Htil)/ds = i[H, Htil],
            # so ||d(Htil)/ds|| <= 2 ||H|| ||Hhat||, and f is Lipschitz in s
            # with that constant.  On spacing h a sampled max can fall short of
            # the true supremum by at most L h / 2.
            L_s = 2.0 * _spectral_norm(np.asarray(H_list[k])) * norms_dH[k]
            gap = max(gap, 0.5 * L_s * dt / (n_grid - 1))

            best_k = local_best
            if polish:
                # f is smooth in s, so polishing each candidate peak with Brent
                # gives a value accurate to ~eps rather than to the grid
                # spacing: near a maximum f(s*+d) = f(s*) - O(d^2).
                for i, v in enumerate(vals):
                    interior = 0 < i < n_grid - 1 and v >= vals[i - 1] and v >= vals[i + 1]
                    if not (interior or v >= local_best):
                        continue
                    a = grid[max(i - 1, 0)]
                    b = grid[min(i + 1, n_grid - 1)]
                    if b <= a:
                        continue
                    res = minimize_scalar(
                        lambda s: -f(k, s), bounds=(a, b), method="bounded",
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
        # Refine until two successive polished sweeps agree to dev_tol.  The
        # polish converges to the local maxima; doubling the seed grid guards
        # against a peak that the coarse grid missed entirely.
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

    # Independent rigorous bracket from isospectrality of unitary conjugation:
    # ||Htil(t)|| = ||Hhat^(k)|| exactly, so the deviation norm is bracketed.
    bracket_lo = max(0.0, max(n - w_avg for n in norms_dH))
    bracket_hi = w_unc + w_avg

    return UncertaintyRates(
        w_unc=float(w_unc),
        w_avg=float(w_avg),
        w_dev=float(w_dev),
        T=float(T),
        w_dev_certified=float(w_dev_sampled + lipschitz_gap),
        w_dev_refinement=float(refinement),
        w_dev_bracket_lo=float(bracket_lo),
        w_dev_bracket_hi=float(bracket_hi),
        n_dev_used=int(n_used),
        dev_converged=dev_converged,
        dev_cycles_max=float(max(cycles)) if cycles else 0.0,
        dev_samples_per_cycle=float(min_samples_per_cycle * scale),
        dev_resolved=dev_resolved,
    )


def time_bandwidth(rates: UncertaintyRates, delta: float) -> float:
    """``T*Omega_bnd`` of their Eq. 29 at perturbation ``delta``.

    Uses the linearity of their Eq. 28 in ``delta``:
    ``T*Omega_bnd(delta) = sqrt(a delta^2 + b |delta|)`` with
    ``a = T^2 w_unc w_dev`` and ``b = 4 T w_avg``.
    """
    d = abs(float(delta))
    a = rates.T**2 * rates.w_unc * rates.w_dev
    b = 4.0 * rates.T * rates.w_avg
    return float(np.sqrt(a * d * d + b * d))


def fidelity_bound(T_omega_bnd: float) -> float:
    """``F_lb`` of their Eq. 30 given ``T*Omega_bnd``."""
    y = float(T_omega_bnd)
    if y < 0:
        raise ValueError("T_omega_bnd must be non-negative")
    if y >= T_OMEGA_MAX:
        return 0.0
    return float(max(1.0 - 0.5 * (np.exp((y / 2.0) ** 2) - 1.0) ** 2, 0.0))


def fidelity_bound_at(rates: UncertaintyRates, delta: float) -> float:
    """``F_lb`` of their Eq. 30 at perturbation ``delta``."""
    return fidelity_bound(time_bandwidth(rates, delta))


def threshold_time_bandwidth(FT: float, nominal_error: float = 0.0) -> float:
    """``T*Omega_bnd`` at which their ``F_lb`` equals the threshold.

    Inverts their Eq. 30 in closed form: ``F_lb = F_eff`` gives
    ``T*Omega_bnd = 2 sqrt( ln(1 + sqrt(2 (1 - F_eff))) )`` where
    ``F_eff = FT + nominal_error`` absorbs the nominal fidelity deficit
    (their Theorem 1 assumes ``F_nom = 1``; see module caveat 1).

    Returns ``0.0`` when ``F_eff >= 1``, i.e. no perturbation is certifiable.
    """
    if not (0.0 < FT < 1.0):
        raise ValueError("FT must satisfy 0 < FT < 1")
    if nominal_error < 0.0:
        raise ValueError("nominal_error must be non-negative")
    eps = 1.0 - FT - nominal_error
    if eps <= 0.0:
        return 0.0
    return float(2.0 * np.sqrt(np.log(1.0 + np.sqrt(2.0 * eps))))


def margin(rates: UncertaintyRates, FT: float, nominal_error: float = 0.0) -> float:
    """Perturbation margin implied by their Theorem 1.

    The largest ``|delta|`` for which their ``F_lb >= FT + nominal_error``,
    i.e. the analogue of ``iterative_margin(...).M`` obtained from their bound.
    Since ``T*Omega_bnd`` is monotone in ``|delta|``, this inverts
    ``a delta^2 + b delta = y*^2`` in closed form with
    ``y* = threshold_time_bandwidth(FT, nominal_error)``.

    Returns ``0.0`` if no positive perturbation is certifiable, and ``inf`` if
    the perturbation does not enter the bound at all (``w_unc*w_dev = 0`` and
    ``w_avg = 0``, e.g. a structure that is annihilated in the interaction
    picture).
    """
    y = threshold_time_bandwidth(FT, nominal_error)
    if y <= 0.0:
        return 0.0
    a = rates.T**2 * rates.w_unc * rates.w_dev
    b = 4.0 * rates.T * rates.w_avg
    y2 = y * y
    if a <= 0.0 and b <= 0.0:
        return float("inf")
    if a <= 0.0:
        return float(y2 / b)
    return float((-b + np.sqrt(b * b + 4.0 * a * y2)) / (2.0 * a))

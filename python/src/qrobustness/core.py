# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Closed-system core: propagation, fidelity, structure constants and margins.

This module holds the piecewise-constant propagator, the trace-amplitude
gate fidelity F, the centred structure constant C_{\\hat H} and the
Lipschitz constant L = B_T C_{\\hat H}, the exact segment derivatives, the
one-dimensional margin algorithm :func:`iterative_margin`, and the loaders
for the problem and controller files. The MATLAB peer is matlab/+qrobustness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from scipy.io import loadmat
from scipy.linalg import expm
from scipy.optimize import brentq, toms748

Array = np.ndarray
HList = Sequence[Array]

MARGIN_METHODS = (
    "algorithm1",
    "lipschitz_brent",
    "lipschitz_toms748",
    "doubling",
    "newton_probe",
)
ROOT_SOLVERS = ("brent", "toms748", "bisection")
DU_METHODS = ("exact", "quadrature")

#: Phase-2 doublings and phase-3 bisection steps. The MATLAB peer uses the
#: same cap (``probe_cap`` in ``iterative_margin.m``); the paper quotes it.
PROBE_CAP = 200


def propagator(H_list: HList, dt: float) -> Array:
    """Propagator ``exp(-i dt H_K) ... exp(-i dt H_1)`` of a piecewise-constant Hamiltonian.

    Parameters
    ----------
    H_list :
        Interval Hamiltonians H^(k), each (N, N) Hermitian, in time order
        (the first element acts first).
    dt :
        Interval length Delta.

    Returns
    -------
    ndarray
        (N, N) unitary propagator U(t_f).
    """
    if len(H_list) == 0:
        raise ValueError("H_list must be non-empty")
    U = np.eye(H_list[0].shape[0], dtype=complex)
    for H in H_list:
        U = expm(-1j * dt * H) @ U
    return U


def gate_fidelity(U: Array, Uf: Array) -> float:
    """Trace-amplitude gate fidelity ``F = |Tr(Uf^dag U)| / N``.

    Parameters
    ----------
    U :
        (N, N) achieved unitary.
    Uf :
        (N, N) target unitary U_f.

    Returns
    -------
    float
        F in [0, 1], invariant under a global phase of either argument.
    """
    N = U.shape[0]
    return float(np.abs(np.trace(Uf.conj().T @ U)) / N)


def lipschitz_constant(FT: float, N: int, C_H: float) -> float:
    """Lipschitz constant ``L = B_T C_{\\hat H}`` with ``B_T = sqrt((1 - F_T^2) / N)``.

    Parameters
    ----------
    FT :
        Fidelity threshold F_T, 0 < F_T < 1.
    N :
        Hilbert-space dimension.
    C_H :
        Structure constant C_{\\hat H} (see :func:`structure_constant`).

    Returns
    -------
    float
        L, valid on the safe set {F > F_T}.
    """
    if not (0.0 < FT < 1.0):
        raise ValueError("FT must satisfy 0 < FT < 1")
    B_T = np.sqrt((1.0 - FT**2) / N)
    return float(B_T * C_H)


#: Tolerances for the Hermiticity check in :func:`traceless`, which every gauge uses.
HERMITIAN_RTOL = 1e-10
HERMITIAN_ATOL = 1e-12


def traceless(Hhat: Array) -> Array:
    """Traceless part ``Hbar = H - (Tr H / N) I`` of a Hermitian structure.

    The trace part only adds a global phase, so centring leaves F unchanged
    and can only tighten the constants.

    Parameters
    ----------
    Hhat :
        (N, N) Hermitian structure \\hat H.

    Returns
    -------
    ndarray
        (N, N) traceless Hbar.

    Raises
    ------
    ValueError
        If ``Hhat`` is not square or not Hermitian within
        ``HERMITIAN_RTOL`` / ``HERMITIAN_ATOL``.
    """
    H = np.asarray(Hhat)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("structure matrix must be square")
    if not np.allclose(H, H.conj().T, rtol=HERMITIAN_RTOL, atol=HERMITIAN_ATOL):
        raise ValueError("structure matrix must be Hermitian")
    N = H.shape[0]
    return H - (np.trace(H) / N) * np.eye(N, dtype=H.dtype)


def structure_constant(
    kind: str,
    Hhat: Array,
    dt: float,
    tau: int,
    controls: Array | None = None,
) -> float:
    """Structure constant C_{\\hat H} of a drift or control structure, centred first.

    Parameters
    ----------
    kind :
        ``'drift'`` (C = tau dt ||Hbar||_F) or ``'control'``
        (C = dt ||u||_1 ||Hbar||_F).
    Hhat :
        (N, N) Hermitian structure \\hat H.
    dt :
        Interval length Delta.
    tau :
        Number of intervals.
    controls :
        Control amplitudes of the perturbed control; required for
        ``kind='control'``.

    Returns
    -------
    float
        C_{\\hat H}.
    """
    nf = float(np.linalg.norm(traceless(Hhat), "fro"))
    kind = kind.lower()
    if kind == "drift":
        return float(tau * dt * nf)
    if kind == "control":
        if controls is None:
            raise ValueError("controls required for kind='control'")
        return float(dt * np.linalg.norm(np.asarray(controls).ravel(), 1) * nf)
    raise ValueError("kind must be 'drift' or 'control'")


def perturbed_hamiltonians(
    H0: Array,
    H1: Array,
    H2: Array,
    u1: Array,
    u2: Array,
    structure: str,
    delta: float,
) -> List[Array]:
    """Interval Hamiltonians with one term scaled by ``(1 + delta)``.

    Parameters
    ----------
    H0, H1, H2 :
        (N, N) drift and control Hamiltonians.
    u1, u2 :
        Control amplitudes, one per interval (equal length).
    structure :
        ``'H0'``, ``'H1'`` or ``'H2'``: the term to perturb.
    delta :
        Relative perturbation mu.

    Returns
    -------
    list of ndarray
        H^(k)(delta) for each interval.
    """
    u1 = np.asarray(u1).ravel()
    u2 = np.asarray(u2).ravel()
    if u1.size != u2.size:
        raise ValueError("u1 and u2 must have equal length")
    structure = structure.upper()
    out: List[Array] = []
    for k in range(u1.size):
        if structure == "H0":
            out.append(H0 * (1.0 + delta) + u1[k] * H1 + u2[k] * H2)
        elif structure == "H1":
            out.append(H0 + u1[k] * H1 * (1.0 + delta) + u2[k] * H2)
        elif structure == "H2":
            out.append(H0 + u1[k] * H1 + u2[k] * H2 * (1.0 + delta))
        else:
            raise ValueError("structure must be H0, H1, or H2")
    return out


def dH_structure(
    H0: Array,
    H1: Array,
    H2: Array,
    u1: Array,
    u2: Array,
    structure: str,
) -> List[Array]:
    """Per-interval structure ``dH^(k)/dmu`` of :func:`perturbed_hamiltonians`.

    Parameters
    ----------
    H0, H1, H2 :
        (N, N) drift and control Hamiltonians.
    u1, u2 :
        Control amplitudes, one per interval.
    structure :
        ``'H0'``, ``'H1'`` or ``'H2'``.

    Returns
    -------
    list of ndarray
        ``H0`` on every interval, or ``u_m[k] H_m`` for a control.
    """
    u1 = np.asarray(u1).ravel()
    u2 = np.asarray(u2).ravel()
    structure = structure.upper()
    if structure == "H0":
        return [H0.copy() for _ in range(u1.size)]
    if structure == "H1":
        return [u1[k] * H1 for k in range(u1.size)]
    if structure == "H2":
        return [u2[k] * H2 for k in range(u2.size)]
    raise ValueError("structure must be H0, H1, or H2")


def gauss_legendre_01(n: int) -> Tuple[Array, Array]:
    """``n``-point Gauss-Legendre nodes and weights on [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    nodes = 0.5 * (x + 1.0)
    weights = 0.5 * w
    return nodes, weights


def dU_dmu_integral(
    H: Array, dH: Array, dt: float, nodes: Array, weights: Array
) -> Array:
    """Segment derivative ``d/dmu exp(-i dt H)`` by quadrature.

    Evaluates ``-i dt int_0^1 e^{-i dt H (1-s)} dH e^{-i dt H s} ds``; the
    closed form is :func:`dU_dmu_exact`.

    Parameters
    ----------
    H, dH :
        (N, N) segment Hamiltonian and its structure.
    dt :
        Interval length Delta.
    nodes, weights :
        Quadrature rule on [0, 1] (see :func:`gauss_legendre_01`).

    Returns
    -------
    ndarray
        (N, N) derivative.
    """
    dU = np.zeros_like(H, dtype=complex)
    for s, w in zip(nodes, weights, strict=True):
        A = expm(-1j * dt * H * (1.0 - s))
        B = expm(-1j * dt * H * s)
        dU = dU + w * (A @ dH @ B)
    return -1j * dt * dU


def segment_eig(H: Array) -> Tuple[Array, Array]:
    """Hermitian eigendecomposition ``H = V diag(lam) V^dag`` of a segment Hamiltonian.

    ``H`` is symmetrised first so the eigenvector matrix is unitary.

    Returns
    -------
    lam : ndarray
        (N,) real eigenvalues.
    V : ndarray
        (N, N) unitary eigenvectors.
    """
    Hs = 0.5 * (np.asarray(H, dtype=complex) + np.asarray(H, dtype=complex).conj().T)
    lam, V = np.linalg.eigh(Hs)
    return lam, V


def segment_propagator(lam: Array, V: Array, dt: float) -> Array:
    """``exp(-i dt H)`` from the eigendecomposition ``(lam, V)`` of H; returns (N, N)."""
    return (V * np.exp(-1j * dt * lam)) @ V.conj().T


def dU_dmu_exact(lam: Array, V: Array, dH: Array, dt: float) -> Array:
    """Closed-form ``d/dmu exp(-i dt H)`` in the eigenbasis of a Hermitian H.

    Uses the divided difference ``exp(a/2) sin(X)/X`` with
    ``X = dt (lam_n - lam_m)/2``, which has no cancellation; Hermitian H only
    (Lindblad generators use :mod:`qrobustness.lindblad`).

    Parameters
    ----------
    lam, V :
        Eigendecomposition of H from :func:`segment_eig`.
    dH :
        (N, N) structure.
    dt :
        Interval length Delta.

    Returns
    -------
    ndarray
        (N, N) derivative.
    """
    X = 0.5 * dt * (lam[None, :] - lam[:, None])
    ph = np.exp(-0.5j * dt * lam)
    P = ph[:, None] * ph[None, :]
    zero = X == 0.0
    S = np.where(zero, 1.0, np.sin(X) / np.where(zero, 1.0, X))
    Phi = P * S
    return -1j * dt * (V @ ((V.conj().T @ dH @ V) * Phi) @ V.conj().T)


def differential_sensitivity(
    H_list: HList,
    dH_list: HList,
    dt: float,
    Uf: Array,
    n_quad: int = 32,
    *,
    method: str = "exact",
) -> float:
    """Differential sensitivity ``zeta = dF/dmu`` at the given point.

    Parameters
    ----------
    H_list :
        Interval Hamiltonians H^(k).
    dH_list :
        Interval structures dH^(k)/dmu.
    dt :
        Interval length Delta.
    Uf :
        Target unitary U_f.
    n_quad :
        Gauss-Legendre nodes for ``method='quadrature'``; ignored for ``'exact'``.
    method :
        ``'exact'`` (closed form, default) or ``'quadrature'``.

    Returns
    -------
    float
        zeta. Raises ValueError if F = 0 (the phase is undefined).
    """
    if method not in DU_METHODS:
        raise ValueError(f"Unknown method={method!r}; expected one of {DU_METHODS}")
    use_exact = method == "exact"

    tau = len(H_list)
    N = H_list[0].shape[0]
    if use_exact:
        eigs = [segment_eig(H) for H in H_list]
        Useg = [segment_propagator(lam, V, dt) for lam, V in eigs]
    else:
        eigs = []
        Useg = [expm(-1j * dt * H) for H in H_list]

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

    if not use_exact:
        nodes, weights = gauss_legendre_01(n_quad)

    zeta = 0.0
    for k in range(tau):
        if use_exact:
            lam, V = eigs[k]
            dUk = dU_dmu_exact(lam, V, dH_list[k], dt)
        else:
            dUk = dU_dmu_integral(H_list[k], dH_list[k], dt, nodes, weights)
        Dk = Suff[k + 1] @ dUk @ Pref[k]
        zeta += float(np.real(np.trace(Uf.conj().T @ Dk * e_minus_i_phi)))
    return zeta / N


@dataclass
class MarginResult:
    """Result of :func:`iterative_margin`.

    Attributes
    ----------
    M_minus, M_plus, M :
        Margins in each direction and ``M = min(M_minus, M_plus)``; M is a
        lower bound on the true margin.
    converged_minus, converged_plus :
        False only when the iteration limit was hit.
    mu_minus, mu_plus :
        Endpoints reached on each ray.
    method :
        Method used.
    n_evals, n_steps :
        Fidelity evaluations and steps (with ``return_diagnostics``).
    status_minus, status_plus :
        Stopping rule: ``'eta_band'``, ``'domain_truncated'`` (certifies only
        the distance to the domain edge, not a resolved margin),
        ``'iteration_limit'`` or ``'stalled'`` (no safe point beyond the last
        one; possible with ``eval_tol > 0``).
    safeguard_minus, safeguard_plus :
        True if an overshoot (F < F_T after a step) was bisected back.
    M_upper_minus, M_upper_plus, M_upper :
        Evaluated unsafe witnesses (``inf`` if none); filled with ``margin_tol``.
    margin_uncertainty :
        ``M_upper - M``.
    reason_minus, reason_plus :
        Bracket outcome with ``margin_tol``: ``'bracketed'`` (width within
        eps), ``'partial'`` (valid bracket wider than eps), ``'unresolved'``
        (stopped at a probe within eps_num of F_T), ``'boundary'`` (domain
        edge reached safe) or ``'exhausted'`` (no unsafe point found).
    certificate :
        ``'segment'`` (every point from mu0 to the endpoint is safe) or
        ``'endpoint'`` (only the endpoint; ``doubling``, ``newton_probe``).
    n_unresolved :
        Probes within eps_num of F_T, classified neither safe nor unsafe.
    n_evals_minus, n_evals_plus :
        Evaluations per ray (with ``return_diagnostics``); ``n_evals`` is
        their sum plus one at mu0.
    """

    M_minus: float
    M_plus: float
    M: float
    converged_minus: bool
    converged_plus: bool
    mu_minus: float
    mu_plus: float
    method: str = "algorithm1"
    n_evals: Optional[int] = None
    n_steps: Optional[int] = None
    status_minus: str = "unknown"
    status_plus: str = "unknown"
    safeguard_minus: bool = False
    safeguard_plus: bool = False
    M_upper_minus: float = float("inf")
    M_upper_plus: float = float("inf")
    M_upper: float = float("inf")
    margin_uncertainty: float = float("inf")
    reason_minus: str = "unknown"
    reason_plus: str = "unknown"
    certificate: str = "unknown"
    n_unresolved: int = 0
    n_evals_minus: Optional[int] = None
    n_evals_plus: Optional[int] = None


@dataclass
class _EvalCounter:
    """Evaluate a fidelity function once per point and count the evaluations."""

    fn: Callable[[float], float]
    n_evals: int = 0
    cache: dict = field(default_factory=dict)

    def __call__(self, mu: float) -> float:
        key = float(mu)
        if key not in self.cache:
            self.n_evals += 1
            self.cache[key] = float(self.fn(mu))
        return self.cache[key]


def _on_boundary(mu: float, mu_lo: float, mu_hi: float) -> bool:
    if np.isfinite(mu_lo) and abs(mu - mu_lo) <= max(
        1e-15, 10 * np.finfo(float).eps * abs(mu_lo)
    ):
        return True
    if np.isfinite(mu_hi) and abs(mu - mu_hi) <= max(
        1e-15, 10 * np.finfo(float).eps * abs(mu_hi)
    ):
        return True
    return False


def _clamp(mu: float, mu_lo: float, mu_hi: float) -> float:
    return min(max(mu, mu_lo), mu_hi)


def _is_safe(F: float, FT: float, strict: bool) -> bool:
    """Safe side of the threshold.

    With ``strict`` (a positive evaluation band) equality is unresolved, so
    safe means ``F > FT``. Otherwise ``F >= FT``, and ``F = FT`` stays safe.
    """
    return F > FT if strict else F >= FT


def _bisect_safe(
    fidelity_fn: Callable[[float], float],
    mu_safe0: float,
    mu_bad: float,
    FT: float,
    eta: float,
    strict: bool = False,
) -> Tuple[float, float]:
    """Bisect ``[mu_safe0, mu_bad]``; return the last safe point and its F."""
    a = mu_safe0
    b = mu_bad
    Fa = fidelity_fn(a)
    for _ in range(60):
        mid = 0.5 * (a + b)
        Fm = fidelity_fn(mid)
        if _is_safe(Fm, FT, strict):
            a = mid
            Fa = Fm
            if (Fa - FT) < eta:
                break
        else:
            b = mid
    return a, Fa


def _bracket_root_safe(
    fidelity_fn: Callable[[float], float],
    mu_safe: float,
    mu_bad: float,
    FT: float,
    eta: float,
    root_solver: str,
    strict: bool = False,
) -> Tuple[float, float]:
    """Locate a safe endpoint in [mu_safe, mu_bad] with 0 <= F - FT < eta when possible.

    Bracketed solvers find F(mu) = FT, then step back by xtol toward
    the safe side so the returned point satisfies F >= FT (certificate side).
    """
    if root_solver == "bisection":
        return _bisect_safe(fidelity_fn, mu_safe, mu_bad, FT, eta, strict)

    def g(mu: float) -> float:
        return fidelity_fn(mu) - FT

    if not _is_safe(fidelity_fn(mu_safe), FT, strict):
        raise ValueError("mu_safe must lie on the safe side of FT")
    if _is_safe(fidelity_fn(mu_bad), FT, strict):
        return mu_safe, fidelity_fn(mu_safe)
    # A root finder needs a strict sign change. Equality on the bad side
    # (the closed band) is bisected instead.
    if g(mu_safe) <= 0 or g(mu_bad) >= 0:
        return _bisect_safe(fidelity_fn, mu_safe, mu_bad, FT, eta, strict)

    xtol = max(eta / 10.0, 1e-14 * max(1.0, abs(mu_safe), abs(mu_bad)))
    a, b = (mu_safe, mu_bad) if mu_safe < mu_bad else (mu_bad, mu_safe)
    if root_solver == "brent":
        root = brentq(g, a, b, xtol=xtol, maxiter=100)
    elif root_solver == "toms748":
        root = toms748(g, a, b, xtol=xtol, maxiter=100)
    else:
        raise ValueError(
            f"Unknown root_solver={root_solver!r}; expected one of {ROOT_SOLVERS}"
        )

    # Prefer the safe side of the root so F >= FT.
    toward_safe = np.sign(mu_safe - root)
    if toward_safe == 0:
        toward_safe = np.sign(mu_safe - mu_bad) or 1.0
    mu_try = root + toward_safe * xtol
    mu_try = _clamp(
        mu_try,
        min(mu_safe, mu_bad),
        max(mu_safe, mu_bad),
    )
    F_try = fidelity_fn(mu_try)
    if _is_safe(F_try, FT, strict):
        return mu_try, F_try
    # Fallback: classical bisection starting from the known safe endpoint.
    return _bisect_safe(fidelity_fn, mu_safe, mu_bad, FT, eta, strict)


#: Stopping rules of :func:`iterative_margin`, in the order they are tested.
MARGIN_STATUS = ("eta_band", "domain_truncated", "iteration_limit", "stalled")


class _DirOutcome(NamedTuple):
    """One direction result: margin, flags, endpoint, and how it stopped."""

    M: float
    converged: bool
    mu_end: float
    n_steps: int
    status: str = "unknown"
    safeguard: bool = False


def _stop_one_direction(
    mu0: float,
    mu_next: float,
    F_next: float,
    FT: float,
    eta: float,
    mu_lo: float,
    mu_hi: float,
    k: int,
    k_max: int,
    strict: bool = False,
) -> Tuple[bool, bool, float, str]:
    """Return (done, converged, M, status)."""
    surplus = F_next - FT
    if _on_boundary(mu_next, mu_lo, mu_hi) and surplus >= eta:
        return True, True, abs(mu0 - mu_next), "domain_truncated"
    in_band = (0 < surplus if strict else 0 <= surplus) and surplus < eta
    if in_band:
        return True, True, abs(mu0 - mu_next), "eta_band"
    if k >= k_max:
        return True, False, abs(mu0 - mu_next), "iteration_limit"
    return False, True, abs(mu0 - mu_next), "running"


def _one_direction_lipschitz(
    fidelity_fn: Callable[[float], float],
    L: float,
    FT: float,
    mu0: float,
    eta: float,
    omega: Tuple[float, float],
    k_max: int,
    ell: int,
    root_solver: str,
    safe_radius_fn: Callable[[float], float],
    strict: bool = False,
) -> _DirOutcome:
    """Certified safe-radius advance; polish any overshoot with root_solver."""
    sign_step = (-1) ** ell
    mu_lo, mu_hi = omega
    k = 1  # tallies evaluated trial points, so k_max of them are permitted
    n_steps = 0
    mu = mu0
    Fmu = fidelity_fn(mu)
    mu_next = mu
    F_next = Fmu
    converged = True
    safeguard = False

    while True:
        mu_next = _clamp(mu + sign_step * safe_radius_fn(Fmu), mu_lo, mu_hi)
        F_next = fidelity_fn(mu_next)
        if not _is_safe(F_next, FT, strict):
            safeguard = True
            mu_next, F_next = _bracket_root_safe(
                fidelity_fn, mu, mu_next, FT, eta, root_solver, strict
            )
        done, converged, M, status = _stop_one_direction(
            mu0, mu_next, F_next, FT, eta, mu_lo, mu_hi, k, k_max, strict
        )
        if done:
            return _DirOutcome(M, converged, mu_next, n_steps, status, safeguard)
        if sign_step * (mu_next - mu) <= 4 * np.finfo(float).eps * max(1.0, abs(mu)):
            # No progress beyond floating-point resolution (no safe point
            # beyond mu): further steps would repeat it.
            return _DirOutcome(abs(mu0 - mu), False, mu, n_steps, "stalled", safeguard)
        k += 1
        n_steps += 1
        mu = mu_next
        Fmu = F_next


def _one_direction_doubling(
    fidelity_fn: Callable[[float], float],
    L: float,
    FT: float,
    mu0: float,
    eta: float,
    omega: Tuple[float, float],
    k_max: int,
    ell: int,
    root_solver: str,
    safe_radius_fn: Callable[[float], float],
    strict: bool = False,
) -> _DirOutcome:
    """Geometric probes past the safe radius, then bracket; endpoint-certified only."""
    sign_step = (-1) ** ell
    mu_lo, mu_hi = omega
    n_steps = 0
    mu_safe = mu0
    F_safe = fidelity_fn(mu_safe)
    # Initial probe: at least the safe radius; eta / L only keeps it nonzero.
    step = max(safe_radius_fn(F_safe), eta / max(L, 1e-30))
    mu_probe = _clamp(mu_safe + sign_step * step, mu_lo, mu_hi)
    F_probe = fidelity_fn(mu_probe)
    k = 1  # tallies evaluated trial points, so k_max of them are permitted

    while _is_safe(F_probe, FT, strict):
        if _on_boundary(mu_probe, mu_lo, mu_hi):
            done, converged, M, status = _stop_one_direction(
                mu0, mu_probe, F_probe, FT, eta, mu_lo, mu_hi, k, k_max, strict
            )
            return _DirOutcome(M, converged, mu_probe, n_steps, status)
        surplus = F_probe - FT
        in_band = (0 < surplus if strict else 0 <= surplus) and surplus < eta
        if in_band:
            return _DirOutcome(abs(mu0 - mu_probe), True, mu_probe, n_steps, "eta_band")
        if k >= k_max:
            return _DirOutcome(
                abs(mu0 - mu_probe), False, mu_probe, n_steps, "iteration_limit"
            )
        mu_safe = mu_probe
        F_safe = F_probe
        step *= 2.0
        mu_probe = _clamp(mu_safe + sign_step * step, mu_lo, mu_hi)
        if abs(mu_probe - mu_safe) <= 0.0:
            # The doubled probe cannot leave mu_safe: the domain edge (or
            # the fp64 floor) is reached while still safe.
            return _DirOutcome(
                abs(mu0 - mu_safe), True, mu_safe, n_steps, "domain_truncated"
            )
        F_probe = fidelity_fn(mu_probe)
        k += 1
        n_steps += 1

    mu_end, F_end = _bracket_root_safe(
        fidelity_fn, mu_safe, mu_probe, FT, eta, root_solver, strict
    )
    return _DirOutcome(abs(mu0 - mu_end), True, mu_end, n_steps, "eta_band", True)


def _one_direction_newton_probe(
    fidelity_fn: Callable[[float], float],
    L: float,
    FT: float,
    mu0: float,
    eta: float,
    omega: Tuple[float, float],
    k_max: int,
    ell: int,
    root_solver: str,
    safe_radius_fn: Callable[[float], float],
    zeta_fn: Callable[[float], float],
    strict: bool = False,
) -> _DirOutcome:
    """Newton-sized probes, never shorter than the safe radius; endpoint-certified only."""
    sign_step = (-1) ** ell
    mu_lo, mu_hi = omega
    k = 1  # tallies evaluated trial points, so k_max of them are permitted
    n_steps = 0
    mu = mu0
    Fmu = fidelity_fn(mu)
    mu_next = mu
    F_next = Fmu
    safeguard = False

    while True:
        cert_step = safe_radius_fn(Fmu)
        zeta = float(zeta_fn(mu))
        if abs(zeta) > 1e-14:
            newt_step = abs((Fmu - FT) / zeta)
        else:
            newt_step = cert_step
        step = max(cert_step, newt_step)
        mu_next = _clamp(mu + sign_step * step, mu_lo, mu_hi)
        F_next = fidelity_fn(mu_next)
        if not _is_safe(F_next, FT, strict):
            mu_next, F_next = _bracket_root_safe(
                fidelity_fn, mu, mu_next, FT, eta, root_solver, strict
            )
            return _DirOutcome(
                abs(mu0 - mu_next), True, mu_next, n_steps, "eta_band", True
            )
        done, converged, M, status = _stop_one_direction(
            mu0, mu_next, F_next, FT, eta, mu_lo, mu_hi, k, k_max, strict
        )
        if done:
            return _DirOutcome(M, converged, mu_next, n_steps, status, safeguard)
        # If still far above FT after a large probe, double as in ``doubling``.
        if step > cert_step * (1.0 + 1e-12) and (F_next - FT) >= eta:
            step2 = 2.0 * step
            mu_probe = _clamp(mu_next + sign_step * step2, mu_lo, mu_hi)
            F_probe = fidelity_fn(mu_probe)
            n_steps += 1
            if not _is_safe(F_probe, FT, strict):
                mu_next, F_next = _bracket_root_safe(
                    fidelity_fn, mu_next, mu_probe, FT, eta, root_solver, strict
                )
                return _DirOutcome(
                    abs(mu0 - mu_next), True, mu_next, n_steps, "eta_band", True
                )
            mu = mu_next
            Fmu = F_next
            mu_next = mu_probe
            F_next = F_probe
        k += 1
        n_steps += 1
        mu = mu_next
        Fmu = F_next


def _certify_direction(
    fidelity_fn: Callable[[float], float],
    mu0: float,
    mu_end: float,
    FT: float,
    ell: int,
    omega: Tuple[float, float],
    margin_tol: float,
    L: float = 0.0,
    safe_radius_fn: Optional[Callable[[float], float]] = None,
    eval_tol: float = 0.0,
) -> Tuple[float, float, str, int]:
    """Bracket the first boundary of the nominal safe component on one ray.

    Probes outward from the continuation endpoint ``mu_end`` for an unsafe
    witness, then bisects. A safe sample becomes the certified end only if
    its own safe radius covers the gap to the current certified end (or
    continuation bridges it), so a safe island beyond the first boundary
    never enlarges M. Probes with ``|F - FT| <= eval_tol`` are unresolved:
    neither promoted nor taken as witnesses, and counted.

    Returns
    -------
    tuple
        ``(M, M_upper, reason, n_unresolved)``; reason is ``'bracketed'``,
        ``'partial'``, ``'unresolved'``, ``'boundary'`` or ``'exhausted'``
        (the last two with ``M_upper = inf``).
    """
    sign_step = (-1) ** ell
    mu_lo, mu_hi = omega
    scale = max(abs(mu_end - mu0), 1e-12)
    if safe_radius_fn is None and L > 0.0:
        safe_radius_fn = lambda F: (F - FT) / L  # noqa: E731

    n_unresolved = 0

    # Each point is evaluated once; the classifiers below share the value.
    evaluated: dict = {}
    raw_fidelity = fidelity_fn

    def fidelity_fn(mu: float) -> float:
        if mu not in evaluated:
            evaluated[mu] = raw_fidelity(mu)
        return evaluated[mu]

    def is_unsafe(mu: float) -> bool:
        """True if F < FT - eval_tol."""
        return fidelity_fn(mu) < FT - eval_tol

    def is_unresolved(mu: float) -> bool:
        return abs(fidelity_fn(mu) - FT) <= eval_tol

    def can_promote(cand: float, cert: float) -> bool:
        F = fidelity_fn(cand)
        # eval_tol == 0 keeps F == FT safe. A positive band is closed, so
        # F == FT + eval_tol stays unresolved and is not promoted.
        if eval_tol > 0.0:
            if F <= FT + eval_tol:
                return False
        elif F < FT:
            return False
        if safe_radius_fn is None:
            # Without a radius rule, never promote across a gap.
            return cand == cert
        return abs(cand - cert) <= safe_radius_fn(F)

    def continue_toward(cert: float, target_pt: float, max_steps: int = 64) -> float:
        """Continue from ``cert`` toward ``target_pt``; return the farthest safe point."""
        if safe_radius_fn is None:
            return cert
        for _ in range(max_steps):
            F = fidelity_fn(cert)
            if (eval_tol > 0.0 and F <= FT + eval_tol) or (eval_tol == 0.0 and F < FT):
                return cert
            step_r = safe_radius_fn(F)
            if step_r <= abs(target_pt - cert) * 1e-15 + 1e-300:
                return cert  # stalled at the band
            nxt = cert + sign_step * min(step_r, abs(target_pt - cert))
            if sign_step * (nxt - target_pt) >= 0:
                target_F = fidelity_fn(target_pt)
                safe = target_F > FT + eval_tol if eval_tol > 0.0 else target_F >= FT
                return target_pt if safe else cert
            cert = nxt
        return cert

    # Geometric probe outward for an unsafe witness.
    mu_cert = mu_end
    frontier = mu_end
    mu_unsafe = None
    step = max(margin_tol * scale, 1e-15)
    for _ in range(PROBE_CAP):
        cand = _clamp(frontier + sign_step * step, mu_lo, mu_hi)
        if cand == frontier:
            return abs(mu0 - mu_cert), float("inf"), "boundary", n_unresolved
        if is_unsafe(cand):
            mu_unsafe = cand
            break
        if eval_tol > 0.0 and is_unresolved(cand):
            n_unresolved += 1
            frontier = cand
            step *= 2.0
            continue
        if can_promote(cand, mu_cert):
            mu_cert = cand
        else:
            mu_cert = continue_toward(mu_cert, cand)
            if mu_cert != cand:
                pass  # stalled: the boundary lies before cand; keep probing
        frontier = cand
        step *= 2.0
    if mu_unsafe is None:
        return abs(mu0 - mu_cert), float("inf"), "exhausted", n_unresolved

    # Refinement: safe midpoints are promoted only as in the probe phase.
    # The second term is the binary64 resolution at the coordinate mu_cert
    # (bisection cannot resolve below it, wherever mu0 is); on a ray from
    # mu0 = 0 it is relative to the margin itself.
    target = max(
        margin_tol * max(abs(mu_cert - mu0), 1e-300), 1e-16 * max(1.0, abs(mu_cert))
    )
    for _ in range(PROBE_CAP):
        if abs(mu_unsafe - mu_cert) <= target:
            break
        mid = 0.5 * (mu_cert + mu_unsafe)
        if mid == mu_cert or mid == mu_unsafe:
            break  # fp64 floor
        if is_unsafe(mid):
            mu_unsafe = mid
            continue
        if eval_tol > 0.0 and is_unresolved(mid):
            # An unresolved midpoint moves neither end: stop with a wider bracket.
            n_unresolved += 1
            return (
                abs(mu0 - mu_cert),
                abs(mu0 - mu_unsafe),
                "unresolved",
                n_unresolved,
            )
        if can_promote(mid, mu_cert):
            mu_cert = mid
        else:
            reached = continue_toward(mu_cert, mid)
            if reached == mu_cert:
                # Continuation stalled: valid bracket wider than margin_tol.
                return (
                    abs(mu0 - mu_cert),
                    abs(mu0 - mu_unsafe),
                    "partial",
                    n_unresolved,
                )
            mu_cert = reached
    reason = (
        "bracketed"
        if abs(mu_unsafe - mu_cert) <= max(target, 2e-16 * max(1.0, abs(mu_cert)))
        else "partial"
    )
    return abs(mu0 - mu_cert), abs(mu0 - mu_unsafe), reason, n_unresolved


def _dispatch_one_direction(
    fidelity_fn: Callable[[float], float],
    L: float,
    FT: float,
    mu0: float,
    eta: float,
    omega: Tuple[float, float],
    k_max: int,
    ell: int,
    method: str,
    root_solver: str,
    zeta_fn: Optional[Callable[[float], float]],
    safe_radius_fn: Callable[[float], float],
    strict: bool = False,
) -> _DirOutcome:
    if method == "algorithm1":
        return _one_direction_lipschitz(
            fidelity_fn,
            L,
            FT,
            mu0,
            eta,
            omega,
            k_max,
            ell,
            "bisection",
            safe_radius_fn,
            strict,
        )
    if method == "lipschitz_brent":
        return _one_direction_lipschitz(
            fidelity_fn,
            L,
            FT,
            mu0,
            eta,
            omega,
            k_max,
            ell,
            "brent",
            safe_radius_fn,
            strict,
        )
    if method == "lipschitz_toms748":
        return _one_direction_lipschitz(
            fidelity_fn,
            L,
            FT,
            mu0,
            eta,
            omega,
            k_max,
            ell,
            "toms748",
            safe_radius_fn,
            strict,
        )
    if method == "doubling":
        rs = root_solver if root_solver != "bisection" else "toms748"
        return _one_direction_doubling(
            fidelity_fn,
            L,
            FT,
            mu0,
            eta,
            omega,
            k_max,
            ell,
            rs,
            safe_radius_fn,
            strict,
        )
    if method == "newton_probe":
        if zeta_fn is None:
            raise ValueError("method='newton_probe' requires zeta_fn")
        rs = root_solver if root_solver != "bisection" else "toms748"
        return _one_direction_newton_probe(
            fidelity_fn,
            L,
            FT,
            mu0,
            eta,
            omega,
            k_max,
            ell,
            rs,
            safe_radius_fn,
            zeta_fn,
            strict,
        )
    raise ValueError(f"Unknown method={method!r}; expected one of {MARGIN_METHODS}")


def iterative_margin(
    fidelity_fn: Callable[[float], float],
    L: float,
    FT: float,
    mu0: float = 0.0,
    eta: float = 1e-6,
    omega: Tuple[float, float] = (-np.inf, np.inf),
    k_max: int = 10000,
    method: str = "algorithm1",
    root_solver: str = "toms748",
    zeta_fn: Optional[Callable[[float], float]] = None,
    return_diagnostics: bool = False,
    margin_tol: Optional[float] = None,
    safe_radius_fn: Optional[Callable[[float], float]] = None,
    eval_tol: float = 0.0,
) -> MarginResult:
    """One-dimensional robustness margin. Continuation starts at mu0 and steps by the safe radius.

    The walk steps outward in both directions by the safe radius until F is
    within eta of F_T. It can then bracket the boundary to the relative
    tolerance eps (``margin_tol``). M is a lower bound on the true margin.
    M_upper is an evaluated unsafe witness. It is not a certified bound.

    Parameters
    ----------
    fidelity_fn :
        Scalar fidelity F(mu).
    L :
        Lipschitz constant B_T C_{\\hat H} (> 0).
    FT :
        Fidelity threshold F_T; requires F(mu0) > F_T.
    mu0 :
        Nominal parameter.
    eta :
        Continuation band: stop once 0 <= F - F_T < eta.
    omega :
        Admissible interval (mu_lo, mu_hi).
    k_max :
        Maximum fidelity evaluations per direction.
    method :
        ``'algorithm1'`` (default; safe-radius steps, bisection on
        overshoot), ``'lipschitz_brent'`` / ``'lipschitz_toms748'`` (same
        steps, Brent or TOMS748 on overshoot); these certify the whole
        segment. ``'doubling'`` and ``'newton_probe'`` probe past the safe
        radius and certify only the endpoint.
    root_solver :
        ``'toms748'`` (default), ``'brent'`` or ``'bisection'``; used by
        ``doubling`` and ``newton_probe`` (``'bisection'`` falls back to
        ``'toms748'`` there), ignored otherwise.
    zeta_fn :
        Differential sensitivity zeta(mu); required for ``'newton_probe'``.
    return_diagnostics :
        If True, fill ``n_evals``, ``n_evals_minus``, ``n_evals_plus`` and
        ``n_steps``.
    margin_tol :
        Relative bracket tolerance eps; if given, fills ``M_upper*``,
        ``reason_*``, ``margin_uncertainty`` and ``n_unresolved``.
    safe_radius_fn :
        Safe radius r(F) at an evaluated safe point, in parameter units;
        must be non-negative, non-decreasing in F and vanish at F_T.
        Default ``(F - F_T)/L``; ``multiparam.directional_margin`` supplies
        the angular radius ``(arccos F_T - arccos F)/C^stat_FS(d)``.
    eval_tol :
        Evaluation band eps_num of ``fidelity_fn``. Continuation treats a
        point as safe only above F_T + eps_num and takes every safe radius
        at F - eps_num; probes within eps_num of F_T are unresolved and
        counted in ``n_unresolved``. ``mu0`` must be safe, F(mu0) > F_T +
        eps_num.

    Returns
    -------
    MarginResult
        Margins, endpoints, stopping status, certificate kind and, with
        ``margin_tol``, the bracket. ``status_* = 'domain_truncated'``
        certifies only the distance to the edge of ``omega``.
    """
    if L <= 0:
        raise ValueError("Lipschitz constant L must be positive")
    if method not in MARGIN_METHODS:
        raise ValueError(f"Unknown method={method!r}; expected one of {MARGIN_METHODS}")
    if root_solver not in ROOT_SOLVERS:
        raise ValueError(
            f"Unknown root_solver={root_solver!r}; expected one of {ROOT_SOLVERS}"
        )

    counter = _EvalCounter(fidelity_fn)
    F0 = counter(mu0)
    if not (FT + eval_tol < F0):
        raise ValueError(
            f"Require FT + eval_tol < F(mu0); got FT={FT}, eval_tol={eval_tol}, F={F0}"
        )
    if safe_radius_fn is None:
        safe_radius_fn = lambda F: (F - FT) / L  # noqa: E731
    if eval_tol > 0.0:
        # With evaluation error up to eval_tol the true fidelity may be as low
        # as F - eval_tol, so every radius is taken there.
        raw_radius = safe_radius_fn

        def safe_radius_fn(F: float) -> float:
            return raw_radius(F - eval_tol) if F - eval_tol > FT else 0.0

    # Continuation accepts a point as safe only above the evaluation band.
    FT_safe = FT + eval_tol
    n0 = counter.n_evals
    out_m = _dispatch_one_direction(
        counter,
        L,
        FT_safe,
        mu0,
        eta,
        omega,
        k_max,
        1,
        method,
        root_solver,
        zeta_fn,
        safe_radius_fn,
        eval_tol > 0.0,
    )
    n_m = counter.n_evals - n0
    n0 = counter.n_evals
    out_p = _dispatch_one_direction(
        counter,
        L,
        FT_safe,
        mu0,
        eta,
        omega,
        k_max,
        2,
        method,
        root_solver,
        zeta_fn,
        safe_radius_fn,
        eval_tol > 0.0,
    )
    n_p = counter.n_evals - n0
    steps_m, steps_p = out_m.n_steps, out_p.n_steps
    result = MarginResult(
        M_minus=out_m.M,
        M_plus=out_p.M,
        M=min(out_m.M, out_p.M),
        converged_minus=out_m.converged,
        converged_plus=out_p.converged,
        mu_minus=out_m.mu_end,
        mu_plus=out_p.mu_end,
        status_minus=out_m.status,
        status_plus=out_p.status,
        safeguard_minus=out_m.safeguard,
        safeguard_plus=out_p.safeguard,
        method=method,
        certificate="segment"
        if method in ("algorithm1", "lipschitz_brent", "lipschitz_toms748")
        else "endpoint",
    )
    if margin_tol is not None:
        if not (margin_tol > 0):
            raise ValueError("margin_tol must be positive")
        n0 = counter.n_evals
        lo_m, up_m, why_m, unres_m = _certify_direction(
            counter,
            mu0,
            out_m.mu_end,
            FT,
            1,
            omega,
            margin_tol,
            L,
            safe_radius_fn,
            eval_tol,
        )
        n_m += counter.n_evals - n0
        n0 = counter.n_evals
        lo_p, up_p, why_p, unres_p = _certify_direction(
            counter,
            mu0,
            out_p.mu_end,
            FT,
            2,
            omega,
            margin_tol,
            L,
            safe_radius_fn,
            eval_tol,
        )
        n_p += counter.n_evals - n0
        # The refined safe ends are stricter lower bounds than the eta-based ones.
        result.M_minus = max(result.M_minus, lo_m)
        result.M_plus = max(result.M_plus, lo_p)
        result.M = min(result.M_minus, result.M_plus)
        result.M_upper_minus = up_m
        result.M_upper_plus = up_p
        result.reason_minus = why_m
        result.reason_plus = why_p
        result.M_upper = min(up_m, up_p)
        result.margin_uncertainty = result.M_upper - result.M
        result.n_unresolved = unres_m + unres_p
    if return_diagnostics:
        result.n_evals = counter.n_evals
        result.n_evals_minus = n_m
        result.n_evals_plus = n_p
        result.n_steps = steps_m + steps_p
    return result


def fidelity_vs_delta(
    fidelity_fn: Callable[[float], float],
    delta_grid: Iterable[float],
) -> Tuple[Array, Array]:
    """Sample ``fidelity_fn`` on ``delta_grid``.

    Returns
    -------
    x, F : ndarray
        The grid and the fidelities on it.
    """
    x = np.asarray(list(delta_grid), dtype=float)
    F = np.array([fidelity_fn(float(d)) for d in x], dtype=float)
    return x, F


def make_fidelity_fn(
    H0: Array,
    H1: Array,
    H2: Array,
    u1: Array,
    u2: Array,
    Uf: Array,
    dt: float,
    structure: str,
) -> Callable[[float], float]:
    """Fidelity of a controller as a function of the relative perturbation ``delta``.

    Parameters
    ----------
    H0, H1, H2 :
        (N, N) drift and control Hamiltonians.
    u1, u2 :
        Control amplitudes per interval.
    Uf :
        Target unitary U_f.
    dt :
        Interval length Delta.
    structure :
        ``'H0'``, ``'H1'`` or ``'H2'`` (see :func:`perturbed_hamiltonians`).

    Returns
    -------
    callable
        ``delta -> F``.
    """

    def fn(delta: float) -> float:
        H_list = perturbed_hamiltonians(H0, H1, H2, u1, u2, structure, delta)
        U = propagator(H_list, dt)
        return gate_fidelity(U, Uf)

    return fn


def load_problem(mat_path: Union[str, Path]) -> dict:
    """Load a problem definition (variable ``problem``) from a MATLAB ``.mat`` file.

    Parameters
    ----------
    mat_path :
        Path to the ``.mat`` file.

    Returns
    -------
    dict
        ``H0``, ``H1``, ``H2`` (Hamiltonians), ``Uf`` (target), ``n_qubits``
        and ``dim = 2**n_qubits``. The stored field ``N`` is the qubit
        count, not the dimension.
    """
    S = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "problem" not in S:
        raise KeyError("Expected variable 'problem'")
    p = S["problem"]
    H = p.H
    H0, H1, H2 = H[0], H[1], H[2]
    n_qubits = int(np.asarray(p.N).reshape(-1)[0])
    return {
        "H0": np.asarray(H0, dtype=complex),
        "H1": np.asarray(H1, dtype=complex),
        "H2": np.asarray(H2, dtype=complex),
        "Uf": np.asarray(p.UT, dtype=complex),
        "n_qubits": n_qubits,
        "dim": 2**n_qubits,
    }


def load_controllers(csv_path: Union[str, Path], max_error: float = 1e-4) -> List[dict]:
    """Load a controller ensemble from CSV and sort it by nominal error.

    Parameters
    ----------
    csv_path :
        CSV; columns 2-4 are ``tf``, ``tau``, ``error``, then the controls
        interleaved ``u1_1, u2_1, u1_2, ...``.
    max_error :
        Keep controllers with ``error <= max_error``; the
        default matches ``scripts/_drivers.DEFAULT_MAX_ERROR`` and MATLAB.

    Returns
    -------
    list of dict
        Per controller: ``tf``, ``tau``, ``error``, ``fid = 1 - error``,
        ``u1``, ``u2`` (length ``tau``).
    """
    data = np.loadtxt(csv_path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    data = data[np.argsort(data[:, 4])]
    data = data[data[:, 4] <= max_error]
    controllers: List[dict] = []
    for row in data:
        tf = float(row[2])
        tau = int(row[3])
        err = float(row[4])
        # MATLAB reshape(data, 2, tau) is column-major: u1,u2 samples interleaved.
        u = row[5:].reshape((2, tau), order="F")
        controllers.append(
            {
                "tf": tf,
                "tau": tau,
                "error": err,
                "fid": 1.0 - err,
                "u1": u[0].astype(float),
                "u2": u[1].astype(float),
            }
        )
    return controllers

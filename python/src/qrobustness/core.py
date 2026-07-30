# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Core numerical routines mirroring matlab/+qrobustness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

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


def propagator(H_list: HList, dt: float) -> Array:
    if len(H_list) == 0:
        raise ValueError("H_list must be non-empty")
    U = np.eye(H_list[0].shape[0], dtype=complex)
    for H in H_list:
        U = expm(-1j * dt * H) @ U
    return U


def gate_fidelity(U: Array, Uf: Array) -> float:
    N = U.shape[0]
    return float(np.abs(np.trace(Uf.conj().T @ U)) / N)


def lipschitz_constant(FT: float, N: int, C_H: float) -> float:
    if not (0.0 < FT < 1.0):
        raise ValueError("FT must satisfy 0 < FT < 1")
    B_T = np.sqrt((1.0 - FT**2) / N)
    return float(B_T * C_H)


def structure_constant(
    kind: str,
    Hhat: Array,
    dt: float,
    tau: int,
    controls: Array | None = None,
) -> float:
    nf = float(np.linalg.norm(Hhat, "fro"))
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


def _gauss_legendre_01(n: int) -> Tuple[Array, Array]:
    x, w = np.polynomial.legendre.leggauss(n)
    nodes = 0.5 * (x + 1.0)
    weights = 0.5 * w
    return nodes, weights


def _dU_dmu_integral(H: Array, dH: Array, dt: float, nodes: Array, weights: Array) -> Array:
    dU = np.zeros_like(H, dtype=complex)
    for s, w in zip(nodes, weights):
        A = expm(-1j * dt * H * (1.0 - s))
        B = expm(-1j * dt * H * s)
        dU = dU + w * (A @ dH @ B)
    return -1j * dt * dU


def _segment_eig(H: Array) -> Tuple[Array, Array]:
    """Hermitian eigendecomposition of a segment Hamiltonian.

    Symmetrises first so the Hermitian LAPACK path is taken unconditionally,
    which is what guarantees a unitary eigenvector matrix.
    """
    Hs = 0.5 * (np.asarray(H, dtype=complex) + np.asarray(H, dtype=complex).conj().T)
    lam, V = np.linalg.eigh(Hs)
    return lam, V


def _segment_propagator(lam: Array, V: Array, dt: float) -> Array:
    """exp(-1j*dt*H) from the eigendecomposition of H."""
    return (V * np.exp(-1j * dt * lam)) @ V.conj().T


def _dU_dmu_exact(lam: Array, V: Array, dH: Array, dt: float) -> Array:
    """Exact d/dmu exp(-1j*dt*H) for constant H, given its eigendecomposition.

    For piecewise-constant controls the Frechet derivative

        dU/dmu = -1j*dt * int_0^1 exp(-1j*dt*H*(1-s)) dH exp(-1j*dt*H*s) ds

    is a divided difference in the eigenbasis.  Writing a = -1j*dt*(lam_n-lam_m),

        (exp(a) - 1) / a = exp(a/2) * sin(X)/X,   X = 0.5*dt*(lam_n - lam_m),

    which is exact because a is purely imaginary, so there is no cancellation
    and no magnitude threshold to tune -- only X == 0 needs masking.
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
    """Gate-fidelity sensitivity zeta at the given point.

    method='exact' (default) evaluates the segment derivative in closed form in
    the eigenbasis of each H^(k), which is exact for the piecewise-constant
    controls assumed throughout; 'quadrature' uses Gauss-Legendre with n_quad
    nodes.  n_quad is inert under 'exact'.
    """
    if method not in DU_METHODS:
        raise ValueError(f"Unknown method={method!r}; expected one of {DU_METHODS}")
    use_exact = method == "exact"

    tau = len(H_list)
    N = H_list[0].shape[0]
    if use_exact:
        eigs = [_segment_eig(H) for H in H_list]
        Useg = [_segment_propagator(lam, V, dt) for lam, V in eigs]
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
        nodes, weights = _gauss_legendre_01(n_quad)

    zeta = 0.0
    for k in range(tau):
        if use_exact:
            lam, V = eigs[k]
            dUk = _dU_dmu_exact(lam, V, dH_list[k], dt)
        else:
            dUk = _dU_dmu_integral(H_list[k], dH_list[k], dt, nodes, weights)
        Dk = Suff[k + 1] @ dUk @ Pref[k]
        zeta += float(np.real(np.trace(Uf.conj().T @ Dk * e_minus_i_phi)))
    return zeta / N


@dataclass
class MarginResult:
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
    # --- error control (populated when margin_tol is given) --------------
    # M is always a point with F >= FT, hence a LOWER bound on the true
    # margin.  M_upper is the nearest point known to violate F >= FT, so the
    # true margin lies in [M, M_upper] and margin_uncertainty bounds the error.
    M_upper_minus: float = float("inf")
    M_upper_plus: float = float("inf")
    M_upper: float = float("inf")
    margin_uncertainty: float = float("inf")
    reason_minus: str = "unknown"
    reason_plus: str = "unknown"
    #: 'segment' if every point between mu0 and mu_end is certified F >= FT
    #: (algorithm1, lipschitz_*); 'endpoint' if only the endpoint is
    #: (doubling, newton_probe -- these probe beyond the Lipschitz radius).
    certificate: str = "unknown"


@dataclass
class _EvalCounter:
    """Wrap a fidelity callable and count evaluations."""

    fn: Callable[[float], float]
    n_evals: int = 0

    def __call__(self, mu: float) -> float:
        self.n_evals += 1
        return float(self.fn(mu))


def _on_boundary(mu: float, mu_lo: float, mu_hi: float) -> bool:
    if np.isfinite(mu_lo) and abs(mu - mu_lo) <= max(1e-15, 10 * np.finfo(float).eps * abs(mu_lo)):
        return True
    if np.isfinite(mu_hi) and abs(mu - mu_hi) <= max(1e-15, 10 * np.finfo(float).eps * abs(mu_hi)):
        return True
    return False


def _clamp(mu: float, mu_lo: float, mu_hi: float) -> float:
    return min(max(mu, mu_lo), mu_hi)


def _bisect_safe(
    fidelity_fn: Callable[[float], float],
    mu_safe0: float,
    mu_bad: float,
    FT: float,
    eta: float,
) -> Tuple[float, float]:
    """Paper Algorithm 1 overshoot polish: keep a point with F >= FT."""
    a = mu_safe0
    b = mu_bad
    Fa = fidelity_fn(a)
    for _ in range(60):
        mid = 0.5 * (a + b)
        Fm = fidelity_fn(mid)
        if Fm >= FT:
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
) -> Tuple[float, float]:
    """Find a safe endpoint in [mu_safe, mu_bad] with 0 <= F - FT < eta when possible.

    Bracketed solvers locate F(mu) = FT, then step back by xtol toward
    the safe side so the returned point satisfies F >= FT (certificate side).
    """
    if root_solver == "bisection":
        return _bisect_safe(fidelity_fn, mu_safe, mu_bad, FT, eta)

    def g(mu: float) -> float:
        return fidelity_fn(mu) - FT

    g_safe = g(mu_safe)
    g_bad = g(mu_bad)
    if g_safe < 0:
        raise ValueError("mu_safe must satisfy F >= FT")
    if g_bad >= 0:
        return mu_safe, fidelity_fn(mu_safe)

    xtol = max(eta / 10.0, 1e-14 * max(1.0, abs(mu_safe), abs(mu_bad)))
    a, b = (mu_safe, mu_bad) if mu_safe < mu_bad else (mu_bad, mu_safe)
    if root_solver == "brent":
        root = brentq(g, a, b, xtol=xtol, maxiter=100)
    elif root_solver == "toms748":
        root = toms748(g, a, b, xtol=xtol, maxiter=100)
    else:
        raise ValueError(f"Unknown root_solver={root_solver!r}; expected one of {ROOT_SOLVERS}")

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
    if F_try >= FT:
        return mu_try, F_try
    # Fallback: classical bisection from the known safe endpoint.
    return _bisect_safe(fidelity_fn, mu_safe, mu_bad, FT, eta)


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
) -> Tuple[bool, bool, float]:
    """Return (done, converged, M)."""
    if _on_boundary(mu_next, mu_lo, mu_hi) and (F_next - FT >= eta):
        return True, True, abs(mu0 - mu_next)
    if 0 <= (F_next - FT) < eta:
        return True, True, abs(mu0 - mu_next)
    if k >= k_max:
        return True, False, abs(mu0 - mu_next)
    return False, True, abs(mu0 - mu_next)


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
) -> Tuple[float, bool, float, int]:
    """Certified Lipschitz advance; polish overshoot with root_solver."""
    sign_step = (-1) ** ell
    mu_lo, mu_hi = omega
    k = 0
    n_steps = 0
    mu = mu0
    Fmu = fidelity_fn(mu)
    mu_next = mu
    F_next = Fmu
    converged = True

    while True:
        mu_next = _clamp(mu + sign_step * (Fmu - FT) / L, mu_lo, mu_hi)
        F_next = fidelity_fn(mu_next)
        if F_next < FT:
            mu_next, F_next = _bracket_root_safe(fidelity_fn, mu, mu_next, FT, eta, root_solver)
        done, converged, M = _stop_one_direction(
            mu0, mu_next, F_next, FT, eta, mu_lo, mu_hi, k, k_max
        )
        if done:
            return M, converged, mu_next, n_steps
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
) -> Tuple[float, bool, float, int]:
    """Aggressive geometric probe beyond the Lipschitz radius, then bracket.

    Certificate is weaker than Algorithm 1 unless F is monotone on the ray:
    only the returned endpoint is guaranteed F >= FT, not the whole segment.
    """
    sign_step = (-1) ** ell
    mu_lo, mu_hi = omega
    n_steps = 0
    mu_safe = mu0
    F_safe = fidelity_fn(mu_safe)
    # Initial probe at least the certified Lipschitz step.
    step = max((F_safe - FT) / L, eta / max(L, 1e-30))
    mu_probe = _clamp(mu_safe + sign_step * step, mu_lo, mu_hi)
    F_probe = fidelity_fn(mu_probe)
    k = 0

    while F_probe >= FT:
        if _on_boundary(mu_probe, mu_lo, mu_hi):
            done, converged, M = _stop_one_direction(
                mu0, mu_probe, F_probe, FT, eta, mu_lo, mu_hi, k, k_max
            )
            return M, converged, mu_probe, n_steps
        if 0 <= (F_probe - FT) < eta:
            return abs(mu0 - mu_probe), True, mu_probe, n_steps
        if k >= k_max:
            return abs(mu0 - mu_probe), False, mu_probe, n_steps
        mu_safe = mu_probe
        F_safe = F_probe
        step *= 2.0
        mu_probe = _clamp(mu_safe + sign_step * step, mu_lo, mu_hi)
        if abs(mu_probe - mu_safe) <= 0.0:
            return abs(mu0 - mu_safe), True, mu_safe, n_steps
        F_probe = fidelity_fn(mu_probe)
        k += 1
        n_steps += 1

    mu_end, F_end = _bracket_root_safe(fidelity_fn, mu_safe, mu_probe, FT, eta, root_solver)
    return abs(mu0 - mu_end), True, mu_end, n_steps


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
    zeta_fn: Callable[[float], float],
) -> Tuple[float, bool, float, int]:
    """Safeguarded Newton-sized probes; beyond Lip radius behaves like doubling.

    When |zeta| is tiny or the Newton step exceeds the Lipschitz radius, the
    probe is treated as an aggressive (non-certified) jump and polished by
    bracketing on overshoot -- same guarantee class as ``doubling``.
    """
    sign_step = (-1) ** ell
    mu_lo, mu_hi = omega
    k = 0
    n_steps = 0
    mu = mu0
    Fmu = fidelity_fn(mu)
    mu_next = mu
    F_next = Fmu

    while True:
        lip_step = (Fmu - FT) / L
        zeta = float(zeta_fn(mu))
        if abs(zeta) > 1e-14:
            newt_step = abs((Fmu - FT) / zeta)
        else:
            newt_step = lip_step
        # Prefer Newton size when it is larger (aggressive); never smaller than Lip.
        step = max(lip_step, newt_step)
        mu_next = _clamp(mu + sign_step * step, mu_lo, mu_hi)
        F_next = fidelity_fn(mu_next)
        if F_next < FT:
            mu_next, F_next = _bracket_root_safe(fidelity_fn, mu, mu_next, FT, eta, root_solver)
            return abs(mu0 - mu_next), True, mu_next, n_steps
        done, converged, M = _stop_one_direction(
            mu0, mu_next, F_next, FT, eta, mu_lo, mu_hi, k, k_max
        )
        if done:
            return M, converged, mu_next, n_steps
        # If still far above FT after a large probe, double like ``doubling``.
        if step > lip_step * (1.0 + 1e-12) and (F_next - FT) >= eta:
            step2 = 2.0 * step
            mu_probe = _clamp(mu_next + sign_step * step2, mu_lo, mu_hi)
            F_probe = fidelity_fn(mu_probe)
            n_steps += 1
            if F_probe < FT:
                mu_next, F_next = _bracket_root_safe(
                    fidelity_fn, mu_next, mu_probe, FT, eta, root_solver
                )
                return abs(mu0 - mu_next), True, mu_next, n_steps
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
) -> Tuple[float, float, str]:
    """Bracket the true ray margin to a relative width of ``margin_tol``.

    ``mu_end`` is a point with ``F >= FT`` produced by one of the ray searches,
    so ``M = |mu0 - mu_end|`` is a *lower* bound on the ray margin.  The
    stopping rules of those searches are expressed in *fidelity* (``eta``), and
    the induced uncertainty in ``mu`` is ``~eta/|zeta|``, which is unbounded as
    ``zeta -> 0`` -- exactly the flat, highly robust controllers of interest.
    This routine converts that into a *margin* statement: step outward until
    ``F < FT``, then bisect, so the true ray margin lies in
    ``[M, M_upper]`` with ``M_upper - M <= margin_tol * max(M, tiny)``.

    Returns ``(M_refined, M_upper, reason)``.  ``M_refined >= M`` is the
    tightened safe end (still certified ``F >= FT``), and ``reason`` is
    ``'bracketed'`` (an unsafe point was located and refined),
    ``'boundary'`` (the domain edge was reached while still safe, so the margin
    is a domain truncation and ``M_upper`` is ``inf``), or ``'exhausted'``.
    """
    sign_step = (-1) ** ell
    mu_lo, mu_hi = omega
    scale = max(abs(mu_end - mu0), 1e-12)

    # Step outward geometrically to find a point with F < FT.
    mu_safe = mu_end
    mu_unsafe = None
    step = max(margin_tol * scale, 1e-15)
    for _ in range(200):
        cand = _clamp(mu_safe + sign_step * step, mu_lo, mu_hi)
        if cand == mu_safe:
            return abs(mu0 - mu_safe), float("inf"), "boundary"
        if fidelity_fn(cand) < FT:
            mu_unsafe = cand
            break
        mu_safe = cand
        step *= 2.0
    if mu_unsafe is None:
        return abs(mu0 - mu_safe), float("inf"), "exhausted"

    # Bisect the safe/unsafe bracket down to the requested relative width.
    target = max(margin_tol * max(abs(mu_safe - mu0), 1e-300), 1e-16 * max(1.0, abs(mu_safe)))
    for _ in range(200):
        if abs(mu_unsafe - mu_safe) <= target:
            break
        mid = 0.5 * (mu_safe + mu_unsafe)
        if mid == mu_safe or mid == mu_unsafe:
            break  # fp64 floor
        if fidelity_fn(mid) >= FT:
            mu_safe = mid
        else:
            mu_unsafe = mid
    return abs(mu0 - mu_safe), abs(mu0 - mu_unsafe), "bracketed"


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
) -> Tuple[float, bool, float, int]:
    if method == "algorithm1":
        return _one_direction_lipschitz(
            fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, "bisection"
        )
    if method == "lipschitz_brent":
        return _one_direction_lipschitz(fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, "brent")
    if method == "lipschitz_toms748":
        return _one_direction_lipschitz(fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, "toms748")
    if method == "doubling":
        rs = root_solver if root_solver != "bisection" else "toms748"
        return _one_direction_doubling(fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, rs)
    if method == "newton_probe":
        if zeta_fn is None:
            raise ValueError("method='newton_probe' requires zeta_fn")
        rs = root_solver if root_solver != "bisection" else "toms748"
        return _one_direction_newton_probe(
            fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, rs, zeta_fn
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
) -> MarginResult:
    """Certified (or exploratory) one-dimensional robustness margin.

    Preferred certified default is ``method='algorithm1'`` (Lipschitz advance +
    bisection). Lipschitz steps rarely overshoot, so Brent/TOMS748 polish does
    not meaningfully reduce fidelity evals versus bisection; keep them as
    optional polish only. Aggressive advance (``doubling`` / ``newton_probe``)
    can cut evals but drops the full segment certificate unless F is monotone
    on the ray. See ``docs/margin-solvers-notes.md``.

    Parameters
    ----------
    method :
        ``algorithm1`` (default) -- paper Algorithm 1: Lipschitz steps + bisection.
        ``lipschitz_brent`` / ``lipschitz_toms748`` -- same certified advance,
        Brent or TOMS748 overshoot polish (full Lipschitz certificate; little
        speed gain when overshoot is rare).
        ``doubling`` -- geometric probes beyond the Lipschitz radius, then
        bracket (weaker: endpoint-safe unless F is monotone on the ray).
        ``newton_probe`` -- Newton-sized probes via ``zeta_fn``; same guarantee
        class as ``doubling`` when probing beyond the Lipschitz radius.
    root_solver :
        Bracket polish for non-``algorithm1`` methods: ``toms748`` (default),
        ``brent``, or ``bisection``. Ignored when ``method='algorithm1'``.
    zeta_fn :
        Required for ``method='newton_probe'``; returns differential sensitivity
        zeta(mu).
    return_diagnostics :
        If True, populate ``n_evals``, ``n_steps``, and ``method`` on the result.
    """
    if L <= 0:
        raise ValueError("Lipschitz constant L must be positive")
    if method not in MARGIN_METHODS:
        raise ValueError(f"Unknown method={method!r}; expected one of {MARGIN_METHODS}")
    if root_solver not in ROOT_SOLVERS:
        raise ValueError(f"Unknown root_solver={root_solver!r}; expected one of {ROOT_SOLVERS}")

    counter = _EvalCounter(fidelity_fn)
    F0 = counter(mu0)
    if not (FT < F0):
        raise ValueError(f"Require FT < F(mu0); got FT={FT}, F={F0}")

    M_minus, conv_minus, mu_minus, steps_m = _dispatch_one_direction(
        counter, L, FT, mu0, eta, omega, k_max, 1, method, root_solver, zeta_fn
    )
    M_plus, conv_plus, mu_plus, steps_p = _dispatch_one_direction(
        counter, L, FT, mu0, eta, omega, k_max, 2, method, root_solver, zeta_fn
    )
    result = MarginResult(
        M_minus=M_minus,
        M_plus=M_plus,
        M=min(M_minus, M_plus),
        converged_minus=conv_minus,
        converged_plus=conv_plus,
        mu_minus=mu_minus,
        mu_plus=mu_plus,
        method=method,
        certificate="segment"
        if method in ("algorithm1", "lipschitz_brent", "lipschitz_toms748")
        else "endpoint",
    )
    if margin_tol is not None:
        if not (margin_tol > 0):
            raise ValueError("margin_tol must be positive")
        lo_m, up_m, why_m = _certify_direction(counter, mu0, mu_minus, FT, 1, omega, margin_tol)
        lo_p, up_p, why_p = _certify_direction(counter, mu0, mu_plus, FT, 2, omega, margin_tol)
        # The refined safe ends are tighter lower bounds than the eta-based ones.
        result.M_minus = max(result.M_minus, lo_m)
        result.M_plus = max(result.M_plus, lo_p)
        result.M = min(result.M_minus, result.M_plus)
        result.M_upper_minus = up_m
        result.M_upper_plus = up_p
        result.reason_minus = why_m
        result.reason_plus = why_p
        result.M_upper = min(up_m, up_p)
        result.margin_uncertainty = result.M_upper - result.M
    if return_diagnostics:
        result.n_evals = counter.n_evals
        result.n_steps = steps_m + steps_p
    return result


def fidelity_vs_delta(
    fidelity_fn: Callable[[float], float],
    delta_grid: Iterable[float],
) -> Tuple[Array, Array]:
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
    def fn(delta: float) -> float:
        H_list = perturbed_hamiltonians(H0, H1, H2, u1, u2, structure, delta)
        U = propagator(H_list, dt)
        return gate_fidelity(U, Uf)

    return fn


def load_problem(mat_path: Union[str, Path]) -> dict:
    S = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "problem" not in S:
        raise KeyError("Expected variable 'problem'")
    p = S["problem"]
    H = p.H
    # H may be object array of length 3
    H0, H1, H2 = H[0], H[1], H[2]
    # problem.N in the MAT file is the number of qubits; the Hilbert space
    # dimension is 2**N.  Return both under unambiguous names.
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
        # MATLAB reshape(data, 2, tau) is column-major: interleaved u1,u2 samples.
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

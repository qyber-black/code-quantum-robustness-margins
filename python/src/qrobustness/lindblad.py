# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Open-system (Lindblad) robustness margins on the process fidelity F^pro.

The module builds column-stacking superoperators, the piecewise-constant
channel, F^pro and the Choi conversions. Diamond-norm upper bounds are
verified: the Watrous SDP through cvxpy, a solver-free variant, and closed
forms. The per-parameter constants are
L_j = 0.5 sum_k Delta ||G_j^(k)||_diamond. Scalar open-system margins use
those constants. The optional ``qrobustness[open]`` extra supplies cvxpy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm

from .core import Array, iterative_margin, MarginResult

__all__ = [
    "hamiltonian_superop",
    "dissipator",
    "generator",
    "channel",
    "unitary_superop",
    "process_fidelity",
    "average_gate_fidelity",
    "frechet_derivative",
    "choi_matrix",
    "superop_from_choi",
    "choi_roundtrip_exact",
    "DiamondNorm",
    "VerificationFailure",
    "common_rate_local_dnorm",
    "hamiltonian_part",
    "hamiltonian_dnorm",
    "diamond_norm",
    "diamond_norm_free",
    "open_structure_constants",
    "make_open_fidelity_fn",
    "open_margin",
    "dephasing_time",
    "relaxation_time",
    "coherence_time",
    "rates_from_times",
]


# --------------------------------------------------------------------------
# Builders (column-stacking convention)
# --------------------------------------------------------------------------


def hamiltonian_superop(H: Array) -> Array:
    """Superoperator of ``-1j*[H, .]``."""
    N = H.shape[0]
    Id = np.eye(N)
    return -1j * (np.kron(Id, H) - np.kron(H.T, Id))


def dissipator(V: Array) -> Array:
    """Lindblad dissipator superoperator ``D[V]`` taken at unit rate."""
    N = V.shape[0]
    Id = np.eye(N)
    VdV = V.conj().T @ V
    return np.kron(V.conj(), V) - 0.5 * (np.kron(Id, VdV) + np.kron(VdV.T, Id))


def generator(
    H: Array, Vs: Sequence[Array] = (), gammas: Sequence[float] = ()
) -> Array:
    """Lindblad generator ``-1j[H, .] + sum_l gamma_l D[V_l]``.

    Parameters
    ----------
    H : (N, N) Hamiltonian.
    Vs : jump operators V_l, each (N, N).
    gammas : rates gamma_l; must have the same length as ``Vs``.

    Returns
    -------
    (N^2, N^2) generator superoperator.
    """
    G = hamiltonian_superop(H)
    for V, g in zip(Vs, gammas, strict=True):
        G = G + g * dissipator(V)
    return G


def channel(G_list: Sequence[Array], dt: float) -> Array:
    """Channel ``S = prod_k expm(dt*G^(k))`` of a piecewise-constant generator.

    Parameters
    ----------
    G_list : per-interval generators G^(k), each (N^2, N^2); must be non-empty.
    dt : interval length Delta.

    Returns
    -------
    (N^2, N^2) superoperator, later intervals applied on the left.
    """
    if len(G_list) == 0:
        raise ValueError("G_list must not be empty")
    N2 = G_list[0].shape[0]
    S = np.eye(N2, dtype=complex)
    for G in G_list:
        S = expm(dt * G) @ S
    return S


def unitary_superop(U: Array) -> Array:
    """Superoperator for the unitary channel ``rho -> U rho U^dag``."""
    return np.kron(U.conj(), U)


def process_fidelity(S: Array, Uf: Array) -> float:
    """Process fidelity ``F^pro = Re Tr(S_f^dag S) / N^2``, ``S_f = conj(Uf) kron Uf``.

    Parameters
    ----------
    S : (N^2, N^2) channel superoperator.
    Uf : (N, N) unitary target.

    Returns
    -------
    F^pro as a float.
    """
    N = Uf.shape[0]
    Sf = unitary_superop(Uf)
    return float(np.real(np.trace(Sf.conj().T @ S)) / N**2)


def average_gate_fidelity(S: Array, Uf: Array) -> float:
    """Average gate fidelity ``(N F^pro + 1)/(N + 1)`` of ``S`` against ``Uf``."""
    N = Uf.shape[0]
    return (N * process_fidelity(S, Uf) + 1.0) / (N + 1.0)


def frechet_derivative(G: Array, E: Array, dt: float) -> Array:
    """Frechet derivative ``d/ds expm(dt*(G + s E))`` at ``s = 0``.

    Uses the upper-right block of ``expm(dt*[[G, E], [0, G]])``, valid for
    defective Lindblad generators (no eigenbasis method is used).

    Parameters
    ----------
    G, E : (n, n) generator and direction.
    dt : interval length.

    Returns
    -------
    (n, n) derivative.
    """
    n = G.shape[0]
    M = np.zeros((2 * n, 2 * n), dtype=complex)
    M[:n, :n] = G
    M[:n, n:] = E
    M[n:, n:] = G
    return expm(dt * M)[:n, n:]


# --------------------------------------------------------------------------
# Diamond norm (Watrous SDP)
# --------------------------------------------------------------------------


def choi_matrix(S: Array) -> Array:
    """Choi matrix ``J = sum_ij Phi(E_ij) kron E_ij`` (output kron input).

    Built by exact reindexing of ``S`` (no arithmetic), so a bound on ``J``
    applies to the stored superoperator; :func:`superop_from_choi` inverts it.

    Parameters
    ----------
    S : (N^2, N^2) column-stacking superoperator.

    Returns
    -------
    (N^2, N^2) Choi matrix.
    """
    N = int(round(np.sqrt(S.shape[0])))
    S4 = np.asarray(S).reshape(N, N, N, N)
    return S4.transpose(1, 3, 0, 2).reshape(N * N, N * N).copy()


def superop_from_choi(J: Array, N: Optional[int] = None) -> Array:
    """Superoperator from a Choi matrix; exact inverse of :func:`choi_matrix`.

    Parameters
    ----------
    J : (N^2, N^2) Choi matrix.
    N : Hilbert-space dimension; inferred from ``J`` if None.

    Returns
    -------
    (N^2, N^2) superoperator.
    """
    if N is None:
        N = int(round(np.sqrt(J.shape[0])))
    J4 = np.asarray(J).reshape(N, N, N, N)
    return J4.transpose(2, 0, 3, 1).reshape(N * N, N * N).copy()


def choi_roundtrip_exact(S: Array) -> bool:
    """True if ``superop_from_choi(choi_matrix(S))`` equals ``S`` bit for bit."""
    back = superop_from_choi(choi_matrix(S), int(round(np.sqrt(S.shape[0]))))
    return bool(np.array_equal(np.asarray(S), back))


@dataclass
class DiamondNorm:
    """Diamond-norm bound returned by the functions of this module.

    Attributes
    ----------
    value : equal to ``value_certified``; use this as the constant.
    raw : solver optimum (or floating objective) before verification.
    gap : ``value - raw``.
    status : solver status, ``"solver_free"``, or ``"analytic"`` (closed form).
    value_certified : verified upper bound on the diamond norm under IEEE-754
        rounding (Rump floating-Cholesky feasibility, upward-rounded
        objective); never returned unverified (VerificationFailure instead).
    feas_shift : diagonal shift that made the SDP iterate verifiably feasible.
    solver : name of the solver or closed form that produced ``raw``.
    """

    value: float
    raw: float
    gap: float
    status: str
    value_certified: float = float("nan")
    feas_shift: float = float("nan")
    solver: str = "solver_free"


#: Inflation of Rump's real Cholesky backward-error constant for complex arithmetic.
_CHOL_COMPLEX_FACTOR = 8


def _up(x: float) -> float:
    """The next float above ``x``: one rounding-up step."""
    return float(np.nextafter(float(x), np.inf))


def _chol_error_bound(A: Array) -> Optional[float]:
    """Verified bound ``c >= ||Delta||_2`` on the Cholesky backward error of ``A``.

    Conservative form of Rump's bound, including the diagonal inflation and
    the underflow term, with all operations rounded upward.

    Returns
    -------
    The bound, or None if a diagonal entry is negative (``A`` not PSD) or the
    dimension is too large for the bound.
    """
    n = A.shape[0]
    u = np.finfo(float).eps / 2.0
    k = _CHOL_COMPLEX_FACTOR * (n + 1)
    if k * u >= 0.5:
        return None
    alpha = _up(k * u / (1.0 - k * u))
    # alpha/(1 - alpha) absorbs the diagonal inflation 1/(1 - alpha_ii).
    coef = _up(alpha / (1.0 - alpha))
    d = np.real(np.diag(A))
    if np.any(d < 0.0):
        return None
    v = np.nextafter(np.sqrt(d), np.inf)  # sqrt is correctly rounded
    g = (n - 1) * u / (1.0 - (n - 1) * u) if n > 1 else 0.0
    tot = _up(float(np.sum(v)) * (1.0 + g))
    main = _up(_up(coef * float(np.max(v))) * tot)
    # Underflow allowance: n entries per row, M eta each.
    eta = np.finfo(float).tiny
    under = _up(_up(float(n) * float(4 * (n + 1))) * eta)
    return _up(main + under)


def _shift_diag_down(a: float, c: float) -> Optional[float]:
    """Largest double ``t`` with ``t <= a - c`` checked in exact rational arithmetic.

    Returns None if none is found within a few steps (non-finite input).
    """
    from fractions import Fraction

    fa, fc = Fraction(float(a)), Fraction(float(c))
    t = float(a) - float(c)
    for _ in range(8):
        if Fraction(t) <= fa - fc:
            return t
        t = float(np.nextafter(t, -np.inf))
    return None


def _verify_psd(A: Array) -> bool:
    """Verify that stored Hermitian ``A`` is PSD (Rump floating-Cholesky test).

    The diagonal is lowered by a verified backward-error bound ``c`` and a
    floating Cholesky of the result is attempted; success proves ``A >= 0``.

    Returns
    -------
    True if ``A >= 0`` is proved, False otherwise (not a proof of indefiniteness).
    """
    A = np.asarray(A)
    if not np.all(np.isfinite(A.view(float))):
        return False
    c = _chol_error_bound(A)
    if c is None:
        return False
    A_test = np.array(A, dtype=complex, copy=True)
    for i in range(A.shape[0]):
        t = _shift_diag_down(float(np.real(A[i, i])), c)
        if t is None:
            return False
        A_test[i, i] = t
    # c was computed from A; check that it also bounds the error for A_test.
    c_test = _chol_error_bound(A_test)
    if c_test is None or not (c_test <= c):
        return False
    try:
        np.linalg.cholesky(A_test)
    except np.linalg.LinAlgError:
        return False
    return True


class VerificationFailure(RuntimeError):
    """Raised when a diamond-norm or spectral bound cannot be verified."""


def _verified_psd_repair(
    Y0: Array,
    Y1: Array,
    J: Array,
    shift0: float,
    max_tries: int = 60,
) -> Tuple[Array, Array, float]:
    """Shift ``(Y0, Y1)`` by ``eps I`` until the Watrous block is verified PSD.

    ``shift0`` is the proposed starting shift; it grows geometrically for up to
    ``max_tries`` attempts, then :class:`VerificationFailure` is raised.

    Returns
    -------
    The stored repaired blocks ``(Y0 + eps I, Y1 + eps I)`` and ``eps``; the
    objective must be evaluated on exactly these blocks.
    """
    d = Y0.shape[0]
    eye = np.eye(d)
    u = np.finfo(float).eps / 2.0
    scale = float(np.max(np.abs(J))) if J.size else 1.0
    eps = max(float(shift0), 0.0) * (1.0 + 16 * u)
    floor = max(scale, 1.0) * 8.0 * u
    if eps <= 0.0:
        eps = floor
    for _ in range(max_tries):
        Y0r = Y0 + eps * eye
        Y1r = Y1 + eps * eye
        B = np.block([[Y0r, -J], [-J.conj().T, Y1r]])
        if _verify_psd(B):
            return Y0r, Y1r, float(eps)
        eps = 2.0 * eps + floor
    raise VerificationFailure(
        "no verified positive-semidefinite repair of the SDP iterate was found"
    )


def _sum_upward(values) -> float:
    """Upper bound on a sum of non-negative floats (summation error enclosed)."""
    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return 0.0
    u = np.finfo(float).eps / 2.0
    g = (n - 1) * u / (1.0 - (n - 1) * u) if n > 1 else 0.0
    return float(np.sum(v)) * (1.0 + g)


def _specnorm_upper(A: Array) -> float:
    """Upper bound on ``||A||_2`` for Hermitian ``A`` (Gershgorin, rounding enclosed)."""
    n = A.shape[0]
    u = np.finfo(float).eps / 2.0
    rows = np.sum(np.abs(A), axis=1)
    return float(np.max(rows)) * (1.0 + (n + 1) * u)


def _partial_trace_specnorm_upper(Y: Array, N: int) -> float:
    """Upper bound on ``||Tr_out Y||_2`` for Hermitian ``Y`` on output kron input.

    Encloses the partial-trace summation and complex-magnitude rounding,
    then bounds the spectral norm by Gershgorin row sums.
    """
    u = np.finfo(float).eps / 2.0
    Yr = Y.reshape(N, N, N, N)
    M = np.trace(Yr, axis1=0, axis2=2)
    A = np.trace(np.abs(Yr), axis1=0, axis2=2)
    ent = np.abs(M) * (1.0 + 4.0 * u) + (N - 1) * u * A * (1.0 + 4.0 * u)
    rows = np.sum(ent, axis=1)
    return float(np.max(rows)) * (1.0 + (N + 1) * u)


def diamond_norm_free(S: Array, iters: int = 600, degtol: float = 1e-6) -> DiamondNorm:
    """Verified diamond-norm upper bound without an external SDP solver.

    Starts from a closed-form feasible point of the Watrous program, improves
    it by projected subgradient steps, then repairs and verifies feasibility
    as in :func:`diamond_norm`. The result may be conservative (not tight)
    on degenerate spectra.

    Parameters
    ----------
    S : (N^2, N^2) superoperator.
    iters : maximum subgradient iterations.
    degtol : relative tolerance for grouping degenerate top eigenvalues.

    Returns
    -------
    DiamondNorm with ``status="solver_free"``; ``value_certified`` is a verified
    upper bound, ``raw`` the floating objective at the repaired point.
    """
    N = int(round(np.sqrt(S.shape[0])))
    J = choi_matrix(S)
    d = N * N

    def psd_sqrt(A: Array) -> Array:
        A = (A + A.conj().T) / 2
        w, V = np.linalg.eigh(A)
        return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.conj().T

    def tr_out(Y: Array) -> Array:
        return np.einsum("ijik->jk", Y.reshape(N, N, N, N))

    def objective(Y0: Array, Y1: Array) -> float:
        return 0.5 * (
            float(np.linalg.norm(tr_out(Y0), 2)) + float(np.linalg.norm(tr_out(Y1), 2))
        )

    def subgrad(Y: Array) -> Array:
        T = tr_out(Y)
        T = (T + T.conj().T) / 2
        w, V = np.linalg.eigh(T)
        i = int(np.argmax(np.abs(w)))
        lam = abs(w[i])
        sgn = np.sign(w[i]) or 1.0
        sel = np.where(np.abs(np.abs(w) - lam) <= degtol * max(lam, 1.0))[0]
        P = sum(np.outer(V[:, j], V[:, j].conj()) for j in sel) / len(sel)
        return np.kron(np.eye(N), sgn * P)

    def project(Y0: Array, Y1: Array, rounds: int = 30):
        for _ in range(rounds):
            B = np.block([[Y0, -J], [-J.conj().T, Y1]])
            B = (B + B.conj().T) / 2
            w, V = np.linalg.eigh(B)
            if w.min() >= -1e-14:
                break
            B = (V * np.clip(w, 0.0, None)) @ V.conj().T
            Y0, Y1 = B[:d, :d], B[d:, d:]
        return Y0, Y1

    Y0 = psd_sqrt(J @ J.conj().T)
    Y1 = psd_sqrt(J.conj().T @ J)
    a = float(np.linalg.norm(tr_out(Y0), 2))
    b = float(np.linalg.norm(tr_out(Y1), 2))
    scale = np.sqrt(b / a) if a > 0 else 1.0
    Y0, Y1 = scale * Y0, Y1 / (scale if scale > 0 else 1.0)
    start = objective(Y0, Y1)

    best = start
    step = 0.1 * max(best, 1e-12)
    for _ in range(iters):
        n0, n1 = project(Y0 - step * 0.5 * subgrad(Y0), Y1 - step * 0.5 * subgrad(Y1))
        val = objective(n0, n1)
        if val < best - 1e-15:
            Y0, Y1, best = n0, n1, val
        else:
            step *= 0.7
            if step < 1e-13 * max(best, 1.0):
                break

    # The eigensolver only proposes the shift; the stored blocks are verified.
    Y0 = (Y0 + Y0.conj().T) / 2
    Y1 = (Y1 + Y1.conj().T) / 2
    B = np.block([[Y0, -J], [-J.conj().T, Y1]])
    lam_min = float(np.linalg.eigvalsh((B + B.conj().T) / 2).min())
    Y0r, Y1r, shift = _verified_psd_repair(Y0, Y1, J, max(0.0, -lam_min))
    value = objective(Y0r, Y1r)
    certified = 0.5 * _sum_upward(
        [
            _partial_trace_specnorm_upper(Y0r, N),
            _partial_trace_specnorm_upper(Y1r, N),
        ]
    )
    return DiamondNorm(
        value=certified,
        raw=value,
        gap=float(certified - value),
        status="solver_free",
        value_certified=certified,
        feas_shift=shift,
    )


#: Default cvxpy solver for the Watrous SDP. Named explicitly because cvxpy's
#: automatic choice varies by environment and changes ``raw`` (not soundness).
DEFAULT_SDP_SOLVER = "CLARABEL"


def diamond_norm(S: Array, solver: Optional[str] = None) -> DiamondNorm:
    """Verified diamond-norm upper bound of ``S`` via the Watrous SDP (needs cvxpy).

    Parameters
    ----------
    S : (N^2, N^2) superoperator (any linear map).
    solver : cvxpy solver name; None uses ``DEFAULT_SDP_SOLVER``, ``""`` lets
        cvxpy choose.

    Returns
    -------
    DiamondNorm; ``value_certified`` is a verified upper bound, ``raw`` the
    solver optimum. Raises RuntimeError if the SDP fails and
    VerificationFailure if feasibility cannot be verified.
    """
    import cvxpy as cp

    N = int(round(np.sqrt(S.shape[0])))
    J = choi_matrix(S)
    d = N * N
    Y0 = cp.Variable((d, d), hermitian=True)
    Y1 = cp.Variable((d, d), hermitian=True)
    block = cp.bmat([[Y0, -J], [-J.conj().T, Y1]])
    constraints = [block >> 0]
    tr0 = cp.partial_trace(Y0, [N, N], axis=0)  # trace out the output system
    tr1 = cp.partial_trace(Y1, [N, N], axis=0)
    objective = 0.5 * (cp.norm(tr0, 2) + cp.norm(tr1, 2))
    # norm(., 2) on a Hermitian matrix variable is the spectral norm inside cvxpy
    prob = cp.Problem(cp.Minimize(objective), constraints)
    chosen = DEFAULT_SDP_SOLVER if solver is None else solver
    kwargs = {"solver": chosen} if chosen else {}
    prob.solve(**kwargs)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"diamond norm SDP failed: {prob.status}")
    # The solver number is diagnostic. value and value_certified come from
    # the verified repair below, including when the status is
    # optimal_inaccurate.
    raw = float(prob.value)

    Y0v = np.asarray(Y0.value)
    Y1v = np.asarray(Y1.value)
    Y0v = (Y0v + Y0v.conj().T) / 2
    Y1v = (Y1v + Y1v.conj().T) / 2
    B = np.block([[Y0v, -J], [-J.conj().T, Y1v]])
    B = (B + B.conj().T) / 2
    lam_min = float(np.min(np.linalg.eigvalsh(B)))
    # The eigensolver only proposes the shift; the stored blocks are verified.
    Y0r, Y1r, eps = _verified_psd_repair(Y0v, Y1v, J, max(0.0, -lam_min))

    # Primal objective (upper bound on the minimum), rounded upward.
    up0 = _partial_trace_specnorm_upper(Y0r, N)
    up1 = _partial_trace_specnorm_upper(Y1r, N)
    certified = 0.5 * _sum_upward([up0, up1])
    gap = certified - raw
    return DiamondNorm(
        value=certified,
        raw=raw,
        gap=gap,
        status=prob.status,
        value_certified=certified,
        feas_shift=eps,
        solver=str(prob.solver_stats.solver_name),
    )


def common_rate_local_dnorm(n_qubits: int) -> DiamondNorm:
    """Diamond norm ``2n`` of ``sum_q D[V_q]`` for local dephasing or amplitude damping.

    Exact closed form for ``V_q = sigma_z^(q)`` or ``sigma_-^(q)`` on ``n``
    qubits (see the xQRM paper); no SDP is needed.

    Parameters
    ----------
    n_qubits : number of qubits n >= 1.

    Returns
    -------
    DiamondNorm with ``value = value_certified = 2n`` and ``status="analytic"``.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    v = float(2 * n_qubits)
    return DiamondNorm(
        value=v,
        raw=v,
        gap=0.0,
        status="analytic",
        value_certified=v,
        feas_shift=0.0,
        solver="closed_form_2n",
    )


def hamiltonian_part(S: Array, rtol: float = 1e-12) -> Optional[Array]:
    """Traceless Hermitian ``B`` with ``S = -1j[B, .]``, or None if there is none.

    Parameters
    ----------
    S : (N^2, N^2) superoperator.
    rtol : relative Frobenius tolerance for accepting ``S`` as Hamiltonian.

    Returns
    -------
    (N, N) matrix ``B``, or None (e.g. for a generator with a dissipative part).
    """
    S = np.asarray(S, dtype=complex)
    n2 = S.shape[0]
    N = int(round(np.sqrt(n2)))
    if N * N != n2 or S.shape != (n2, n2):
        raise ValueError("S must be a square superoperator of size N^2")
    blocksum = sum(S[a * N : (a + 1) * N, a * N : (a + 1) * N] for a in range(N))
    B = 1j * blocksum / N
    B = (B + B.conj().T) / 2
    scale = float(np.linalg.norm(S))
    if scale == 0.0:
        return B
    if float(np.linalg.norm(S - hamiltonian_superop(B))) > rtol * scale:
        return None
    return B


def hamiltonian_dnorm(S: Array, rtol: float = 1e-12) -> DiamondNorm:
    """Verified diamond norm of a Hamiltonian superoperator ``-1j[B, .]``.

    Uses ``||-1j[B, .]||_diamond = lambda_max(B) - lambda_min(B)``, with the
    eigenvalue bounds verified by the Rump PSD test and a term added for the
    difference between ``S`` and the superoperator of the extracted ``B``.

    Parameters
    ----------
    S : (N^2, N^2) Hamiltonian superoperator.
    rtol : tolerance passed to :func:`hamiltonian_part`.

    Returns
    -------
    DiamondNorm; ``value_certified`` is a verified upper bound, ``raw`` the
    computed spectral spread. Raises ValueError if ``S`` is not Hamiltonian
    and VerificationFailure if the spectral bounds do not verify.
    """
    S = np.asarray(S, dtype=complex)
    B = hamiltonian_part(S, rtol)
    if B is None:
        raise ValueError("S is not a Hamiltonian superoperator")
    N = B.shape[0]
    u = np.finfo(float).eps / 2.0
    w = np.linalg.eigvalsh(B)
    raw = float(w[-1] - w[0])
    scale = max(float(np.max(np.abs(w))), np.finfo(float).tiny)
    slack = 64.0 * N * u * scale
    hi = lo = None
    for _ in range(12):
        h_c = _up(float(w[-1]) + slack)
        l_c = float(np.nextafter(float(w[0]) - slack, -np.inf))
        A_hi = -np.array(B, copy=True)
        A_lo = np.array(B, copy=True)
        ok = True
        for i in range(N):
            bii = float(np.real(B[i, i]))
            t_hi = _shift_diag_down(h_c, bii)
            t_lo = _shift_diag_down(bii, l_c)
            if t_hi is None or t_lo is None:
                ok = False
                break
            A_hi[i, i] = t_hi
            A_lo[i, i] = t_lo
        if ok and _verify_psd(A_hi) and _verify_psd(A_lo):
            hi, lo = h_c, l_c
            break
        slack *= 4.0
    if hi is None:
        raise VerificationFailure("spectral bounds of the Hamiltonian did not verify")
    spread = _up(hi - lo)
    # Representation term: N ||S - S_B||_F, upward-bounded.
    SB = hamiltonian_superop(B)
    diag_err = 2.0 * u * 2.0 * float(np.max(np.abs(np.real(np.diag(B))))) * N
    fro = float(np.linalg.norm(S - SB))
    fro = _up(_up(fro * (1.0 + 2.0 * _gamma(N * N))) + diag_err)
    rep = _up(N * fro)
    v = _up(spread + rep)
    return DiamondNorm(
        value=v,
        raw=raw,
        gap=_up(v - raw) if v >= raw else 0.0,
        status="analytic",
        value_certified=v,
        feas_shift=0.0,
        solver="closed_form_spread",
    )


def _gamma(m: int) -> float:
    """Summation error factor ``gamma_m = m u / (1 - m u)``, rounded up."""
    u = np.finfo(float).eps / 2.0
    return _up(m * u / (1.0 - m * u))


# --------------------------------------------------------------------------
# Coherence times
# --------------------------------------------------------------------------


def relaxation_time(gamma_minus: float) -> float:
    """``T_1 = 1/gamma_-``; a zero rate gives an infinite time."""
    g = float(gamma_minus)
    if g < 0.0:
        raise ValueError("gamma_minus must be non-negative")
    return float("inf") if g == 0.0 else 1.0 / g


def dephasing_time(gamma_z: float) -> float:
    """Pure-dephasing time ``T_phi = 1/(2 gamma_z)``; infinite for a zero rate.

    This equals ``T_2`` only without amplitude damping; see :func:`coherence_time`.
    """
    g = float(gamma_z)
    if g < 0.0:
        raise ValueError("gamma_z must be non-negative")
    return float("inf") if g == 0.0 else 1.0 / (2.0 * g)


def coherence_time(gamma_z: float, gamma_minus: float = 0.0) -> float:
    """Total coherence time ``T_2`` with ``1/T_2 = 2 gamma_z + gamma_-/2``.

    Parameters
    ----------
    gamma_z : dephasing rate (non-negative).
    gamma_minus : amplitude-damping rate (non-negative).

    Returns
    -------
    ``T_2``; infinite if both rates are zero.
    """
    gz, gm = float(gamma_z), float(gamma_minus)
    if gz < 0.0 or gm < 0.0:
        raise ValueError("rates must be non-negative")
    inv = 2.0 * gz + 0.5 * gm
    return float("inf") if inv == 0.0 else 1.0 / inv


def rates_from_times(T1: float, T2: float) -> Tuple[float, float]:
    """Rates ``(gamma_z, gamma_-)`` from ``T_1``, ``T_2``; inverts :func:`coherence_time`.

    ``gamma_z = 1/(2 T_2) - 1/(4 T_1)`` and ``gamma_- = 1/T_1``, defined for
    ``T_2 <= 2 T_1``; infinite times give zero rates. A budget on both rates
    is a joint constraint, not two separate one-dimensional intercepts.

    Parameters
    ----------
    T1, T2 : positive times (may be inf).

    Returns
    -------
    ``(gamma_z, gamma_minus)``.
    """
    t1, t2 = float(T1), float(T2)
    if t1 <= 0.0 or t2 <= 0.0:
        raise ValueError("T1 and T2 must be positive")
    if t2 > 2.0 * t1 * (1.0 + 1e-12):
        raise ValueError("unphysical pair: require T2 <= 2 T1")
    gm = 0.0 if np.isinf(t1) else 1.0 / t1
    gz = (0.0 if np.isinf(t2) else 0.5 / t2) - 0.25 * gm
    return float(max(gz, 0.0)), float(gm)


# --------------------------------------------------------------------------
# Open-system structure constants and margins
# --------------------------------------------------------------------------


def open_structure_constants(
    G_structs: Sequence[Sequence[Array]],
    dt: float,
    solver: Optional[str] = None,
) -> Tuple[Array, List[List[DiamondNorm]]]:
    """Per-parameter constants ``L_j = 0.5 * sum_k Delta * ||G_j^(k)||_diamond``.

    Parameters
    ----------
    G_structs : ``G_structs[j]`` lists the structure generators G_j^(k) of
        parameter j over the intervals.
    dt : interval length Delta (for a single time-independent generator pass
        the gate time t_f, giving ``0.5 * t_f * ||G_j||_diamond``).
    solver : passed to :func:`diamond_norm`.

    Returns
    -------
    L : (p,) array of L_j (from verified diamond-norm upper bounds).
    diags : per-parameter lists of the DiamondNorm results.
    """
    L = []
    diags: List[List[DiamondNorm]] = []
    for structs in G_structs:
        norms = [diamond_norm(G, solver=solver) for G in structs]
        diags.append(norms)
        dts = np.asarray(dt, dtype=float)
        if dts.ndim == 0:
            dts = np.full(len(structs), float(dts))
        if dts.shape != (len(structs),):
            raise ValueError("dt must be a scalar or one length per interval")
        L.append(
            0.5 * float(sum(dk * n.value for dk, n in zip(dts, norms, strict=True)))
        )
    return np.asarray(L, dtype=float), diags


def rate_lipschitz(dnorm_value: float, t_f: float) -> float:
    """Lipschitz constant ``0.5 * t_f * ||G||_diamond`` of F^pro in a constant rate.

    Parameters
    ----------
    dnorm_value : diamond norm of the structure generator (pass
        ``value_certified`` for a certified constant).
    t_f : gate time.

    Returns
    -------
    The constant L.
    """
    return 0.5 * t_f * dnorm_value


def make_open_fidelity_fn(
    build_generators: Callable[[Array], Sequence[Array]],
    dt: float,
    Uf: Array,
) -> Callable[[Array], float]:
    """F^pro as a function of the open-system parameter vector ``mu``.

    Parameters
    ----------
    build_generators : maps ``mu`` to the per-interval generators G^(k)(mu).
    dt : interval length Delta.
    Uf : (N, N) unitary target.

    Returns
    -------
    Callable ``mu -> F^pro``.
    """

    def fn(mu: Array) -> float:
        G_list = build_generators(np.atleast_1d(np.asarray(mu, dtype=float)))
        return process_fidelity(channel(G_list, dt), Uf)

    return fn


def open_margin(
    fidelity_fn: Callable[[float], float],
    L: float,
    FT_pro: float,
    omega: Tuple[float, float] = (0.0, np.inf),
    **kwargs,
) -> MarginResult:
    """Certified margin M of a scalar open-system parameter (e.g. a rate).

    Wrapper over :func:`qrobustness.iterative_margin` with default domain
    ``[0, inf)``.

    Parameters
    ----------
    fidelity_fn : scalar parameter -> F^pro.
    L : Lipschitz constant (e.g. from :func:`rate_lipschitz`).
    FT_pro : threshold F_T on F^pro.
    omega : admissible parameter interval.
    **kwargs : passed to :func:`qrobustness.iterative_margin`.

    Returns
    -------
    MarginResult (see :func:`qrobustness.iterative_margin`).
    """
    return iterative_margin(fidelity_fn, L, FT_pro, omega=omega, **kwargs)


def local_ops(op: Array, n_qubits: int) -> list:
    """Single-qubit ``op`` embedded on each site of an ``n_qubits`` register.

    Returns
    -------
    List of ``2^n x 2^n`` operators ``I x .. x op x .. x I`` in site order.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    eye = np.eye(2, dtype=complex)
    out = []
    for q in range(n_qubits):
        factors = [eye] * n_qubits
        factors[q] = op
        M = factors[0]
        for f in factors[1:]:
            M = np.kron(M, f)
        out.append(M)
    return out


def local_dephasing_ops(n_qubits: int) -> list:
    """``sigma_z`` on each qubit of an ``n_qubits`` register (see :func:`local_ops`)."""
    return local_ops(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex), n_qubits)

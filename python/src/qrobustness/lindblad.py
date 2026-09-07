# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Open-system certified margins for piecewise-constant Lindblad dynamics.

Density operators are vectorised by column stacking, ``vec(A X B) =
(B^T kron A) vec(X)``.  The generator

    G = -1j*(kron(I, H) - kron(H^T, I)) + sum_l gamma_l D[V_l],
    D[V] = kron(conj(V), V)
           - 0.5*(kron(I, V^dag V) + kron((V^dag V)^T, I)),

is affine in both the coherent amplitudes (through ``H``) and the rates
``gamma_l``.  The channel is the ordered product of interval propagators
``expm(dt * G^(k))`` and the performance measure is the process fidelity

    F_pro(S) = Tr(S_f^dag S) / N^2,   S_f = kron(conj(U_f), U_f),

which is linear in ``S``, real and non-negative for CPTP ``S``, and equals
``|Tr(U_f^dag U)|^2 / N^2`` in the closed-system limit.

Lipschitz constants come from the diamond norm: CPTP invariance of the
diamond norm replaces unitary invariance of the Frobenius norm, giving

    |F_pro(mu) - F_pro(nu)| <= sum_j L_j |mu_j - nu_j|,
    L_j = 0.5 * sum_k dt * dnorm(G_j^(k)),

with the diamond norms of the fixed structure generators computed once by
the Watrous semidefinite program (:func:`diamond_norm`; requires the
optional ``qrobustness[open]`` dependency set).  The margins themselves
reuse the certified scalar iteration unchanged.

Error control: the SDP solver's duality gap is recorded and added
conservatively to each diamond norm, so ``L_j`` is a certified upper bound
up to solver feasibility tolerances; every downstream margin retains the
``margin_tol`` bracket machinery.

The per-interval Frechet derivative of ``expm(dt*G)`` in a direction
``E`` is evaluated by the block method,
``expm(dt*[[G, E], [0, G]])`` upper-right block, which needs no
diagonalisability assumption; the Hermitian eigenbasis formula of the
closed-system layer must not be used here, since Lindblad generators can
be defective.
"""

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
    """Lindblad dissipator superoperator ``D[V]`` at unit rate."""
    N = V.shape[0]
    Id = np.eye(N)
    VdV = V.conj().T @ V
    return np.kron(V.conj(), V) - 0.5 * (np.kron(Id, VdV) + np.kron(VdV.T, Id))


def generator(
    H: Array, Vs: Sequence[Array] = (), gammas: Sequence[float] = ()
) -> Array:
    """Lindblad generator, affine in ``H`` and the rates ``gammas``."""
    G = hamiltonian_superop(H)
    # strict: a rate list shorter than the jump operators silently
    # dropped a dissipator, and every margin read off the result was
    # then wrong with no error. All current callers pass matched
    # lists, so this cannot change a present result.
    for V, g in zip(Vs, gammas, strict=True):
        G = G + g * dissipator(V)
    return G


def channel(G_list: Sequence[Array], dt: float) -> Array:
    """Ordered product of interval propagators ``expm(dt*G^(k))``.

    Empty input is rejected by name, as the MATLAB peer already did; the
    unguarded version failed on ``G_list[0]`` with a bare IndexError that
    said nothing about which argument was wrong.
    """
    if len(G_list) == 0:
        raise ValueError("G_list must not be empty")
    N2 = G_list[0].shape[0]
    S = np.eye(N2, dtype=complex)
    for G in G_list:
        S = expm(dt * G) @ S
    return S


def unitary_superop(U: Array) -> Array:
    """Superoperator of the unitary channel ``rho -> U rho U^dag``."""
    return np.kron(U.conj(), U)


def process_fidelity(S: Array, Uf: Array) -> float:
    """``F_pro = Tr(S_f^dag S)/N^2`` with unitary target ``Uf``."""
    N = Uf.shape[0]
    Sf = unitary_superop(Uf)
    return float(np.real(np.trace(Sf.conj().T @ S)) / N**2)


def average_gate_fidelity(S: Array, Uf: Array) -> float:
    """``(N F_pro + 1)/(N + 1)``."""
    N = Uf.shape[0]
    return (N * process_fidelity(S, Uf) + 1.0) / (N + 1.0)


def frechet_derivative(G: Array, E: Array, dt: float) -> Array:
    """Exact ``d/ds expm(dt*(G + s E))`` at ``s = 0`` by the block method.

    Valid for arbitrary (possibly defective) generators; the upper-right
    block of ``expm(dt*[[G, E], [0, G]])`` is the Frechet derivative.
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

    Implemented as pure reindexing: every entry of ``J`` is a stored
    entry of ``S``, moved without arithmetic.  With column stacking,
    ``Phi(E_ij)[a, b] = S[a + bN, i + jN]``, and the definition places
    that value at ``J[aN + i, bN + j]``, so the whole map is one
    transpose of the four-index view.  The appendix's claim that the
    verified SDP bound applies to the *represented* superoperator rests
    on this being an exact rearrangement rather than a computation;
    :func:`superop_from_choi` inverts it and
    :func:`choi_roundtrip_exact` checks the round trip bit for bit.

    The earlier form built ``J`` by multiplying ``S`` against each
    matrix unit and accumulating, which is arithmetic on floating data
    and made the representation correspondence an assumption.
    """
    N = int(round(np.sqrt(S.shape[0])))
    S4 = np.asarray(S).reshape(N, N, N, N)
    return S4.transpose(1, 3, 0, 2).reshape(N * N, N * N).copy()


def superop_from_choi(J: Array, N: Optional[int] = None) -> Array:
    """Inverse of :func:`choi_matrix`, again pure reindexing."""
    if N is None:
        N = int(round(np.sqrt(J.shape[0])))
    J4 = np.asarray(J).reshape(N, N, N, N)
    return J4.transpose(2, 0, 3, 1).reshape(N * N, N * N).copy()


def choi_roundtrip_exact(S: Array) -> bool:
    """True when ``superop_from_choi(choi_matrix(S))`` reproduces every
    stored entry of ``S`` exactly (bit for bit, NaNs excluded)."""
    back = superop_from_choi(choi_matrix(S), int(round(np.sqrt(S.shape[0]))))
    return bool(np.array_equal(np.asarray(S), back))


@dataclass
class DiamondNorm:
    """Diamond norm value with a certified upper bound.

    ``value_certified`` is a VERIFIED upper bound on the diamond norm
    under the IEEE-754 rounding model: the solver's primal iterate is
    repaired to a STORED pair whose feasibility is proved by Rump's
    floating-Cholesky criterion (``_verified_psd_repair``,
    ``_verify_psd``), and the primal objective -- an upper bound for
    this minimisation -- is evaluated on exactly those stored blocks
    and bounded above by Gershgorin discs with explicit summation
    rounding inflation (``_partial_trace_specnorm_upper``,
    ``_sum_upward``).  Gershgorin makes the bound conservative on
    tight instances (up to a small factor for non-diagonal partial
    traces) but rigorous.  If no shift verifies,
    :class:`VerificationFailure` is raised and no value is returned:
    an unproved repair is not a certificate.
    Robustness constants must over-estimate the diamond norm, so use
    ``value_certified``; ``value`` equals it, ``raw`` is the solver's
    reported optimum, ``feas_shift`` the verified repair shift.

    ``status`` is ``"analytic"`` for the closed-form constants of
    :func:`common_rate_local_dnorm`, which need no SDP at all.

    ``solver`` names the solver that produced ``raw``, recorded because
    the value depends on it: leaving the choice to cvxpy made published
    numbers differ across environments with identical package versions.
    """

    value: float
    raw: float
    gap: float
    status: str
    value_certified: float = float("nan")
    feas_shift: float = float("nan")
    solver: str = "solver_free"


#: Backward-error constant of the floating Cholesky factorisation.
#:
#: Rump's Theorem 2.3 gives, for real symmetric data, an exact
#: factorisation of ``A + Delta`` with
#: ``|Delta_ij| <= c_n sqrt(A_ii A_jj)`` and
#: ``c_n = (n+1)u/(1 - (n+1)u)``.  LAPACK's complex Hermitian routine
#: (``zpotrf``) obeys the same shape with a larger constant, since a
#: complex multiply-add carries several real roundings; the factor
#: below covers that with room to spare.  It costs nothing: at
#: ``n = 128`` the whole constant is still about ``1e-13``.
_CHOL_COMPLEX_FACTOR = 8


def _up(x: float) -> float:
    """The next float above ``x``: one rounding-up step."""
    return float(np.nextafter(float(x), np.inf))


def _chol_error_bound(A: Array) -> Optional[float]:
    """Verified ``c >= ||Delta(A)||_2`` for the Cholesky backward error.

    Rump's Theorem 2.3 bounds the perturbation entrywise by a
    non-negative matrix of the form

        E_ij = alpha_ij d_i d_j + M eta,
        d_i  = sqrt(A_ii / (1 - alpha_ii)),

    with ``alpha_ij`` the operation-count coefficients of the
    factorisation and ``M eta`` an underflow allowance.  Neither the
    ``1/(1 - alpha_ii)`` inflation of the diagonal scaling nor the
    underflow term may be dropped from a rigorous claim, so this
    function bounds the whole of it rather than its leading factor.

    The simplification used is conservative and dominates the exact
    bound termwise.  Write ``alpha`` for an upper bound on every
    ``alpha_ij`` (below); then ``d_i <= sqrt(A_ii/(1 - alpha))`` and

        E_ij <= alpha/(1 - alpha) * sqrt(A_ii A_jj) + M eta.

    The first part is symmetric, non-negative and rank one, so its
    spectral norm is at most its maximum row sum; the constant part
    contributes at most ``n M eta``.  Every operation below is rounded
    upward.

    ``alpha`` is taken as ``k u/(1 - k u)`` with ``k = 8(n+1)``: Rump's
    real-arithmetic coefficients run to ``gamma_{n+1}``, and the factor
    eight covers the extra roundings of complex arithmetic in the
    Hermitian LAPACK path.  ``eta`` is taken as the smallest positive
    NORMAL double, itself an over-estimate of any underflow unit, and
    ``M`` as ``4(n+1)``; the whole underflow term is then below 1e-300
    for any dimension of interest, which is why it can be bounded so
    crudely without loss.

    Returns ``None`` when a diagonal entry is negative, in which case
    the matrix is not positive semidefinite and no test is needed.
    """
    n = A.shape[0]
    u = np.finfo(float).eps / 2.0
    k = _CHOL_COMPLEX_FACTOR * (n + 1)
    if k * u >= 0.5:
        return None
    alpha = _up(k * u / (1.0 - k * u))
    # The diagonal inflation 1/(1 - alpha_ii) of Theorem 2.3, folded
    # into the leading coefficient: alpha/(1 - alpha) >= alpha_ij /
    # sqrt((1 - alpha_ii)(1 - alpha_jj)) for every i, j.
    coef = _up(alpha / (1.0 - alpha))
    d = np.real(np.diag(A))
    if np.any(d < 0.0):
        return None
    v = np.nextafter(np.sqrt(d), np.inf)  # sqrt is correctly rounded
    # Upward-bounded sum: floating sum inflated by the standard bound.
    g = (n - 1) * u / (1.0 - (n - 1) * u) if n > 1 else 0.0
    tot = _up(float(np.sum(v)) * (1.0 + g))
    main = _up(_up(coef * float(np.max(v))) * tot)
    # Underflow allowance: n entries per row, M eta each.
    eta = np.finfo(float).tiny
    under = _up(_up(float(n) * float(4 * (n + 1))) * eta)
    return _up(main + under)


def _shift_diag_down(a: float, c: float) -> Optional[float]:
    """Largest stored double ``t`` verified to satisfy ``t <= a - c``.

    Round-to-nearest subtraction does not establish that inequality, so
    the candidate is stepped down until an exact rational comparison on
    the stored doubles confirms it.  Rump's Lemma 2.5 gives the
    equivalent floating construction; the exact check is used here
    because it is unarguable and costs one comparison per diagonal
    entry.  Returns ``None`` if no such float is found in a few steps
    (which happens only for non-finite input).
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
    """Verify ``A >= 0`` exactly, for the Hermitian floating matrix ``A``
    as it is stored, by Rump's floating-Cholesky criterion.

    A verified ``c >= ||Delta||_2`` is computed for the Cholesky
    backward error from the diagonal of ``A``, a test matrix
    ``A_test`` is formed with the off-diagonal entries of ``A``
    unchanged and diagonal entries verified to satisfy
    ``A_test_ii <= A_ii - c``, the pair ``(A_test, c)`` is then checked
    to satisfy the bound for ``A_test`` itself, and the floating
    Cholesky is run on it as a success/failure predicate.  On success
    the computed factor is exact for ``A_test + Delta`` with
    ``||Delta||_2 <= c``, so ``A_test >= -c I`` and therefore
    ``A = A_test + diag(A_ii - A_test_ii) >= A_test + c I >= 0``.

    The eigensolver plays no part in this: it only proposes shifts
    elsewhere.
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
    # The bound above was computed from A, not from A_test, which would
    # be circular: A_test is built from c. The bound is non-decreasing in
    # the diagonal entries and A_test_ii <= A_ii, so c dominates the
    # bound for A_test -- the route of Rump's Corollary 2.4. That is
    # checked here rather than argued, so the pair (A_test, c) is
    # verified as a pair after construction.
    c_test = _chol_error_bound(A_test)
    if c_test is None or not (c_test <= c):
        return False
    try:
        np.linalg.cholesky(A_test)
    except np.linalg.LinAlgError:
        return False
    return True


class VerificationFailure(RuntimeError):
    """No verified positive-semidefinite repair was obtained.

    Raised rather than returning a number: a certified diamond norm
    whose feasibility was not proved is not a certificate, and the
    callers of this module use the value as a robustness constant that
    must over-estimate.
    """


def _verified_psd_repair(
    Y0: Array,
    Y1: Array,
    J: Array,
    shift0: float,
    max_tries: int = 60,
) -> Tuple[Array, Array, float]:
    """Repair ``(Y0, Y1)`` to verified feasibility of the Watrous block.

    Returns the STORED repaired blocks and the shift that produced
    them, so the caller evaluates the objective on exactly the matrices
    whose feasibility was proved.  The block is reassembled from those
    stored blocks and verified as it stands; nothing assumes that
    adding a floating shift produced the mathematical matrix
    ``B + eps I``.

    ``shift0`` is only a proposal (typically from the floating
    eigensolver).  On failure the shift grows geometrically; if no
    shift verifies within ``max_tries``, :class:`VerificationFailure`
    is raised.
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
    """Upper bound on a sum of nonnegative floats: floating sum inflated
    by the standard (n-1)u/(1-(n-1)u) summation error bound."""
    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return 0.0
    u = np.finfo(float).eps / 2.0
    g = (n - 1) * u / (1.0 - (n - 1) * u) if n > 1 else 0.0
    return float(np.sum(v)) * (1.0 + g)


def _specnorm_upper(A: Array) -> float:
    """Rigorous upper bound on the spectral norm of Hermitian ``A`` via
    Gershgorin discs, inflated by the summation rounding bound."""
    n = A.shape[0]
    u = np.finfo(float).eps / 2.0
    rows = np.sum(np.abs(A), axis=1)
    return float(np.max(rows)) * (1.0 + (n + 1) * u)


def _partial_trace_specnorm_upper(Y: Array, N: int) -> float:
    """Rigorous upper bound on ``||Tr_out Y||_2`` for Hermitian ``Y`` on
    output (x) input, enclosing the partial-trace summation error.

    Each entry of ``M = Tr_out Y`` is a floating sum of ``N`` complex
    numbers, so ``|M_ij - fl(M_ij)| <= (N-1) u * A_ij`` with ``A`` the
    partial trace of the entrywise absolute values (a sum of
    nonnegative numbers, itself bounded upward by the same model).
    The Gershgorin row sums are then taken over
    ``|fl(M)| + (N-1)u * A`` with the usual row-sum inflation; complex
    magnitudes carry an extra ``(1 + 4u)`` cushion for the rounding of
    ``abs`` on complex data.
    """
    u = np.finfo(float).eps / 2.0
    Yr = Y.reshape(N, N, N, N)
    M = np.trace(Yr, axis1=0, axis2=2)
    A = np.trace(np.abs(Yr), axis1=0, axis2=2)
    ent = np.abs(M) * (1.0 + 4.0 * u) + (N - 1) * u * A * (1.0 + 4.0 * u)
    rows = np.sum(ent, axis=1)
    return float(np.max(rows)) * (1.0 + (N + 1) * u)


def diamond_norm_free(S: Array, iters: int = 600, degtol: float = 1e-6) -> DiamondNorm:
    """Diamond norm upper bound with no external solver.

    The Watrous program is a MINIMISATION, so every feasible point is an
    upper bound and no solver is needed to obtain one -- only to make it
    tight. That is what this exploits, and it is why the open-system layer
    needs no cvxpy, and on the MATLAB side no CVX or SDPT3, neither of
    which is portable to Octave.

    Start at a closed-form feasible point. With the polar factors
    ``|J^dag| = (J J^dag)^{1/2}`` and ``|J| = (J^dag J)^{1/2}``, the block

        [[ |J^dag|, -J ], [ -J^dag, |J| ]]

    is positive semidefinite -- write ``J = W|J|`` and it is a congruence
    of ``[[A,-A],[-A,A]] >= 0`` -- and remains so under the scaling
    ``(s|J^dag|, |J|/s)``, whose objective is minimised at
    ``s = sqrt(b/a)`` giving ``sqrt(a b)`` with ``a``, ``b`` the spectral
    norms of the two partial traces. That alone is a certified bound, and
    it is exact for the dephasing families of the case studies.

    Then improve it: scaled subgradient steps on the objective, with
    feasibility restored by projecting the constraint block onto the
    positive semidefinite cone (the off-diagonal blocks are fixed by
    ``J``, so this is alternating projection), and a backtracking step
    size. The iterate is finally repaired to exact feasibility by a
    shift, so the returned value is an upper bound whatever the
    iteration did.

    Measured against cvxpy on the shipped generators ``raw`` -- the
    floating objective at the repaired feasible point -- is exact to
    five decimals for dissipative and mixed maps, and up to about 3%
    conservative for a pure Hamiltonian superoperator, where the
    subgradient stalls on a degenerate spectrum. Conservative is the
    safe direction: a robustness constant must over-estimate.

    Two numbers come back, and they differ. ``raw`` is that tight
    floating objective. ``value`` (equal to ``value_certified``) is the
    same feasible point bounded UPWARD with the partial-trace,
    complex-magnitude, row-sum and final-combination roundings all
    enclosed, so it is rigorous under the arithmetic model; Gershgorin
    makes it noticeably looser on a non-diagonal partial trace. Use
    ``value_certified`` wherever the number is a certificate and
    ``raw`` only for tightness diagnostics.
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

    # Verified-feasibility repair, so the value is an upper bound
    # whatever the iteration converged to. The eigensolver proposes the
    # shift; the proof is Rump's criterion on the stored repaired
    # block, and the objective is evaluated on exactly those blocks.
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


#: Solver for the Watrous SDP, named rather than left to cvxpy.
#:
#: The program is a *complex* SDP, and cvxpy's automatic choice for that
#: class is SCS even though CLARABEL ranks above it in cvxpy's own conic
#: preference order; CLARABEL solves it through the complex-to-real
#: reduction. The difference is not cosmetic. On the generators this
#: package certifies, SCS returns a repair gap up to 1.8e-4 and warns that
#: the solution may be inaccurate, while CLARABEL stays below 3e-8 -- five
#: orders tighter on the same problem.
#:
#: It is named here because leaving it to the resolver made the published
#: numbers depend on which cvxpy happened to be installed: the same code,
#: same package versions by ``pip freeze``, produced open-system results
#: that differed by up to 4e-5 relative across a virtualenv rebuild. Pinning
#: dependency versions does not fix that on its own; naming the solver does.
#:
#: Soundness never depended on the choice -- the repair step makes the
#: returned value a rigorous upper bound whatever the solver reports -- but
#: reproducibility and tightness both did.
DEFAULT_SDP_SOLVER = "CLARABEL"


def diamond_norm(S: Array, solver: Optional[str] = None) -> DiamondNorm:
    """Diamond norm of the superoperator ``S`` by the Watrous SDP.

    Uses the semidefinite program of Watrous (2013): with ``J`` the Choi
    matrix on (output kron input),

        dnorm(S) = min 0.5*||Tr_out Y0||_inf + 0.5*||Tr_out Y1||_inf
                   s.t. [[Y0, -J], [-J^dag, Y1]] >= 0,

    valid for arbitrary linear maps.  Requires ``cvxpy``
    (``pip install qrobustness[open]``).

    ``solver`` defaults to ``DEFAULT_SDP_SOLVER``; pass a name to override
    it, or ``""`` to hand the choice back to cvxpy.
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
    # norm(., 2) on a Hermitian matrix variable is the spectral norm in cvxpy
    prob = cp.Problem(cp.Minimize(objective), constraints)
    chosen = DEFAULT_SDP_SOLVER if solver is None else solver
    kwargs = {"solver": chosen} if chosen else {}
    prob.solve(**kwargs)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"diamond norm SDP failed: {prob.status}")
    raw = float(prob.value)

    # Certified upper bound: repair the solver's primal iterate to exact
    # feasibility and evaluate the primal objective (an upper bound for a
    # minimisation) outside the solver.  If lambda_min of the constraint
    # block is -eps < 0, then (Y0 + eps I, Y1 + eps I) is feasible, since
    # the block gains eps I_{2d}.
    Y0v = np.asarray(Y0.value)
    Y1v = np.asarray(Y1.value)
    Y0v = (Y0v + Y0v.conj().T) / 2
    Y1v = (Y1v + Y1v.conj().T) / 2
    B = np.block([[Y0v, -J], [-J.conj().T, Y1v]])
    B = (B + B.conj().T) / 2
    lam_min = float(np.min(np.linalg.eigvalsh(B)))
    # Verified feasibility repair. The eigensolver only PROPOSES the
    # shift; feasibility is proved by Rump's floating-Cholesky criterion
    # applied to the block reassembled from the STORED repaired blocks,
    # so nothing assumes that adding a floating shift produced the
    # mathematical matrix B + eps I. Failure raises rather than
    # returning an unproved number.
    Y0r, Y1r, eps = _verified_psd_repair(Y0v, Y1v, J, max(0.0, -lam_min))

    # Rigorous upward-bounded primal objective, evaluated on exactly the
    # repaired blocks whose feasibility was verified: partial-trace
    # summation errors enclosed entrywise, spectral norms bounded above
    # by Gershgorin discs with rounding inflation, and the final
    # combination inflated once more.
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
    """Exact diamond norm ``2n`` of the common-rate local jump families.

    For ``V_q = sigma_z^(q)`` (local Pauli dephasing) and for
    ``V_q = sigma_-^(q)`` (local amplitude damping) on ``n`` qubits,

        || sum_q D[V_q] ||_diamond = 2n

    exactly, proved in the paper (upper bound by the triangle
    inequality on the three terms of the dissipator, lower bound by an
    explicit witness: GHZ for dephasing, the fully excited state for
    amplitude damping).  These two constants therefore need no SDP,
    and the drivers use this value rather than a solver optimum.  The
    SDP is retained only as an independent cross-check of the closed
    form (``run_dnorm_certificates.py``).

    ``2n`` is exactly representable in binary64 for every ``n`` of
    interest, so the returned value carries no rounding at all.
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
    """``T_phi = 1/(2 gamma_z)``, the PURE-dephasing time.

    This is not the total ``T_2`` unless amplitude damping is absent;
    use :func:`coherence_time` when both channels act.  Returning
    ``T_phi`` under its own name is the point: reporting it as ``T_2``
    in a simultaneous-noise model overstates the coherence time.
    """
    g = float(gamma_z)
    if g < 0.0:
        raise ValueError("gamma_z must be non-negative")
    return float("inf") if g == 0.0 else 1.0 / (2.0 * g)


def coherence_time(gamma_z: float, gamma_minus: float = 0.0) -> float:
    """Total ``T_2`` from both rates, ``1/T_2 = 1/T_phi + 1/(2 T_1)``.

    In the rates themselves, ``1/T_2 = 2 gamma_z + gamma_-/2``.  With
    ``gamma_- = 0`` this reduces to ``T_2 = T_phi = 1/(2 gamma_z)``,
    the dephasing-only special case; with both rates zero the time is
    infinite.
    """
    gz, gm = float(gamma_z), float(gamma_minus)
    if gz < 0.0 or gm < 0.0:
        raise ValueError("rates must be non-negative")
    inv = 2.0 * gz + 0.5 * gm
    return float("inf") if inv == 0.0 else 1.0 / inv


def rates_from_times(T1: float, T2: float) -> Tuple[float, float]:
    """``(gamma_z, gamma_-)`` from measured ``T_1`` and ``T_2``.

    Inverts :func:`coherence_time` and :func:`relaxation_time`:

        gamma_z = 1/(2 T_2) - 1/(4 T_1),   gamma_- = 1/T_1,

    on the physical domain ``T_2 <= 2 T_1``, outside which no
    non-negative dephasing rate reproduces the pair.  Infinite times
    are accepted and map to zero rates.  A two-rate budget
    ``L_z gamma_z + L_- gamma_- <= beta`` converts through this
    substitution as one JOINT constraint; separate one-dimensional
    intercepts are not equivalent to it.
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
    """Per-parameter constants ``L_j = 0.5 * sum_k dt * dnorm(G_j^(k))``.

    ``G_structs[j]`` is the list of per-interval structure generators for
    parameter ``j``; for time-independent structures pass a length-1 list,
    which is treated as constant over the gate (the sum collapses to
    ``0.5 * t_f * dnorm(G_j)`` when multiplied by ``tau`` by the caller).
    Returns the constants and the per-structure diamond-norm diagnostics.
    """
    L = []
    diags: List[List[DiamondNorm]] = []
    for structs in G_structs:
        norms = [diamond_norm(G, solver=solver) for G in structs]
        diags.append(norms)
        L.append(0.5 * dt * sum(n.value for n in norms))
    return np.asarray(L, dtype=float), diags


def rate_lipschitz(dnorm_value: float, t_f: float) -> float:
    """Lipschitz constant of the process fidelity in a common Lindblad rate.

    The time-independent case of :func:`open_structure_constants`: a
    structure generator applied at constant strength over the whole gate
    contributes ``0.5 * t_f * dnorm(G)``, the factor one half being the
    process fidelity's sensitivity to a channel deviation measured in the
    diamond norm. Four drivers computed this inline; the derivation lives
    with the general form above.

    Pass ``dn.value`` for the solved diamond norm or ``dn.value_certified``
    for the verified upper bound -- which of the two is appropriate is the
    caller's decision, not this helper's.
    """
    return 0.5 * t_f * dnorm_value


def make_open_fidelity_fn(
    build_generators: Callable[[Array], Sequence[Array]],
    dt: float,
    Uf: Array,
) -> Callable[[Array], float]:
    """Process fidelity as a function of the open-system parameter vector.

    ``build_generators(mu)`` returns the per-interval generators at ``mu``.
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
    """Certified margin for a scalar open-system parameter.

    Thin wrapper over :func:`qrobustness.iterative_margin` with the
    open-system Lipschitz constant and, by default, the one-sided domain
    ``[0, inf)`` appropriate for a decoherence rate; the lower boundary is
    reported through the usual ``omega``/boundary machinery.  All error
    control (``margin_tol`` brackets) passes through.
    """
    return iterative_margin(fidelity_fn, L, FT_pro, omega=omega, **kwargs)


def local_ops(op: Array, n_qubits: int) -> list:
    """``op`` acting on each qubit of an ``n_qubits`` register in turn.

    Returns one full-register operator per site, ``I x .. x op x .. x I``,
    in site order. Five drivers carried byte-identical copies of this, and
    a sixth caller reached one of them by inserting ``scripts/`` on
    ``sys.path`` and importing from a driver -- which is what a shared
    helper in the library is for.

    The Kronecker products are built left to right exactly as those copies
    built them, so the results are unchanged.
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
    """``sigma_z`` acting on each qubit of the register."""
    return local_ops(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex), n_qubits)

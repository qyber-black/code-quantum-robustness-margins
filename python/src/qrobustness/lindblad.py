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
    "DiamondNorm",
    "diamond_norm",
    "diamond_norm_free",
    "open_structure_constants",
    "make_open_fidelity_fn",
    "open_margin",
]


# --------------------------------------------------------------------------
# Builders (column-stacking convention)
# --------------------------------------------------------------------------


def hamiltonian_superop(H: Array) -> Array:
    """Superoperator of ``-1j*[H, .]``."""
    N = H.shape[0]
    I = np.eye(N)
    return -1j * (np.kron(I, H) - np.kron(H.T, I))


def dissipator(V: Array) -> Array:
    """Lindblad dissipator superoperator ``D[V]`` at unit rate."""
    N = V.shape[0]
    I = np.eye(N)
    VdV = V.conj().T @ V
    return np.kron(V.conj(), V) - 0.5 * (np.kron(I, VdV) + np.kron(VdV.T, I))


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
    """Ordered product of interval propagators ``expm(dt*G^(k))``."""
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
    """Choi matrix ``J = sum_ij Phi(E_ij) kron E_ij`` (output kron input)."""
    N = int(round(np.sqrt(S.shape[0])))
    J = np.zeros((N * N, N * N), dtype=complex)
    for i in range(N):
        for j in range(N):
            Eij = np.zeros((N, N), dtype=complex)
            Eij[i, j] = 1.0
            out = (S @ Eij.reshape(-1, order="F")).reshape((N, N), order="F")
            J += np.kron(out, Eij)
    return J


@dataclass
class DiamondNorm:
    """Diamond norm value with a certified upper bound.

    ``value_certified`` is a VERIFIED upper bound on the diamond norm
    under the IEEE-754 rounding model: the solver's primal iterate is
    repaired with a shift whose positive semidefiniteness is proved by
    the Rump floating-Cholesky criterion (``_verified_psd_shift``),
    and the primal objective -- an upper bound for this minimisation
    -- is bounded above by Gershgorin discs with explicit summation
    rounding inflation (``_specnorm_upper``, ``_sum_upward``).
    Gershgorin makes the bound conservative on tight instances (up to
    a small factor for non-diagonal partial traces) but rigorous.
    Robustness constants must over-estimate the diamond norm, so use
    ``value_certified``; ``value`` equals it, ``raw`` is the solver's
    reported optimum, ``feas_shift`` the verified repair shift.
    """

    value: float
    raw: float
    gap: float
    status: str
    value_certified: float = float("nan")
    feas_shift: float = float("nan")


def _verified_psd_shift(B: Array, shift0: float) -> float:
    """Smallest verified shift ``s`` (from a geometric search upward of
    ``shift0``) such that ``B + s I`` is provably positive semidefinite
    under the IEEE-754 rounding model.

    Verification uses the floating-Cholesky criterion of Rump: for a
    Hermitian floating-point matrix ``A`` of dimension ``n`` with unit
    roundoff ``u``, if the floating Cholesky factorisation of
    ``A - delta I`` succeeds with

        delta = c_n * norm_bound,  c_n = (n + 1) u / (1 - (n + 1) u),

    and ``norm_bound >= max_i A_ii`` (an upper bound on the spectral
    norm scale entering the standard backward-error bound), then ``A``
    is positive semidefinite exactly.  The candidate shift comes from
    the floating eigensolver, inflated by the same rounding constant;
    on failure the shift is grown geometrically.
    """
    n = B.shape[0]
    u = np.finfo(float).eps / 2.0
    c_n = (n + 1) * u / (1.0 - (n + 1) * u)
    # Gershgorin upper bound on ||B||_2: rigorous up to summation
    # rounding, inflated accordingly.
    row_sums = np.max(np.sum(np.abs(B), axis=1))
    norm_up = float(row_sums) * (1.0 + n * u)
    delta = c_n * norm_up
    s = max(shift0, 0.0) * (1.0 + 16 * u) + delta
    for _ in range(60):
        A = B + s * np.eye(n)
        try:
            np.linalg.cholesky(A - delta * np.eye(n))
            return float(s)
        except np.linalg.LinAlgError:
            s = 2.0 * s + delta + norm_up * u
    raise RuntimeError("verified PSD shift search failed")


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
    tight. That is what this exploits, and it is why the open-system
    layer no longer depends on cvxpy (or, on the MATLAB side, on CVX or
    SDPT3, neither of which is portable to Octave).

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

    Measured against cvxpy on the shipped generators the result is exact
    to five decimals for dissipative and mixed maps, and up to about 3%
    conservative for a pure Hamiltonian superoperator, where the
    subgradient stalls on a degenerate spectrum. Conservative is the
    safe direction: a robustness constant must over-estimate.
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

    # Exact-feasibility repair, so the value is an upper bound whatever
    # the iteration converged to.
    B = np.block([[Y0, -J], [-J.conj().T, Y1]])
    B = (B + B.conj().T) / 2
    lam_min = float(np.linalg.eigvalsh(B).min())
    shift = 0.0
    if lam_min < 0.0:
        shift = -lam_min * (1.0 + 1e-12) + 1e-15
        Y0 = Y0 + shift * np.eye(d)
        Y1 = Y1 + shift * np.eye(d)
    value = objective(Y0, Y1)
    return DiamondNorm(
        value=value,
        raw=value,
        gap=float(start - value),
        status="solver_free",
        value_certified=value,
        feas_shift=shift,
    )


def diamond_norm(S: Array, solver: Optional[str] = None) -> DiamondNorm:
    """Diamond norm of the superoperator ``S`` by the Watrous SDP.

    Uses the semidefinite program of Watrous (2013): with ``J`` the Choi
    matrix on (output kron input),

        dnorm(S) = min 0.5*||Tr_out Y0||_inf + 0.5*||Tr_out Y1||_inf
                   s.t. [[Y0, -J], [-J^dag, Y1]] >= 0,

    valid for arbitrary linear maps.  Requires ``cvxpy``
    (``pip install qrobustness[open]``).
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
    kwargs = {"solver": solver} if solver else {}
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
    # Verified feasibility repair: smallest shift with a Rump
    # floating-Cholesky proof of positive semidefiniteness (IEEE-754
    # rounding model), grown geometrically from the floating estimate.
    eps = _verified_psd_shift(B, max(0.0, -lam_min))
    eye = np.eye(d)

    # Rigorous upward-bounded primal objective: partial-trace summation
    # errors enclosed entrywise, spectral norms bounded above by
    # Gershgorin discs with rounding inflation, and the final
    # combination inflated once more.
    up0 = _partial_trace_specnorm_upper(Y0v + eps * eye, N)
    up1 = _partial_trace_specnorm_upper(Y1v + eps * eye, N)
    certified = 0.5 * _sum_upward([up0, up1])
    gap = certified - raw
    return DiamondNorm(
        value=certified,
        raw=raw,
        gap=gap,
        status=prob.status,
        value_certified=certified,
        feas_shift=eps,
    )


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
    ``export_golden_next`` reached one of them by inserting ``scripts/`` on
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

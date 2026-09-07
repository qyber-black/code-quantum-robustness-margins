% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function r = diamond_norm(S, varargin)
%DIAMOND_NORM Diamond norm upper bound, with no external SDP solver.
%
%   r = qrobustness.lindblad.diamond_norm(S) returns a struct with fields
%   value, raw, gap, status, value_certified and feas_shift, mirroring
%   the Python DiamondNorm record. value equals value_certified and is
%   the RIGOROUS upward-bounded number; raw is the tight floating
%   objective at the same verified feasible point.
%
%   Name-value options:
%     'iters'  (default 600)   maximum refinement steps
%     'degtol' (default 1e-6)  relative tolerance for a degenerate top
%                              eigenvalue, over which the subgradient is
%                              averaged
%
%   WHY THIS EXISTS. The Watrous program
%
%       dnorm(S) = min 0.5*||Tr_out Y0|| + 0.5*||Tr_out Y1||
%                  s.t. [[Y0, -J], [-J', Y1]] >= 0
%
%   is a MINIMISATION, so every feasible point is already an upper bound
%   and no solver is needed to get one -- only to make it tight. That is
%   what this exploits, and it is why the open-system layer needs neither
%   CVX (MATLAB only) nor SDPT3, neither of which ports cleanly to
%   Octave. A robustness constant must over-estimate the diamond norm,
%   so an upper bound is the quantity actually wanted.
%
%   METHOD. Start from a closed-form feasible point: with the polar
%   factors |J'| = (J J')^(1/2) and |J| = (J' J)^(1/2), the block
%   [[|J'|, -J], [-J', |J|]] is positive semidefinite (write J = W|J| and
%   it is a congruence of [[A,-A],[-A,A]] >= 0), and stays feasible under
%   the scaling (s|J'|, |J|/s), whose objective is smallest at
%   s = sqrt(b/a), giving sqrt(a*b). Then take scaled subgradient steps,
%   restoring feasibility by projecting the constraint block onto the PSD
%   cone (the off-diagonal blocks are pinned to J, so this is alternating
%   projection), with a backtracking step size. Finally repair the
%   iterate to VERIFIED feasibility with a shift -- proved by Rump's
%   floating-Cholesky criterion on the stored repaired block, not by the
%   eigensolver that proposed it -- and bound the objective upward on
%   exactly those stored blocks, so value_certified is an upper bound
%   whatever the iteration did. If no shift verifies, the function
%   errors instead of returning an unproved number.
%
%   ACCURACY. Against the Python cvxpy path on the shipped generators
%   this agrees to five decimals for dissipative and mixed maps and is
%   about 3% conservative for a pure Hamiltonian superoperator, where the
%   subgradient stalls on a degenerate spectrum. On the dephasing family
%   of the case studies it is exact, and it is in fact closer to the
%   analytic value 2n than the cvxpy result.
%
%   Peer of python/src/qrobustness/lindblad.py:diamond_norm_free.

    p = inputParser;
    addParameter(p, 'iters', 600);
    addParameter(p, 'degtol', 1e-6);
    parse(p, varargin{:});
    iters = p.Results.iters;
    degtol = p.Results.degtol;

    N = round(sqrt(size(S, 1)));
    J = qrobustness.lindblad.choi_matrix(S);
    d = N * N;

    Y0 = psd_sqrt(J * J');
    Y1 = psd_sqrt(J' * J);
    a = norm(tr_out(Y0, N), 2);
    b = norm(tr_out(Y1, N), 2);
    if a > 0
        sc = sqrt(b / a);
        Y0 = sc * Y0;
        Y1 = Y1 / sc;
    end
    start = objective(Y0, Y1, N);

    best = start;
    step = 0.1 * max(best, 1e-12);
    for it = 1:iters
        [n0, n1] = project(Y0 - step * 0.5 * subgrad(Y0, N, degtol), ...
                           Y1 - step * 0.5 * subgrad(Y1, N, degtol), J, d);
        val = objective(n0, n1, N);
        if val < best - 1e-15
            Y0 = n0; Y1 = n1; best = val;
        else
            step = step * 0.7;
            if step < 1e-13 * max(best, 1)
                break
            end
        end
    end

    % Verified-feasibility repair. The eigensolver only PROPOSES the
    % shift; feasibility is proved by Rump's floating-Cholesky criterion
    % applied to the block reassembled from the STORED repaired blocks,
    % and the certified objective is evaluated on exactly those blocks.
    % Nothing assumes that adding a floating shift produced the
    % mathematical matrix B + eps*I.
    Y0 = (Y0 + Y0') / 2;
    Y1 = (Y1 + Y1') / 2;
    B = [Y0, -J; -J', Y1];
    lam_min = min(eig((B + B') / 2));
    [Y0, Y1, shift] = verified_repair(Y0, Y1, J, max(0, -lam_min), d);
    value = objective(Y0, Y1, N);
    certified = 0.5 * sum_upward([pt_specnorm_upper(Y0, N), ...
                                  pt_specnorm_upper(Y1, N)]);

    % value/value_certified is the RIGOROUS number (upward-bounded on
    % the verified feasible point); raw is the tight floating objective
    % at that same point, for tightness diagnostics only.
    r = struct('value', certified, 'raw', value, ...
               'gap', certified - value, ...
               'status', 'solver_free', 'value_certified', certified, ...
               'feas_shift', shift);
end

function x = up(x)
%UP A float at or above the next one up: one rounding-up step.
    x = x + abs(x) * eps + realmin;
end

function c = chol_error_bound(A)
%CHOL_ERROR_BOUND Verified c >= ||Delta(A)||_2 for the Cholesky backward
%   error, or -1 when A has a negative diagonal entry and no test is
%   needed.
%
%   Rump's Theorem 2.3 bounds the perturbation entrywise by a
%   non-negative matrix E_ij = alpha_ij*d_i*d_j + M*eta with
%   d_i = sqrt(A_ii/(1 - alpha_ii)). Neither the diagonal inflation nor
%   the underflow term may be dropped from a rigorous claim, so the
%   whole of it is bounded here. With alpha an upper bound on every
%   alpha_ij, d_i <= sqrt(A_ii/(1 - alpha)) and
%
%       E_ij <= alpha/(1 - alpha)*sqrt(A_ii*A_jj) + M*eta,
%
%   whose first part is symmetric, non-negative and rank one (spectral
%   norm at most its maximum row sum) and whose constant part
%   contributes at most n*M*eta. Every operation is rounded upward.
%
%   alpha uses k = 8*(n+1): Rump's real coefficients run to
%   gamma_{n+1}, and the factor eight covers the extra roundings of
%   complex arithmetic in the Hermitian path. eta is the smallest
%   positive NORMAL double, over-estimating any underflow unit, and
%   M = 4*(n+1); the underflow term stays below 1e-300 at any dimension
%   of interest, which is why it can be bounded this crudely.
    n = size(A, 1);
    u = eps / 2;
    k = 8 * (n + 1);
    if k * u >= 0.5
        c = -1;
        return
    end
    alpha = up(k * u / (1 - k * u));
    coef = up(alpha / (1 - alpha));
    dg = real(diag(A));
    if any(dg < 0)
        c = -1;
        return
    end
    v = up(sqrt(dg));
    if n > 1
        g = (n - 1) * u / (1 - (n - 1) * u);
    else
        g = 0;
    end
    tot = up(sum(v) * (1 + g));
    main = up(up(coef * max(v)) * tot);
    eta = realmin;
    under = up(up(n * (4 * (n + 1))) * eta);
    c = up(main + under);
end

function [t, ok] = shift_diag_down(a, c)
%SHIFT_DIAG_DOWN Largest stored double t VERIFIED to satisfy t <= a - c.
%   Round-to-nearest subtraction does not establish that inequality, so
%   the candidate is stepped down until the check passes. The check
%   itself is exact: t stays within a factor two of a, where Sterbenz's
%   lemma makes a - t exact in binary64.
    t = a - c;
    ok = false;
    for it = 1:8
        if (a - t) >= c
            ok = true;
            return
        end
        t = t - abs(t) * eps - realmin;
    end
end

function ok = verify_psd(A)
%VERIFY_PSD Verify A >= 0 exactly for the Hermitian floating matrix A as
%   it is STORED, by Rump's floating-Cholesky criterion. Successful
%   factorisation of the test matrix A_test (off-diagonals unchanged,
%   A_test_ii <= A_ii - c) is exact for A_test + Delta with
%   ||Delta||_2 <= c, so A_test >= -c*I and hence
%   A = A_test + diag(A_ii - A_test_ii) >= A_test + c*I >= 0.
%   The eigensolver plays no part in this.
    ok = false;
    if ~all(isfinite(A(:)))
        return
    end
    c = chol_error_bound(A);
    if c < 0
        return
    end
    A_test = A;
    for i = 1:size(A, 1)
        [t, good] = shift_diag_down(real(A(i, i)), c);
        if ~good
            return
        end
        A_test(i, i) = t;
    end
    % c came from A, not from A_test, which would be circular: A_test is
    % built from c. The bound is non-decreasing in the diagonal and
    % A_test_ii <= A_ii, so c dominates the bound for A_test -- the route
    % of Rump's Corollary 2.4. Checked here rather than argued, so the
    % pair (A_test, c) is verified as a pair after construction.
    c_test = chol_error_bound(A_test);
    if c_test < 0 || ~(c_test <= c)
        return
    end
    [~, p] = chol(A_test);
    ok = (p == 0);
end

function [Y0r, Y1r, epsv] = verified_repair(Y0, Y1, J, shift0, d)
%VERIFIED_REPAIR Repair (Y0, Y1) to VERIFIED feasibility of the Watrous
%   block, returning the stored repaired blocks and the shift that
%   produced them. On failure it errors rather than returning an
%   unproved number: a diamond norm whose feasibility was not proved is
%   not a certificate.
    u = eps / 2;
    sc = max(max(abs(J(:))), 1);
    epsv = max(shift0, 0) * (1 + 16 * u);
    flr = sc * 8 * u;
    if epsv <= 0
        epsv = flr;
    end
    for it = 1:60
        Y0r = Y0 + epsv * eye(d);
        Y1r = Y1 + epsv * eye(d);
        B = [Y0r, -J; -J', Y1r];
        if verify_psd(B)
            return
        end
        epsv = 2 * epsv + flr;
    end
    error('qrobustness:verificationFailure', ...
          'no verified positive-semidefinite repair of the iterate was found');
end

function s = sum_upward(v)
%SUM_UPWARD Upper bound on a sum of non-negative floats.
    n = numel(v);
    if n == 0
        s = 0;
        return
    end
    u = eps / 2;
    if n > 1
        g = (n - 1) * u / (1 - (n - 1) * u);
    else
        g = 0;
    end
    s = sum(v) * (1 + g);
end

function s = pt_specnorm_upper(Y, N)
%PT_SPECNORM_UPPER Rigorous upper bound on ||Tr_out Y||_2 for Hermitian
%   Y on (output kron input), enclosing the partial-trace summation
%   error. Each entry of M = Tr_out Y is a floating sum of N complex
%   numbers, so |M_ij - fl(M_ij)| <= (N-1)*u*A_ij with A the partial
%   trace of the entrywise absolute values. Gershgorin row sums are
%   taken over |fl(M)| + (N-1)*u*A, with an extra (1+4u) cushion for the
%   rounding of complex magnitudes.
    u = eps / 2;
    M = tr_out(Y, N);
    A = tr_out(abs(Y), N);
    ent = abs(M) * (1 + 4 * u) + (N - 1) * u * A * (1 + 4 * u);
    s = max(sum(ent, 2)) * (1 + (N + 1) * u);
end

function A = psd_sqrt(A)
%   Principal square root of a Hermitian matrix, negative eigenvalues
%   clamped to zero: the input is PSD in exact arithmetic and only
%   rounding can make it otherwise.
    A = (A + A') / 2;
    [V, D] = eig(A);
    w = max(real(diag(D)), 0);
    A = V * diag(sqrt(w)) * V';
    A = (A + A') / 2;
end

function T = tr_out(Y, N)
%TR_OUT Partial trace over the output factor of (output kron input).
    T = zeros(N, N);
    for i = 1:N
        idx = (i - 1) * N + (1:N);
        T = T + Y(idx, idx);
    end
end

function v = objective(Y0, Y1, N)
%   The Watrous objective at a feasible point, which is already an
%   upper bound on the diamond norm.
    v = 0.5 * (norm(tr_out(Y0, N), 2) + norm(tr_out(Y1, N), 2));
end

function G = subgrad(Y, N, degtol)
%SUBGRAD Subgradient of 0.5*||Tr_out Y||_2 in Y, averaged over the top
%   eigenspace so a degenerate maximum does not pick an arbitrary vector.
    T = tr_out(Y, N);
    T = (T + T') / 2;
    [V, D] = eig(T);
    w = real(diag(D));
    [lam, i] = max(abs(w));
    sgn = sign(w(i));
    if sgn == 0
        sgn = 1;
    end
    sel = find(abs(abs(w) - lam) <= degtol * max(lam, 1));
    P = zeros(N, N);
    for k = 1:numel(sel)
        v = V(:, sel(k));
        P = P + v * v';
    end
    P = sgn * P / numel(sel);
    G = kron(eye(N), P);
end

function [Y0, Y1] = project(Y0, Y1, J, d)
%PROJECT Alternating projection onto the PSD cone with the off-diagonal
%   blocks pinned to -J.
    for r = 1:30
        B = [Y0, -J; -J', Y1];
        B = (B + B') / 2;
        [V, D] = eig(B);
        w = real(diag(D));
        if min(w) >= -1e-14
            break
        end
        B = V * diag(max(w, 0)) * V';
        B = (B + B') / 2;
        Y0 = B(1:d, 1:d);
        Y1 = B(d + 1:end, d + 1:end);
    end
end

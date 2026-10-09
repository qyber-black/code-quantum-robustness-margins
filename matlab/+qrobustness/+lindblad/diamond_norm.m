% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function r = diamond_norm(S, varargin)
%DIAMOND_NORM Verified diamond-norm upper bound without an SDP solver.
%   S - column-stacked superoperator
%   r - struct with fields
%         value, value_certified - verified upper bound (equal)
%         raw        - floating Watrous objective at the same feasible point
%         gap        - value - raw
%         status     - 'solver_free'
%         feas_shift - shift used to repair feasibility
%
%   Name-value options:
%     'iters'  - maximum refinement steps (default 600)
%     'degtol' - relative tolerance on a degenerate top eigenvalue (default 1e-6)
%
%   Verified: bounded by Rump's floating-Cholesky criterion; see the xQRM paper, verified diamond-norm appendix.
%   Errors if no shift verifies.
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

    % Repair to verified feasibility: the eigensolver proposes the shift, the floating-Cholesky check verifies the stored
    % repaired blocks, and the certified objective is evaluated on those blocks.
    Y0 = (Y0 + Y0') / 2;
    Y1 = (Y1 + Y1') / 2;
    B = [Y0, -J; -J', Y1];
    lam_min = min(eig((B + B') / 2));
    [Y0, Y1, shift] = verified_repair(Y0, Y1, J, max(0, -lam_min), d);
    value = objective(Y0, Y1, N);
    certified = 0.5 * sum_upward([pt_specnorm_upper(Y0, N), ...
                                  pt_specnorm_upper(Y1, N)]);

    % value/value_certified is the verified upper bound; raw is the floating objective at the same point.
    r = struct('value', certified, 'raw', value, ...
               'gap', certified - value, ...
               'status', 'solver_free', 'value_certified', certified, ...
               'feas_shift', shift);
end

function x = up(x)
%UP A float at or above the next one up: a single rounding-up step.
    x = x + abs(x) * eps + realmin;
end

function c = chol_error_bound(A)
%CHOL_ERROR_BOUND Verified c >= ||Delta(A)||_2 for the Cholesky backward error, or -1 if A has a negative diagonal entry.
%   Rump's entrywise bound, evaluated with upward rounding.
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
%SHIFT_DIAG_DOWN Largest stored double t verified to satisfy t <= a - c.
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
%VERIFY_PSD Verify A >= 0 for the stored Hermitian matrix A by Rump's floating-Cholesky criterion.
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
    % c is computed from A; since A_test_ii <= A_ii it also bounds A_test. Checked on the constructed pair.
    c_test = chol_error_bound(A_test);
    if c_test < 0 || ~(c_test <= c)
        return
    end
    [~, p] = chol(A_test);
    ok = (p == 0);
end

function [Y0r, Y1r, epsv] = verified_repair(Y0, Y1, J, shift0, d)
%VERIFIED_REPAIR Shift (Y0, Y1) to verified feasibility of the Watrous block; return the stored blocks and the shift.
%   Errors if no shift verifies.
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
%SUM_UPWARD Upper bound on a sum of floats that are non-negative.
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
%PT_SPECNORM_UPPER Upper bound on ||Tr_out Y||_2 for Hermitian Y, including partial-trace rounding (Gershgorin).
    u = eps / 2;
    M = tr_out(Y, N);
    A = tr_out(abs(Y), N);
    ent = abs(M) * (1 + 4 * u) + (N - 1) * u * A * (1 + 4 * u);
    s = max(sum(ent, 2)) * (1 + (N + 1) * u);
end

function A = psd_sqrt(A)
% Principal square root of a Hermitian matrix, negative eigenvalues clamped to zero.
    A = (A + A') / 2;
    [V, D] = eig(A);
    w = max(real(diag(D)), 0);
    A = V * diag(sqrt(w)) * V';
    A = (A + A') / 2;
end

function T = tr_out(Y, N)
%TR_OUT Partial trace over the output factor of output kron input.
    T = zeros(N, N);
    for i = 1:N
        idx = (i - 1) * N + (1:N);
        T = T + Y(idx, idx);
    end
end

function v = objective(Y0, Y1, N)
% Watrous objective at a feasible point, an upper bound on the diamond norm.
    v = 0.5 * (norm(tr_out(Y0, N), 2) + norm(tr_out(Y1, N), 2));
end

function G = subgrad(Y, N, degtol)
%SUBGRAD Subgradient of 0.5*||Tr_out Y||_2, averaged over the top eigenspace.
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
%PROJECT Alternating projection onto the PSD cone with off-diagonal blocks pinned to -J.
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

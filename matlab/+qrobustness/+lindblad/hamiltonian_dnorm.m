% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function r = hamiltonian_dnorm(S, rtol)
%HAMILTONIAN_DNORM Verified diamond norm lambda_max(B) - lambda_min(B) of S = -1i*[B, .].
%   S    - column-stacked superoperator
%   rtol - tolerance passed to hamiltonian_part (default 1e-12)
%   r    - struct as qrobustness.lindblad.diamond_norm, status 'analytic'
%
%   Verified: bounded by Rump's floating-Cholesky criterion; see the xQRM paper, verified diamond-norm appendix.
%   Errors if S is not a Hamiltonian superoperator or the bounds do not verify.
%
%   Peer of python/src/qrobustness/lindblad.py:hamiltonian_dnorm.

    if nargin < 2 || isempty(rtol)
        rtol = 1e-12;
    end
    B = qrobustness.lindblad.hamiltonian_part(S, rtol);
    if isempty(B)
        error('qrobustness:lindblad:notHamiltonian', 'S is not a Hamiltonian superoperator');
    end
    N = size(B, 1);
    u = eps / 2;
    w = sort(real(eig(B)));
    raw = w(end) - w(1);
    slack = 64 * N * u * max(max(abs(w)), realmin);
    verified = false;
    for it = 1:12
        hi = up(w(end) + slack);
        lo = -up(-(w(1) - slack));
        A_hi = -B;
        A_lo = B;
        ok = true;
        for i = 1:N
            bii = real(B(i, i));
            [t_hi, g1] = shift_diag_down(hi, bii);
            [t_lo, g2] = shift_diag_down(bii, lo);
            if ~(g1 && g2)
                ok = false;
                break
            end
            A_hi(i, i) = t_hi;
            A_lo(i, i) = t_lo;
        end
        if ok && verify_psd(A_hi) && verify_psd(A_lo)
            verified = true;
            break
        end
        slack = 4 * slack;
    end
    if ~verified
        error('qrobustness:lindblad:verification', 'spectral bounds of the Hamiltonian did not verify');
    end
    spread = up(hi - lo);
    % Only the N^2 diagonal entries B_aa - B_cc of S_B are rounded.
    diag_err = up(up(4 * u * max(abs(real(diag(B))))) * N);
    D = S - qrobustness.lindblad.hamiltonian_superop(B);
    sq = up(up(real(D).^2) + up(imag(D).^2));
    fro = up(sqrt(sum_upward(sq(:))));
    fro = up(up(fro * up(1 + 2 * u)) + diag_err);
    v = up(spread + up(N * fro));
    r = struct('value', v, 'raw', raw, 'gap', max(v - raw, 0), 'status', 'analytic', ...
               'value_certified', v, 'feas_shift', 0);
end

% The helpers below duplicate those in diamond_norm.m (Octave does not resolve private/ inside a package); keep them identical.

function x = up(x)
%UP A float at or above the next one up: a single rounding-up step.
    x = x + abs(x) * eps + realmin;
end

function x = down(x)
    % A double at most x: one rounding-down step.
    x = x - abs(x) * eps - realmin;
end

function c = chol_error_bound(A)
    % Verified bound c >= ||Delta||_2 on the Cholesky backward error of A
    % (Rump, BIT 46 (2006), Theorem 2.3, with alpha = gamma_{8(n+1)}); -1 if
    % a diagonal entry is negative or n is too large.
    n = size(A, 1);
    u = eps / 2;
    k = 8 * (n + 1);
    if k * u >= 0.5
        c = -1;
        return
    end
    alpha = up(k * u / (1 - k * u));
    coef = up(alpha / down(1 - alpha));
    dg = real(diag(A));
    if any(dg < 0)
        c = -1;
        return
    end
    v = up(sqrt(dg));
    tot = sum_upward(v);
    main = up(up(coef * max(v)) * tot);
    % Rump's underflow term: n M eta with M = 3(2n + max a_ii), eta = 2^-1074.
    M = up(3 * up(2 * n + max(dg)));
    under = up(up(n * M) * 2^-1074);
    c = up(main + under);
end

function [t, ok] = shift_diag_down(a, c)
    % A double t with t <= a - c, checked exactly; ok false if none is found.
    t = a - c;
    ok = false;
    if ~isfinite(a) || ~isfinite(c)
        return
    end
    for it = 1:8
        if exact_ge(a, t, c)
            ok = true;
            return
        end
        t = t - abs(t) * eps - realmin;
    end
end

function ok = exact_ge(a, t, c)
    % True only if a - t >= c holds exactly (error-free transformations).
    [s, e1] = two_sum(a, -t);
    [d, e2] = two_sum(s, -c);
    ok = isfinite(s) && isfinite(d) && ...
        (d > 2 * (abs(e1) + abs(e2)) || (d >= 0 && e1 == 0 && e2 == 0));
end

function [s, e] = two_sum(a, b)
    % s = fl(a + b) and the exact error e = (a + b) - s.
    s = a + b;
    bb = s - a;
    e = (a - (s - bb)) + (b - bb);
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

function s = sum_upward(v)
    % Upper bound on the exact sum of non-negative v, in any summation order.
    v = v(:);
    m = numel(v);
    if m == 0
        s = 0;
        return
    end
    s = up(sum(v));
    if m > 1
        s = up(s * up(1 + 2 * gam(m - 1)));
    end
end

function g = gam(m)
    % Upper bound on gamma_m = m u / (1 - m u); requires m u <= 0.01.
    u = eps / 2;
    if m * u > 0.01
        error('qrobustness:verificationFailure', 'summation too long for the error bound');
    end
    g = up(1.02 * m * u);
end

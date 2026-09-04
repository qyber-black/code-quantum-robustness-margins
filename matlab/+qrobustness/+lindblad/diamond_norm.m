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
%   the Python DiamondNorm record.
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
%   iterate to exact feasibility with a shift, so the value returned is
%   an upper bound whatever the iteration did.
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
                break;
            end
        end
    end

    % Exact-feasibility repair, so the value is an upper bound whatever
    % the iteration converged to.
    B = [Y0, -J; -J', Y1];
    B = (B + B') / 2;
    lam_min = min(eig(B));
    shift = 0;
    if lam_min < 0
        shift = -lam_min * (1 + 1e-12) + 1e-15;
        Y0 = Y0 + shift * eye(d);
        Y1 = Y1 + shift * eye(d);
    end
    value = objective(Y0, Y1, N);

    r = struct('value', value, 'raw', value, 'gap', start - value, ...
               'status', 'solver_free', 'value_certified', value, ...
               'feas_shift', shift);
end

function A = psd_sqrt(A)
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
            break;
        end
        B = V * diag(max(w, 0)) * V';
        B = (B + B') / 2;
        Y0 = B(1:d, 1:d);
        Y1 = B(d+1:end, d+1:end);
    end
end

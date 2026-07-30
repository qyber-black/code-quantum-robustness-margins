function rates = uncertainty_rates(H_list, dH_list, dt, n_quad, n_dev, dev_tol, n_dev_max, adaptive_dev, dev_samples_per_cycle)
%UNCERTAINTY_RATES Uncertainty measures of their Eq. 28, per unit delta.
%   rates = UNCERTAINTY_RATES(H_list, dH_list, dt) returns a struct with
%   fields w_unc, w_avg, w_dev (their Omega_unc, Omega_avg, Omega_avg^dev at
%   delta = 1) and T (the gate duration tau*dt), plus the error-control
%   fields listed below.
%
%   H_list      : cell array of nominal per-interval Hamiltonians H^(k)
%                 (e.g. qrobustness.perturbed_hamiltonians(..., 0))
%   dH_list     : cell array of per-interval structures Hhat^(k)
%                 (e.g. qrobustness.dH_structure(...)), H_unc^(k)=delta*Hhat^(k)
%   n_quad      : accepted and unused; <Htil> is evaluated in closed form so
%                 there is no quadrature to tune
%   n_dev       : MINIMUM grid points per interval for the Omega_avg^dev
%                 supremum; the actual seed grid is derived from the
%                 interval bandwidth (see below)               (default 17)
%   dev_tol     : relative tolerance for adaptive refinement   (default 1e-9)
%   n_dev_max   : cap on the refined grid size per interval    (default 4097)
%   adaptive_dev: false samples once without refining
%   dev_samples_per_cycle : samples per Bohr cycle to seed with (default 16)
%
%   Within interval k the nominal propagator is
%   U_S(t_{k-1}+s) = expm(-1i*H^(k)*s) * P_{k-1}, so
%   Htil = P_{k-1}' * expm(1i*H^(k)*s) * Hhat^(k) * expm(-1i*H^(k)*s) * P_{k-1}.
%   Each H^(k) is diagonalised once and the conjugations are formed from its
%   eigenbasis, avoiding repeated matrix exponentials.
%
%   ERROR CONTROL
%     w_unc is exact: H_unc is piecewise constant, so the supremum over t is a
%     maximum over intervals.
%
%     w_avg is exact to roundoff: the time average is a closed-form divided
%     difference (see TIME_AVERAGE_HTIL below), not a quadrature.
%
%     w_dev is a supremum of a smooth function of s and is the only
%     approximated quantity.  Sampling can only under-estimate a supremum, and
%     a smaller w_dev yields a larger margin, so the residual error is biased
%     in the optimistic direction.  Htil(k,.) is a trigonometric polynomial
%     whose frequencies are the Bohr frequencies lam_m - lam_n, so its
%     bandwidth is known in closed form and the sampling density is derived
%     from it; candidate maxima are then polished with fminbnd and the grid
%     refined until two successive sweeps agree to dev_tol.
%
%     Returned diagnostics:
%       w_dev_certified   rigorous upper bound from d(Htil)/ds = 1i*[H,Htil]
%       w_dev_bracket_lo, w_dev_bracket_hi
%                         rigorous bracket from isospectrality of unitary
%                         conjugation, ||Htil(t)|| = ||Hhat^(k)||
%       w_dev_refinement  relative gain of the polish over plain sampling
%       n_dev_used        grid size at which refinement stopped
%       dev_converged     whether successive sweeps agreed to dev_tol
%       dev_cycles_max    Bohr cycles completed over one interval
%       dev_samples_per_cycle
%                         samples per cycle achieved
%       dev_resolved      whether the requested samples-per-cycle was met on
%                         every interval; false means n_dev_max capped the grid
%                         before the bandwidth was resolved, and the supremum
%                         may then be under-estimated
%
%     docs/time-bandwidth-bound.md gives the derivation and the measured
%     accuracy.
%
%   See also QROBUSTNESS.KOSUT.MARGIN, QROBUSTNESS.KOSUT.TIME_BANDWIDTH.

    if nargin < 4; n_quad = []; end  %#ok<NASGU> % accepted, unused
    if nargin < 5 || isempty(n_dev);        n_dev = 17;         end
    if nargin < 6 || isempty(dev_tol);      dev_tol = 1e-9;     end
    if nargin < 7 || isempty(n_dev_max);    n_dev_max = 4097;   end
    if nargin < 8 || isempty(adaptive_dev); adaptive_dev = true; end
    if nargin < 9 || isempty(dev_samples_per_cycle); dev_samples_per_cycle = 16; end

    tau = numel(H_list);
    if tau == 0
        error('qrobustness:kosut:EmptyH', 'H_list must be non-empty.');
    end
    if numel(dH_list) ~= tau
        error('qrobustness:kosut:LenMismatch', ...
            'H_list and dH_list must have equal length.');
    end
    if ~(dt > 0)
        error('qrobustness:kosut:BadDt', 'dt must be positive.');
    end

    N = size(H_list{1}, 1);
    T = tau * dt;

    % Per-interval eigendecomposition and left-accumulated propagator P_{k-1}.
    Vs = cell(1, tau);
    lams = cell(1, tau);
    Pref = cell(1, tau + 1);
    Pref{1} = eye(N);
    for k = 1:tau
        Hk = (H_list{k} + H_list{k}') / 2;   % enforce Hermitian symmetry
        [V, D] = eig(Hk);
        lam = real(diag(D));
        Vs{k} = V;
        lams{k} = lam;
        Useg = V * diag(exp(-1i * dt * lam)) * V';
        Pref{k + 1} = Useg * Pref{k};
    end

    % Omega_unc: H_unc is piecewise constant, so the sup is over intervals.
    w_unc = 0;
    for k = 1:tau
        w_unc = max(w_unc, norm(dH_list{k}, 2));
    end

    % <Htil> = (1/T) sum_k int_0^dt Htil(k,s) ds, in closed form per interval.
    acc = zeros(N);
    for k = 1:tau
        M = time_average_htil(Vs{k}, lams{k}, dH_list{k}, dt);
        acc = acc + Pref{k}' * M * Pref{k};
    end
    Havg = acc / T;
    w_avg = norm(Havg, 2);

    % Omega_avg^dev = sup_t ||Htil(t) - <Htil>||: locate candidate maxima on a
    % coarse grid, then polish each with fminbnd.  f is smooth in s, so the
    % polished value is accurate to ~eps rather than to the grid spacing.
    % Htil(k,s) = P' * expm(1i*H*s) * Hhat * expm(-1i*H*s) * P is a
    % trigonometric polynomial in s whose frequencies are exactly the Bohr
    % frequencies lam_m - lam_n of H^(k).  Its bandwidth is therefore known in
    % closed form, and the sampling density needed to resolve it can be DERIVED
    % rather than assumed: over an interval of length dt the fastest component
    % completes cycles_k = range(lam_k)*dt/(2*pi) cycles.  Seeding each interval
    % with dev_samples_per_cycle samples per cycle ensures the supremum search
    % remains resolved for controllers with long intervals or wide spectra, for
    % which a fixed grid would under-resolve without indication.
    cycles = zeros(1, tau);
    n_seed = zeros(1, tau);
    achieved = inf(1, tau);
    for k = 1:tau
        cycles(k) = (max(lams{k}) - min(lams{k})) * dt / (2 * pi);
        n_seed(k) = min(max([round(n_dev), 3, ceil(dev_samples_per_cycle * cycles(k)) + 1]), n_dev_max);
        if cycles(k) > 0
            achieved(k) = (n_seed(k) - 1) / cycles(k);
        end
    end
    min_spc = min(achieved);
    dev_resolved = (min_spc >= dev_samples_per_cycle);

    [w_dev_sampled, lipschitz_gap] = sweep(1, false, Vs, lams, dH_list, Pref, H_list, Havg, dt, tau, n_seed, n_dev_max);

    if ~adaptive_dev
        w_dev = w_dev_sampled;
        refinement = 0;
        dev_converged = false;
        scale = 1;
    else
        scale = 1;
        w_dev = sweep(scale, true, Vs, lams, dH_list, Pref, H_list, Havg, dt, tau, n_seed, n_dev_max);
        dev_converged = false;
        while max(n_seed) * scale < n_dev_max
            scale = scale * 2;
            w_next = sweep(scale, true, Vs, lams, dH_list, Pref, H_list, Havg, dt, tau, n_seed, n_dev_max);
            change = abs(w_next - w_dev) / max(w_next, 1e-300);
            w_dev = max(w_dev, w_next);
            if change <= dev_tol
                dev_converged = true;
                break;
            end
        end
        if w_dev > 0
            refinement = (w_dev - w_dev_sampled) / w_dev;
        else
            refinement = 0;
        end
    end
    n_used = min(scale * (max(n_seed) - 1) + 1, n_dev_max);

    % Independent rigorous bracket from isospectrality of unitary conjugation:
    % ||Htil(t)|| = ||Hhat^(k)|| exactly, so the deviation norm is bracketed.
    bracket_lo = 0;
    for k = 1:tau
        bracket_lo = max(bracket_lo, norm(dH_list{k}, 2) - w_avg);
    end
    bracket_lo = max(0, bracket_lo);
    bracket_hi = w_unc + w_avg;

    rates = struct('w_unc', w_unc, 'w_avg', w_avg, 'w_dev', w_dev, 'T', T, ...
        'w_dev_certified', w_dev_sampled + lipschitz_gap, ...
        'w_dev_refinement', refinement, ...
        'w_dev_bracket_lo', bracket_lo, ...
        'w_dev_bracket_hi', bracket_hi, ...
        'n_dev_used', n_used, ...
        'dev_converged', dev_converged, ...
        'dev_cycles_max', max(cycles), ...
        'dev_samples_per_cycle', min_spc * scale, ...
        'dev_resolved', dev_resolved);
end

function [best, gap] = sweep(scale, polish, Vs, lams, dH_list, Pref, H_list, Havg, dt, tau, n_seed, n_dev_max)
%SWEEP Best of grid (and optional Brent polish), plus the Lipschitz shortfall.
%   scale multiplies each interval's bandwidth-derived seed grid.
    best = 0;
    gap = 0;
    for k = 1:tau
        n_grid = min(scale * (n_seed(k) - 1) + 1, n_dev_max);
        grid = linspace(0, dt, n_grid);
        vals = zeros(1, n_grid);
        for j = 1:n_grid
            vals(j) = fdev(grid(j), Vs{k}, lams{k}, dH_list{k}, Pref{k}, Havg);
        end
        local_best = max(vals);

        % Rigorous shortfall of a sampled maximum: d(Htil)/ds = i[H,Htil], so
        % ||d(Htil)/ds|| <= 2||H|| ||Hhat||, and f is Lipschitz in s with that
        % constant.  On spacing h a sampled max falls short by <= L*h/2.
        L_s = 2 * norm(H_list{k}, 2) * norm(dH_list{k}, 2);
        gap = max(gap, 0.5 * L_s * dt / (n_grid - 1));

        best_k = local_best;
        if polish
            opts = optimset('TolX', 1e-15);
            for i = 1:n_grid
                interior = i > 1 && i < n_grid && vals(i) >= vals(i-1) && vals(i) >= vals(i+1);
                if ~(interior || vals(i) >= local_best)
                    continue;
                end
                a = grid(max(i-1, 1));
                b = grid(min(i+1, n_grid));
                if b <= a
                    continue;
                end
                g = @(s) -fdev(s, Vs{k}, lams{k}, dH_list{k}, Pref{k}, Havg);
                [~, fval] = fminbnd(g, a, b, opts);
                best_k = max(best_k, -fval);
            end
        end
        best = max(best, best_k);
    end
end

function v = fdev(s, V, lam, dH, P, Havg)
    v = norm(htil(V, lam, dH, P, s) - Havg, 2);
end

function M = time_average_htil(V, lam, dH, dt)
%TIME_AVERAGE_HTIL Exact int_0^dt expm(+1i*H*s) dH expm(-1i*H*s) ds.
%   H is constant on the interval, so in its eigenbasis the integral is a
%   divided difference: the (m,n) entry picks up
%   int_0^dt exp(1i*s*(lam_m-lam_n)) ds = dt*exp(1i*Y)*sin(Y)/Y with
%   Y = dt*(lam_m-lam_n)/2.  The exponent is purely imaginary, so there is no
%   cancellation and no threshold to tune; only Y == 0 is masked.
    lam = lam(:);
    Y = 0.5 * dt * (lam - lam.');
    S = ones(size(Y));
    nz = (Y ~= 0);
    S(nz) = sin(Y(nz)) ./ Y(nz);
    W = dt * exp(1i * Y) .* S;
    M = V * ((V' * dH * V) .* W) * V';
end

function M = htil(V, lam, dH, P, s)
%HTIL Interaction-picture uncertainty at t_{k-1}+s, per unit delta.
    E = V * diag(exp(1i * s * lam)) * V';   % expm(+1i*H^(k)*s)
    M = P' * (E * dH * E') * P;
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

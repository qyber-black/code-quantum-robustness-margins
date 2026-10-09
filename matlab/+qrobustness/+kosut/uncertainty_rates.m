function rates = uncertainty_rates(H_list, dH_list, dt, n_quad, n_dev, dev_tol, n_dev_max, adaptive_dev, dev_samples_per_cycle)
%UNCERTAINTY_RATES Per-unit uncertainty rates of the Kosut-Lidar-Rabitz bound.
%   rates = qrobustness.kosut.uncertainty_rates(H_list, dH_list, dt, n_quad, n_dev,
%           dev_tol, n_dev_max, adaptive_dev, dev_samples_per_cycle)
%
%   H_list                - nominal interval Hamiltonians H^(k)
%   dH_list               - structures \hat H^(k), H_unc^(k) = delta \hat H^(k)
%   dt                    - interval length
%   n_quad                - accepted and unused
%   n_dev                 - minimum grid points per interval for the w_dev
%                           supremum (default 17)
%   dev_tol               - relative tolerance of grid refinement (default 1e-9)
%   n_dev_max             - maximum grid points per interval (default 4097)
%   adaptive_dev          - refine the grid (default true)
%   dev_samples_per_cycle - seed samples per Bohr cycle (default 16)
%
%   rates fields:
%     w_unc, w_avg, w_dev   - Omega_unc, Omega_avg, Omega_avg^dev at delta = 1
%     T                     - gate time tau*dt
%     w_avg_traj, w_dev_traj - trajectory rates
%     w_dev_certified       - rigorous upper bound on w_dev
%     w_dev_bracket_lo, w_dev_bracket_hi - rigorous bracket on w_dev
%     w_dev_refinement      - relative gain of polishing over plain sampling
%     n_dev_used            - grid size at which refinement stopped
%     dev_converged         - successive sweeps agreed to dev_tol
%     dev_cycles_max        - most Bohr cycles in one interval
%     dev_samples_per_cycle - samples per cycle achieved
%     dev_resolved          - false when n_dev_max caps the grid that was
%                             actually used; w_dev may then be under-estimated
%
%   w_dev is sampled, so it can only under-estimate the supremum.
%
%   Peer of python/src/qrobustness/kosut.py:uncertainty_rates.

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

    % Eigendecomposition on each interval and the left-accumulated propagator P_{k-1}.
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

    % Omega_unc: H_unc is piecewise constant, so the supremum is taken over intervals.
    w_unc = 0;
    for k = 1:tau
        w_unc = max(w_unc, norm(dH_list{k}, 2));
    end

    % <Htil> = (1/T) sum_k int_0^dt Htil(k,s) ds, in closed form on each interval.
    acc = zeros(N);
    for k = 1:tau
        M = time_average_htil(Vs{k}, lams{k}, dH_list{k}, dt);
        acc = acc + Pref{k}' * M * Pref{k};
    end
    Havg = acc / T;
    w_avg = norm(Havg, 2);

    % Omega_avg^dev = sup_t ||Htil(t) - <Htil>||: candidate maxima on a grid, polished with fminbnd. The grid is seeded with
    % dev_samples_per_cycle samples per Bohr cycle, cycles_k = range(lam_k)*dt/(2*pi).
    cycles = zeros(1, tau);
    n_seed = zeros(1, tau);
    for k = 1:tau
        cycles(k) = (max(lams{k}) - min(lams{k})) * dt / (2 * pi);
        % fix, not round: the Python peer applies int(n_dev), which truncates.
        n_seed(k) = min(max([fix(n_dev), 3, ceil(dev_samples_per_cycle * cycles(k)) + 1]), n_dev_max);
    end
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
                break
            end
        end
        if w_dev > 0
            refinement = (w_dev - w_dev_sampled) / w_dev;
        else
            refinement = 0;
        end
    end
    n_used = min(scale * (max(n_seed) - 1) + 1, n_dev_max);
    % Samples per cycle on the grid actually used, after n_dev_max.
    samples_used = inf;
    for k = 1:tau
        n_final = min(scale * (n_seed(k) - 1) + 1, n_dev_max);
        if cycles(k) > 0
            samples_used = min(samples_used, (n_final - 1) / cycles(k));
        end
    end
    dev_resolved = (samples_used >= dev_samples_per_cycle);

    % Independent rigorous bracket from isospectrality of unitary conjugation: ||Htil(t)|| = ||Hhat^(k)|| exactly, so the deviation
    % norm remains bracketed.
    bracket_lo = 0;
    for k = 1:tau
        bracket_lo = max(bracket_lo, norm(dH_list{k}, 2) - w_avg);
    end
    bracket_lo = max(0, bracket_lo);
    bracket_hi = w_unc + w_avg;

    % Trajectory rates for |delta(t)| <= 1: w_avg_traj = mean_k ||Hhat^(k)||, w_dev_traj = w_unc + w_avg_traj.
    norms_dH = zeros(1, tau);
    for k = 1:tau
        norms_dH(k) = norm(dH_list{k}, 2);
    end
    w_avg_traj = mean(norms_dH);

    rates = struct('w_unc', w_unc, 'w_avg', w_avg, 'w_dev', w_dev, 'T', T, ...
        'w_avg_traj', w_avg_traj, 'w_dev_traj', w_unc + w_avg_traj, ...
        'w_dev_certified', w_dev_sampled + lipschitz_gap, ...
        'w_dev_refinement', refinement, ...
        'w_dev_bracket_lo', bracket_lo, ...
        'w_dev_bracket_hi', bracket_hi, ...
        'n_dev_used', n_used, ...
        'dev_converged', dev_converged, ...
        'dev_cycles_max', max(cycles), ...
        'dev_samples_per_cycle', samples_used, ...
        'dev_resolved', dev_resolved);
end

function [best, gap] = sweep(scale, polish, Vs, lams, dH_list, Pref, H_list, Havg, dt, tau, n_seed, n_dev_max)
%SWEEP Grid maximum of the deviation (optionally polished) and its Lipschitz shortfall gap.
%   scale multiplies each interval's seed grid.
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

        % Rigorous shortfall of a sampled maximum: d(Htil)/ds = i[H,Htil], hence
        % ||d(Htil)/ds|| <= 2||H|| ||Hhat||, and f is Lipschitz in s with that
        % constant. On spacing h a sampled max falls short by no more than L*h/2.
        L_s = 2 * norm(H_list{k}, 2) * norm(dH_list{k}, 2);
        gap = max(gap, 0.5 * L_s * dt / (n_grid - 1));

        best_k = local_best;
        if polish
            opts = optimset('TolX', 1e-15);
            for i = 1:n_grid
                interior = i > 1 && i < n_grid && vals(i) >= vals(i - 1) && vals(i) >= vals(i + 1);
                if ~(interior || vals(i) >= local_best)
                    continue
                end
                a = grid(max(i - 1, 1));
                b = grid(min(i + 1, n_grid));
                if b <= a
                    continue
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
% Deviation of Htil from its time average at interval time s. This is the objective whose supremum over s is Omega_avg^dev.
    v = norm(htil(V, lam, dH, P, s) - Havg, 2);
end

function M = time_average_htil(V, lam, dH, dt)
%TIME_AVERAGE_HTIL int_0^dt expm(1i*H*s) dH expm(-1i*H*s) ds, as a divided difference in the eigenbasis of H.
    lam = lam(:);
    Y = 0.5 * dt * (lam - lam.');
    S = ones(size(Y));
    nz = (Y ~= 0);
    S(nz) = sin(Y(nz)) ./ Y(nz);
    W = dt * exp(1i * Y) .* S;
    M = V * ((V' * dH * V) .* W) * V';
end

function M = htil(V, lam, dH, P, s)
%HTIL Interaction-picture uncertainty at t_{k-1}+s, per unit of delta.
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

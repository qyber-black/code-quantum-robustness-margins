function test_kosut_bound()
%TEST_KOSUT_BOUND Unit tests for +qrobustness/+kosut (peer of test_kosut.py).

    this_dir = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(this_dir));
    addpath(fullfile(root, 'matlab'));

    tol = 1e-10;
    SX = [0 1; 1 0];
    SY = [0 -1i; 1i 0];
    SZ = [1 0; 0 -1];

    % --- F_lb monotone, correct endpoints ------------------------------------
    ymax = qrobustness.kosut.t_omega_max();
    assert(abs(ymax - 1.8776) < 1e-4, 'T_OMEGA_MAX value');
    assert(abs(qrobustness.kosut.fidelity_bound(0) - 1) < tol, 'F_lb(0) = 1');
    ys = linspace(0, ymax, 50);
    Fs = arrayfun(@(y) qrobustness.kosut.fidelity_bound(y), ys);
    assert(all(diff(Fs) <= 1e-14), 'F_lb must be non-increasing');
    assert(qrobustness.kosut.fidelity_bound(ymax) == 0, 'F_lb vacuous at ymax');
    assert(qrobustness.kosut.fidelity_bound(2 * ymax) == 0, 'F_lb vacuous beyond');

    % --- closed-form threshold inversion is exact ----------------------------
    for FT = [0.9 0.99 0.999 0.9999]
        y = qrobustness.kosut.threshold_time_bandwidth(FT);
        assert(abs(qrobustness.kosut.fidelity_bound(y) - FT) < 1e-12, ...
            'threshold inversion at FT=%g', FT);
    end
    assert(qrobustness.kosut.threshold_time_bandwidth(0.999, 1e-4) < ...
           qrobustness.kosut.threshold_time_bandwidth(0.999), ...
           'nominal error must tighten the threshold');
    assert(qrobustness.kosut.threshold_time_bandwidth(0.999, 2e-3) == 0, ...
           'no headroom -> nothing certifiable');

    % --- rates scale correctly; margin inverts the bound ---------------------
    [H_list, dt] = deterministic_pwc(0, 5, 0.3, SX, SY, SZ);
    dH_list = repmat({0.3 * SX}, 1, numel(H_list));
    rates = qrobustness.kosut.uncertainty_rates(H_list, dH_list, dt);

    assert(abs(rates.T - numel(H_list) * dt) < tol, 'T');
    assert(abs(rates.w_unc - 0.3 * norm(SX, 2)) < tol, 'w_unc');
    assert(rates.w_avg >= 0 && rates.w_dev >= 0, 'non-negative rates');
    assert(abs(qrobustness.kosut.time_bandwidth(rates, 0)) < tol, 'TOb(0) = 0');
    assert(abs(qrobustness.kosut.time_bandwidth(rates, -0.02) - ...
               qrobustness.kosut.time_bandwidth(rates, 0.02)) < tol, 'TOb symmetric');
    assert(qrobustness.kosut.time_bandwidth(rates, 0.01) < ...
           qrobustness.kosut.time_bandwidth(rates, 0.02), 'TOb increasing');

    FT = 0.999;
    M = qrobustness.kosut.margin(rates, FT);
    assert(M > 0, 'positive margin');
    assert(abs(qrobustness.kosut.fidelity_bound_at(rates, M) - FT) < 1e-9, ...
        'margin inverts F_lb');
    assert(qrobustness.kosut.fidelity_bound_at(rates, 1.01 * M) < FT, ...
        'beyond margin the bound drops below FT');

    % --- rates match a brute-force interaction-picture sample ----------------
    [H_list, dt] = deterministic_pwc(3, 4, 0.3, SX, SY, SZ);
    dH_list = repmat({0.5 * SY}, 1, numel(H_list));
    rates = qrobustness.kosut.uncertainty_rates(H_list, dH_list, dt, 40, 201);

    nsamp = 400;
    P = eye(2);
    samples = cell(1, numel(H_list) * nsamp);
    idx = 0;
    ss = linspace(0, dt, nsamp + 1);
    ss = ss(1:end-1);
    for k = 1:numel(H_list)
        for j = 1:nsamp
            US = expm(-1i * H_list{k} * ss(j)) * P;
            idx = idx + 1;
            samples{idx} = US' * dH_list{k} * US;
        end
        P = expm(-1i * H_list{k} * dt) * P;
    end
    Havg = zeros(2);
    for j = 1:idx
        Havg = Havg + samples{j};
    end
    Havg = Havg / idx;
    assert(abs(norm(Havg, 2) - rates.w_avg) <= 1e-3 * rates.w_avg, 'w_avg brute force');
    dev = 0;
    for j = 1:idx
        dev = max(dev, norm(samples{j} - Havg, 2));
    end
    assert(abs(dev - rates.w_dev) <= 1e-2 * rates.w_dev, 'w_dev brute force');

    % --- the bound must actually lower-bound the true perturbed fidelity -----
    [H_list, dt] = deterministic_pwc(7, 6, 0.3, SX, SY, SZ);
    Hhat = SZ;
    dH_list = repmat({Hhat}, 1, numel(H_list));
    rates = qrobustness.kosut.uncertainty_rates(H_list, dH_list, dt);
    Uf = qrobustness.propagator(H_list, dt);   % nominal exact: F_nom = 1
    for delta = [1e-4 1e-3 5e-3 1e-2 3e-2]
        Hp = cell(size(H_list));
        for k = 1:numel(H_list)
            Hp{k} = H_list{k} + delta * Hhat;
        end
        Fp = qrobustness.gate_fidelity(qrobustness.propagator(Hp, dt), Uf);
        assert(Fp >= qrobustness.kosut.fidelity_bound_at(rates, delta) - 1e-12, ...
            'bound violated at delta=%g', delta);
    end

    % --- the implied margin is conservative vs the true crossing -------------
    [H_list, dt] = deterministic_pwc(11, 6, 0.3, SX, SY, SZ);
    dH_list = repmat({SZ}, 1, numel(H_list));
    rates = qrobustness.kosut.uncertainty_rates(H_list, dH_list, dt);
    Uf = qrobustness.propagator(H_list, dt);
    M = qrobustness.kosut.margin(rates, 0.999);
    Hp = cell(size(H_list));
    for k = 1:numel(H_list)
        Hp{k} = H_list{k} + M * SZ;
    end
    assert(qrobustness.gate_fidelity(qrobustness.propagator(Hp, dt), Uf) >= 0.999, ...
        'implied margin must not exceed the true threshold radius');

    % --- degenerate and invalid inputs ---------------------------------------
    [H_list, dt] = deterministic_pwc(5, 3, 0.3, SX, SY, SZ);
    zero_list = repmat({zeros(2)}, 1, numel(H_list));
    rates0 = qrobustness.kosut.uncertainty_rates(H_list, zero_list, dt);
    assert(rates0.w_unc == 0, 'zero structure -> zero w_unc');
    assert(isinf(qrobustness.kosut.margin(rates0, 0.999)), 'zero structure -> Inf margin');

    assert_error(@() qrobustness.kosut.uncertainty_rates({}, {}, dt), 'empty H_list');
    assert_error(@() qrobustness.kosut.uncertainty_rates(H_list, {SX}, dt), 'length mismatch');
    assert_error(@() qrobustness.kosut.uncertainty_rates(H_list, zero_list, 0), 'dt <= 0');
    assert_error(@() qrobustness.kosut.threshold_time_bandwidth(1.0), 'FT out of range');
    assert_error(@() qrobustness.kosut.fidelity_bound(-0.1), 'negative T*Omega_bnd');

end

function [H_list, dt] = deterministic_pwc(seed, tau, dt, SX, SY, SZ)
%DETERMINISTIC_PWC PWC single-qubit Hamiltonians from a fixed closed form.
%   RNG-free so that MATLAB and Octave agree exactly; the Python peer tests use
%   their own draws, since only the golden fixtures are cross-language.
    H_list = cell(1, tau);
    for k = 1:tau
        f1 = sin(1.7 * (k + seed));
        f2 = cos(2.3 * (k + seed) + 0.5);
        H_list{k} = 0.5 * SZ + f1 * SX + f2 * SY;
    end
end

function assert_error(fh, what)
    ok = false;
    try
        fh();
    catch
        ok = true;
    end
    assert(ok, 'expected an error: %s', what);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

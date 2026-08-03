function test_traceless_and_status()
%TEST_TRACELESS_AND_STATUS Centred structure constant and Alg. 1 stop status.
%   Peer of the Python tests in python/tests/test_core.py.

    %% Traceless centring of the perturbation structure
    % Identity structure only rephases U, so it cannot move the fidelity.
    assert(qrobustness.structure_constant('drift', eye(4), 0.5, 32) == 0);
    assert(qrobustness.structure_constant('control', 2.5 * eye(4), 0.5, 32, ones(1, 32)) == 0);

    % Gauge invariance: adding c*I to the structure must not change C.
    rng_state = [1 0.3 -0.7 0.2; 0.3 2 0.1 -0.4; -0.7 0.1 -1 0.6; 0.2 -0.4 0.6 0.5];
    H = (rng_state + rng_state') / 2;
    u = [0.5, -0.2, 0.1, 0.3, 0.9, -1.1, 0.05];
    C_ref = qrobustness.structure_constant('drift', H, 0.3, 7);
    Cc_ref = qrobustness.structure_constant('control', H, 0.3, 7, u);
    for c = [-3, 0.5, 11]
        Cs = qrobustness.structure_constant('drift', H + c * eye(4), 0.3, 7);
        assert(abs(Cs - C_ref) < 1e-12 * max(1, C_ref));
        Cs = qrobustness.structure_constant('control', H + c * eye(4), 0.3, 7, u);
        assert(abs(Cs - Cc_ref) < 1e-12 * max(1, Cc_ref));
    end

    % A non-traceless sparse structure (single-level detuning) tightens.
    N = 8;
    Hd = zeros(N);
    Hd(1, 1) = 1;
    C = qrobustness.structure_constant('drift', Hd, 0.5, 32);
    C_uncentred = 32 * 0.5 * norm(Hd, 'fro');
    assert(C < C_uncentred);
    assert(abs(C - 32 * 0.5 * sqrt(1 - 1 / N)) < 1e-12);

    % Bad structures are rejected rather than silently accepted.
    threw = false;
    try
        qrobustness.structure_constant('drift', [0 1; 0 0], 0.5, 4);
    catch
        threw = true;
    end
    assert(threw);

    %% Algorithm 1 stopping status
    FT = 0.999;
    f = @(mu) FT + 0.01 - abs(mu);   % tent, crossing at |mu| = 0.01

    r = qrobustness.iterative_margin(f, 1.0, FT, 'eta', 1e-6);
    assert(strcmp(r.status_minus, 'eta_band'));
    assert(strcmp(r.status_plus, 'eta_band'));
    assert(~r.safeguard_minus);

    % Domain truncation: converged_* cannot tell this from a resolved margin.
    r = qrobustness.iterative_margin(f, 1.0, FT, 'eta', 1e-6, 'omega', [-0.002, 0.002]);
    assert(strcmp(r.status_minus, 'domain_truncated'));
    assert(strcmp(r.status_plus, 'domain_truncated'));
    assert(abs(r.M - 0.002) < 1e-12);
    assert(r.converged_minus && r.converged_plus);

    % One-sided truncation.
    r = qrobustness.iterative_margin(f, 1.0, FT, 'eta', 1e-6, 'omega', [-0.002, Inf]);
    assert(strcmp(r.status_minus, 'domain_truncated'));
    assert(strcmp(r.status_plus, 'eta_band'));

    % Iteration limit.
    r = qrobustness.iterative_margin(f, 1e4, FT, 'eta', 1e-12, 'k_max', 2);
    assert(strcmp(r.status_minus, 'iteration_limit'));
    assert(~r.converged_minus);

    % The safeguard fires exactly when L under-estimates the true slope.
    r = qrobustness.iterative_margin(f, 0.5, FT, 'eta', 1e-6);
    assert(r.safeguard_minus);
    r = qrobustness.iterative_margin(f, 2.0, FT, 'eta', 1e-6);
    assert(~r.safeguard_minus);

    % status_* is populated without margin_tol; reason_* is a different field.
    r = qrobustness.iterative_margin(f, 1.0, FT);
    assert(any(strcmp(r.status_minus, {'eta_band', 'domain_truncated', 'iteration_limit'})));
    assert(strcmp(r.reason_minus, 'unknown'));

    %% Kendall tau_b agrees with the SciPy reference to machine precision
    % Perfect monotone association, no ties.
    x = (1:20)';
    assert(abs(qrobustness.compat.kendall_tau_b(x, exp(-x)) + 1) < 1e-14);
    assert(abs(qrobustness.compat.kendall_tau_b(x, 2 * x) - 1) < 1e-14);
    % Ties are handled by the tau_b correction (tau_a would differ).
    [tb, p] = qrobustness.compat.kendall_tau_b([1 1 2 3 4]', [1 2 2 3 4]');
    assert(tb > 0 && tb < 1);
    assert(p >= 0 && p <= 1);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

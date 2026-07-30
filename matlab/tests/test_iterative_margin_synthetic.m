function test_iterative_margin_synthetic()
    % Synthetic landscape F(mu) = 1 - a*(mu-mu_star)^2 near peak, clipped.
    % Use a monotone-decreasing-from-nominal model for Algorithm 1.
    FT = 0.99;
    mu0 = 0;
    % F(mu) = 1 - 0.05*|mu|  => crosses FT=0.99 at |mu|=0.2
    fidelity_fn = @(mu) max(0, 1 - 0.05 * abs(mu));
    F0 = fidelity_fn(mu0);
    assert(F0 > FT);

    % Lipschitz: |dF/dmu| = 0.05
    L = 0.05;
    res = qrobustness.iterative_margin(fidelity_fn, L, FT, ...
        'mu0', mu0, 'eta', 1e-8, 'k_max', 1000);

    assert(abs(res.M_minus - 0.2) < 1e-5, 'M_minus=%g', res.M_minus);
    assert(abs(res.M_plus - 0.2) < 1e-5, 'M_plus=%g', res.M_plus);
    assert(abs(res.M - 0.2) < 1e-5);
    assert(res.converged_minus && res.converged_plus);

    % Domain boundary: omega = [-0.05, Inf] should stop at boundary on minus side
    res_b = qrobustness.iterative_margin(fidelity_fn, L, FT, ...
        'mu0', mu0, 'eta', 1e-8, 'omega', [-0.05, Inf], 'k_max', 1000);
    assert(abs(res_b.M_minus - 0.05) < 1e-8);
    assert(abs(res_b.mu_minus + 0.05) < 1e-8);

    % Overshoot / bisection: large L still finds the threshold neighborhood
    res_big = qrobustness.iterative_margin(fidelity_fn, 10, FT, ...
        'mu0', mu0, 'eta', 1e-6, 'k_max', 5000);
    assert(abs(res_big.M - 0.2) < 5e-4, 'M=%g with large L', res_big.M);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

function test_margin_solvers_selectable()
%TEST_MARGIN_SOLVERS_SELECTABLE Selectable methods vs Algorithm 1 on synthetic F.

    FT = 0.99;
    mu0 = 0;
    fidelity_fn = @(mu) max(0, 1 - 0.05 * abs(mu));
    L = 0.05;

    base = qrobustness.iterative_margin(fidelity_fn, L, FT, ...
        'mu0', mu0, 'eta', 1e-8, 'k_max', 1000, 'method', 'algorithm1');

    for methods = {'lipschitz_brent', 'lipschitz_toms748', 'doubling'}
        method = methods{1};
        alt = qrobustness.iterative_margin(fidelity_fn, L, FT, ...
            'mu0', mu0, 'eta', 1e-8, 'k_max', 1000, 'method', method, ...
            'return_diagnostics', true);
        assert(abs(alt.M - base.M) < 1e-5, '%s M=%g vs %g', method, alt.M, base.M);
        assert(isfield(alt, 'n_evals') && alt.n_evals > 0);
    end

    zeta_fn = @(mu) local_zeta(mu);
    newt = qrobustness.iterative_margin(fidelity_fn, L, FT, ...
        'mu0', mu0, 'eta', 1e-8, 'k_max', 1000, ...
        'method', 'newton_probe', 'zeta_fn', zeta_fn, ...
        'return_diagnostics', true);
    assert(abs(newt.M - base.M) < 1e-4, 'newton_probe M=%g', newt.M);

    % Default path still Algorithm 1
    def = qrobustness.iterative_margin(fidelity_fn, L, FT, ...
        'mu0', mu0, 'eta', 1e-8, 'k_max', 1000);
    assert(abs(def.M - base.M) < 1e-14);
    assert(strcmp(def.method, 'algorithm1'));
end

function z = local_zeta(mu)
    if mu > 0
        z = -0.05;
    elseif mu < 0
        z = 0.05;
    else
        z = -0.05;
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

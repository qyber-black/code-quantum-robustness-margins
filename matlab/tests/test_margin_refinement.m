function test_margin_refinement()
%TEST_MARGIN_REFINEMENT Safe islands must not enlarge the nominal component.
    knots = [0 1.01 1.02 3 4.1];
    values = [1 0.798 1 1 0.78];
    calls = 0;
    f = @counted_fidelity;
    r = qrobustness.iterative_margin(f, 20.2, 0.8, 'eta', 0.05, ...
        'margin_tol', 0.01, 'return_diagnostics', true);
    assert(r.n_evals == calls);
    assert(r.M <= 1 + 1e-12, 'Refinement crossed the first unsafe interval');
    assert(r.M_upper >= 1 && f(r.M_upper) < 0.8);
    assert(r.n_evals == r.n_evals_continuation + r.n_evals_refinement);
    assert(r.n_evals_continuation > 0 && r.n_evals_refinement > 0);
    calls = 0;
    basic = qrobustness.iterative_margin(f, 20.2, 0.8, 'eta', 0.05, ...
        'return_diagnostics', true);
    assert(basic.n_evals == calls);
    assert(basic.n_evals_refinement == 0);
    assert(basic.n_evals == basic.n_evals_continuation);
    assert(basic.M == min(abs(r.mu_minus), abs(r.mu_plus)));

    function value = counted_fidelity(mu)
        calls = calls + 1;
        value = interp1(knots, values, min(abs(mu), 4.1), 'linear');
    end

end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

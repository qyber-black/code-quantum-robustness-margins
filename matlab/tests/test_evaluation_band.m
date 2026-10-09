% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function test_evaluation_band()
%TEST_EVALUATION_BAND Evaluation band, zero-gauge directions and per-direction evaluation counts.
%   Peer of python/tests/test_evaluation_band.py.

    bias = 1e-4;
    biased = @(mu) 1 - abs(mu) + bias;
    for tol = {[], 1e-8}
        r = qrobustness.iterative_margin(biased, 1.0, 0.5, 'eval_tol', bias, 'margin_tol', tol{1});
        assert(r.M_plus <= 0.5 && r.M_minus <= 0.5, 'band keeps the prefix below the true crossing');
    end
    r = qrobustness.iterative_margin(biased, 1.0, 0.5);
    assert(r.M_plus > 0.5, 'without the band the biased evaluation overshoots');

    % F = F_T + eval_tol is inside the closed band, so the continuation does not certify it.
    FT = 0.5;
    tol = 0.1;
    fn = @(mu) 0.9 * (abs(mu) <= 1) + (FT + tol) * (abs(mu) > 1);
    r = qrobustness.iterative_margin(fn, 1.0, FT, 'eval_tol', tol, 'omega', [-5, 5]);
    assert(r.M_plus <= 1 && r.M_minus <= 1, 'continuation certified the band edge');

    f = @(mu) cos(mu)^2;
    a = qrobustness.iterative_margin(f, 2.0, 0.9, 'margin_tol', 1e-8);
    b = qrobustness.iterative_margin(f, 2.0, 0.9, 'margin_tol', 1e-8, 'eval_tol', 0);
    assert(isequal([a.M_minus, a.M_plus, a.M_upper], [b.M_minus, b.M_plus, b.M_upper]), 'zero band unchanged');
    r = qrobustness.iterative_margin(f, 2.0, 0.9, 'margin_tol', 1e-8, 'return_diagnostics', true);
    assert(r.n_evals_minus > 0 && r.n_evals_plus > 0 && r.n_evals_minus + r.n_evals_plus + 1 == r.n_evals, 'per-ray counts');

    X = [0, 1; 1, 0];
    Z = diag([1, -1]);
    K = 4;
    dt = 0.25;
    H = repmat({pi / 2 * X}, 1, K);
    Uf = qrobustness.propagator(H, dt);
    fid = @(x) qrobustness.gate_fidelity(qrobustness.propagator(cellfun(@(h) h + (x(1) + x(2)) * Z, H, ...
                                                                        'UniformOutput', false), dt), Uf);
    d = [1; -1] / sqrt(2);
    ag = qrobustness.multiparam.angular_gauge({repmat({Z}, 1, K), repmat({Z}, 1, K)}, dt);
    assert(abs(qrobustness.lengthspace.path_gauge_C(ag, d)) < 1e-15);
    r = qrobustness.multiparam.directional_margin(fid, [1; 1], 0.99, d, 'angular_gauge', ag, 'omega', [-2, 3]);
    assert(strcmp(r.reason_plus, 'zero_gauge') && r.M == 2, 'zero angular gauge certifies the ray');
end

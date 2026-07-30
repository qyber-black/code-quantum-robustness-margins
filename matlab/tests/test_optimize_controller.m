function test_optimize_controller()
%TEST_OPTIMIZE_CONTROLLER GRAPE gradient FD check + one optimisation improves F.

    this_dir = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(this_dir));
    CTRL = fullfile(root, 'data', 'controllers', 'problem9_tf15_K32_quasi-newton');
    problem = qrobustness.load_problem(fullfile(CTRL, 'problem9.mat'));

    % FD check on both segment-derivative paths.
    methods = {'exact', 'quadrature'};
    for mi = 1:numel(methods)
        method = methods{mi};

        rng(0);
        tau = 4;
        tf = 2;
        dt = tf / tau;
        u1 = randn(1, tau);
        u2 = randn(1, tau);

        fg = @(a, b) qrobustness.fidelity_and_gradient(problem.H0, problem.H1, problem.H2, ...
            a, b, problem.Uf, dt, 'method', method, 'n_quad', 48);

        [F, g1, g2] = fg(u1, u2);

        H_list = qrobustness.perturbed_hamiltonians( ...
            problem.H0, problem.H1, problem.H2, u1, u2, 'H0', 0);
        Fref = qrobustness.gate_fidelity(qrobustness.propagator(H_list, dt), problem.Uf);
        assert(abs(F - Fref) < 1e-12, '%s: F vs propagator diff %g', method, abs(F - Fref));

        h = 1e-7;
        k = 2;
        u1p = u1; u1p(k) = u1p(k) + h;
        u1m = u1; u1m(k) = u1m(k) - h;
        Fp = fg(u1p, u2);
        Fm = fg(u1m, u2);
        g1_fd = (Fp - Fm) / (2 * h);
        assert(abs(g1(k) - g1_fd) / max(1e-8, abs(g1_fd)) < 5e-4, '%s: g1 FD mismatch', method);

        u2p = u2; u2p(k) = u2p(k) + h;
        u2m = u2; u2m(k) = u2m(k) - h;
        Fp = fg(u1, u2p);
        Fm = fg(u1, u2m);
        g2_fd = (Fp - Fm) / (2 * h);
        assert(abs(g2(k) - g2_fd) / max(1e-8, abs(g2_fd)) < 5e-4, '%s: g2 FD mismatch', method);
    end

    rng(42);
    u1 = randn(1, 32);
    u2 = randn(1, 32);
    res = qrobustness.optimize_controller( ...
        problem.H0, problem.H1, problem.H2, problem.Uf, 15, 32, ...
        'u1_init', u1, 'u2_init', u2, 'maxiter', 80, 'ftol', 1e-10);
    assert(res.fid > res.fid_init);
    assert(abs(res.error - (1 - res.fid)) < 1e-15);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

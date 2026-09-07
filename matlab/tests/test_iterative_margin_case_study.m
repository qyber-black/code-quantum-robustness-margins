function test_iterative_margin_case_study()
%TEST_ITERATIVE_MARGIN_CASE_STUDY Algorithm 1 certificate on real case-study data.

    this_dir = fileparts(mfilename('fullpath'));
    root_dir = fileparts(fileparts(this_dir));
    CTRL = fullfile(root_dir, 'data', 'controllers', 'problem9_tf15_K32_quasi-newton');

    FT = 0.999;
    eta = 1e-6;
    problem = qrobustness.load_problem(fullfile(CTRL, 'problem9.mat'));
    controllers = qrobustness.load_controllers(fullfile(CTRL, 'controllers.csv'), 1e-4);
    c = controllers{1};  % controller 1 (paper indexing)
    dt = c.tf / c.tau;

    C = qrobustness.structure_constant('drift', problem.H0, dt, c.tau);
    L = qrobustness.lipschitz_constant(FT, problem.dim, C);
    fid_fn = qrobustness.make_fidelity_fn( ...
        problem.H0, problem.H1, problem.H2, c.u1, c.u2, problem.Uf, dt, 'H0');
    % margin_tol matches the production driver: the published margins table
    % this test compares against is generated with bracket refinement, so
    % without it the two differ by ~5e-4 and the 1e-10 assertions below fail.
    res = qrobustness.iterative_margin(fid_fn, L, FT, ...
        'mu0', 0, 'eta', eta, 'margin_tol', 1e-8);

    assert(res.M == min(res.M_minus, res.M_plus));
    assert(res.converged_minus && res.converged_plus);
    assert(fid_fn(-res.M) >= FT - 1e-12);
    assert(fid_fn(res.M) >= FT - 1e-12);

    table_csv = fullfile(root_dir, 'results', 'lipschitz-margin-matlab', 'margins_table_0.999.csv');
    if exist(table_csv, 'file')
        T = qrobustness.compat.read_margins_csv(table_csv);
        assert(abs(res.M - T.M_H0(1)) < 1e-10);
        assert(abs(res.M_minus - T.Mm_H0(1)) < 1e-10);
        assert(abs(res.M_plus - T.Mp_H0(1)) < 1e-10);
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

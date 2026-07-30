function test_load_case_study_smoke()
    this_dir = fileparts(mfilename('fullpath'));
    root_dir = fileparts(fileparts(this_dir));
    CTRL = fullfile(root_dir, 'data', 'controllers', 'problem9_tf15_K32_quasi-newton');
    mat_path = fullfile(CTRL, 'problem9.mat');
    csv_path = fullfile(CTRL, 'controllers.csv');

    problem = qrobustness.load_problem(mat_path);
    assert(problem.dim == 8);
    assert(isequal(size(problem.H0), [8 8]));

    controllers = qrobustness.load_controllers(csv_path, 1e-4);
    assert(numel(controllers) == 61);

    c = controllers{1};
    dt = c.tf / c.tau;
    H_list = qrobustness.perturbed_hamiltonians( ...
        problem.H0, problem.H1, problem.H2, c.u1, c.u2, 'H0', 0);
    U = qrobustness.propagator(H_list, dt);
    F = qrobustness.gate_fidelity(U, problem.Uf);
    assert(abs(F - c.fid) < 1e-4, 'F=%g csv_fid=%g', F, c.fid);

    C = qrobustness.structure_constant('drift', problem.H0, dt, c.tau);
    L = qrobustness.lipschitz_constant(0.999, problem.dim, C);
    assert(L > 0);

    dH = qrobustness.dH_structure(problem.H0, problem.H1, problem.H2, c.u1, c.u2, 'H0');
    zeta = qrobustness.differential_sensitivity(H_list, dH, dt, problem.Uf, 24);
    assert(isfinite(zeta));
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

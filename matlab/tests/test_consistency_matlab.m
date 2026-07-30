function test_consistency_matlab()
%TEST_CONSISTENCY_MATLAB Compare live MATLAB values to Python golden JSON.

    this_dir = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(this_dir));
    addpath(fullfile(root, 'matlab'));

    golden_path = fullfile(root, 'data', 'reference', 'case_study_subset.json');
    assert(isfile(golden_path), 'Missing golden JSON; run make export-golden');

    txt = fileread(golden_path);
    golden = jsondecode(txt);
    records = golden.records;

    FT = 0.999;
    eta = 1e-6;
    rtol = 1e-8;
    atol = 1e-10;

    CTRL = fullfile(root, 'data', 'controllers', 'problem9_tf15_K32_quasi-newton');
    problem = qrobustness.load_problem(fullfile(CTRL, 'problem9.mat'));
    controllers = qrobustness.load_controllers(fullfile(CTRL, 'controllers.csv'), 1e-4);

    nrec = numel(records);
    for k = 1:nrec
        rec = records(k);
        idx = rec.controller_index + 1; % JSON is 0-based
        tag = char(rec.structure);
        c = controllers{idx};
        dt = c.tf / c.tau;
        switch tag
            case 'H0'
                C = qrobustness.structure_constant('drift', problem.H0, dt, c.tau);
            case 'H1'
                C = qrobustness.structure_constant('control', problem.H1, dt, c.tau, c.u1);
            case 'H2'
                C = qrobustness.structure_constant('control', problem.H2, dt, c.tau, c.u2);
        end
        L = qrobustness.lipschitz_constant(FT, problem.dim, C);
        fid_fn = qrobustness.make_fidelity_fn( ...
            problem.H0, problem.H1, problem.H2, c.u1, c.u2, problem.Uf, dt, tag);
        F0 = fid_fn(0);
        margin = qrobustness.iterative_margin(fid_fn, L, FT, 'mu0', 0, 'eta', eta);
        H_list = qrobustness.perturbed_hamiltonians( ...
            problem.H0, problem.H1, problem.H2, c.u1, c.u2, tag, 0);
        dH = qrobustness.dH_structure(problem.H0, problem.H1, problem.H2, c.u1, c.u2, tag);
        zeta = qrobustness.differential_sensitivity(H_list, dH, dt, problem.Uf, 32);
        rates = qrobustness.kosut.uncertainty_rates(H_list, dH, dt, 24, 17);
        kM = qrobustness.kosut.margin(rates, FT, c.error);

        assert_close(F0, rec.fid, rtol, atol, sprintf('%s fid', tag));
        assert_close(L, rec.L, rtol, atol, sprintf('%s L', tag));
        assert_close(C, rec.C, rtol, atol, sprintf('%s C', tag));
        assert_close(zeta, rec.zeta, rtol, atol, sprintf('%s zeta', tag));
        assert_close(margin.M, rec.M, rtol, atol, sprintf('%s M', tag));
        assert_close(margin.M_minus, rec.M_minus, rtol, atol, sprintf('%s M-', tag));
        assert_close(margin.M_plus, rec.M_plus, rtol, atol, sprintf('%s M+', tag));
        assert(logical(margin.converged_minus), '%s converged_minus', tag);
        assert(logical(margin.converged_plus), '%s converged_plus', tag);
        assert(logical(rec.converged_minus), '%s golden converged_minus', tag);
        assert(logical(rec.converged_plus), '%s golden converged_plus', tag);
        % Supplementary Kosut et al. fields (absent from pre-Kosut goldens).
        if isfield(rec, 'k_M')
            assert_close(rates.w_unc, rec.k_w_unc, rtol, atol, sprintf('%s k_w_unc', tag));
            assert_close(rates.w_avg, rec.k_w_avg, rtol, atol, sprintf('%s k_w_avg', tag));
            assert_close(rates.w_dev, rec.k_w_dev, rtol, atol, sprintf('%s k_w_dev', tag));
            assert_close(rates.T, rec.k_T, rtol, atol, sprintf('%s k_T', tag));
            assert_close(kM, rec.k_M, rtol, atol, sprintf('%s k_M', tag));
            assert(kM <= margin.M, '%s structured margin should not be the smaller one', tag);
        end
    end
    assert(nrec >= 9, 'expected >=3 controllers x 3 structures, got %d', nrec);
    fprintf('PASS  test_consistency_matlab (%d records)\n', nrec);
end

function assert_close(a, b, rtol, atol, name)
    if ~(abs(a - b) <= atol + rtol * max(abs(a), abs(b)))
        error('qrobustness:consistency', '%s: %g vs %g', name, a, b);
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

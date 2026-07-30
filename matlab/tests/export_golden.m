function export_golden()
%EXPORT_GOLDEN Write MATLAB numerics for the golden subset (for comparison).
% Indices match scripts/export_golden.py (0-based in JSON): 0, 58, 60.

    this_dir = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(this_dir));
    addpath(fullfile(root, 'matlab'));

    FT = 0.999;
    eta = 1e-6;
    kosut_n_quad = 24;
    kosut_n_dev = 17;
    idx_list = [1 59 61];  % 1-based; matches Python 0, 58, 60 after filter
    structures = {'H0', 'H1', 'H2'};
    CTRL = fullfile(root, 'data', 'controllers', 'problem9_tf15_K32_quasi-newton');

    problem = qrobustness.load_problem(fullfile(CTRL, 'problem9.mat'));
    controllers = qrobustness.load_controllers(fullfile(CTRL, 'controllers.csv'), 1e-4);

    records = struct([]);
    r = 0;
    for ii = 1:numel(idx_list)
        c = controllers{idx_list(ii)};
        dt = c.tf / c.tau;
        for s = 1:3
            tag = structures{s};
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
            % Supplementary Kosut et al. bound (arXiv:2507.01215).
            rates = qrobustness.kosut.uncertainty_rates(H_list, dH, dt, kosut_n_quad, kosut_n_dev);
            kM = qrobustness.kosut.margin(rates, FT, c.error);

            r = r + 1;
            records(r).controller_index = idx_list(ii) - 1; % 0-based like Python
            records(r).structure = tag;
            records(r).fid = F0;
            records(r).L = L;
            records(r).C = C;
            records(r).zeta = zeta;
            records(r).M = margin.M;
            records(r).M_minus = margin.M_minus;
            records(r).M_plus = margin.M_plus;
            records(r).converged_minus = margin.converged_minus;
            records(r).converged_plus = margin.converged_plus;
            records(r).k_w_unc = rates.w_unc;
            records(r).k_w_avg = rates.w_avg;
            records(r).k_w_dev = rates.w_dev;
            records(r).k_T = rates.T;
            records(r).k_M = kM;
        end
    end

    out_dir = fullfile(root, 'data', 'reference');
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end
    out_mat = fullfile(out_dir, 'case_study_subset_matlab.mat');
    save(out_mat, 'records', 'FT', 'eta', '-v7');

    out_json = fullfile(out_dir, 'case_study_subset_matlab.json');
    fid = fopen(out_json, 'w');
    % n_quad / kosut_n_quad are recorded for provenance; zeta and <Htil> are
    % closed form and do not use them.  The *_method fields say what is used.
    fprintf(fid, '{\n  "meta": {"FT": %.10g, "eta": %.10g, "n_quad": 32, "indices": [0, 58, 60], "kosut_n_quad": %d, "kosut_n_dev": %d, "zeta_method": "exact", "kosut_w_avg": "exact", "kosut_w_dev": "grid+brent, adaptive"},\n  "records": [\n', FT, eta, kosut_n_quad, kosut_n_dev);
    for k = 1:numel(records)
        rec = records(k);
        fprintf(fid, '    {"controller_index": %d, "structure": "%s", "fid": %.16g, "L": %.16g, "C": %.16g, "zeta": %.16g, "M": %.16g, "M_minus": %.16g, "M_plus": %.16g, "converged_minus": %s, "converged_plus": %s, "k_w_unc": %.16g, "k_w_avg": %.16g, "k_w_dev": %.16g, "k_T": %.16g, "k_M": %.16g}', ...
            rec.controller_index, rec.structure, rec.fid, rec.L, rec.C, rec.zeta, ...
            rec.M, rec.M_minus, rec.M_plus, ...
            lower(mat2str(logical(rec.converged_minus))), ...
            lower(mat2str(logical(rec.converged_plus))), ...
            rec.k_w_unc, rec.k_w_avg, rec.k_w_dev, rec.k_T, rec.k_M);
        if k < numel(records)
            fprintf(fid, ',\n');
        else
            fprintf(fid, '\n');
        end
    end
    fprintf(fid, '  ]\n}\n');
    fclose(fid);
    fprintf('Wrote %s and %s (%d records)\n', out_mat, out_json, numel(records));
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

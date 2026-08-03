function run_lipschitz_margin_case_study(varargin)
%RUN_LIPSCHITZ_MARGIN_CASE_STUDY Reproduce the three-qubit robustness-margin case study.
%
%   Name-value options:
%     'max_controllers'  [] = all accepted (61); set small for smoke runs
%     'FT'               0.999
%     'max_error'        1e-4
%     'eta'              1e-6
%     'do_sweep'         true  (fidelity-vs-delta overlays)
%     'root'             auto-detect repo root
%     'controller_set'   'problem9_tf15_K32_quasi-newton' (under data/controllers/)
%     'controller_dir'   '' -> data/controllers/<controller_set>; or absolute path
%                        (e.g. results/synth-matlab) with problem9.mat + controllers.csv
%     'results_id'       'lipschitz-margin-matlab'  -> results/<id>/
%     'build_dir'        build/ (regenerable intermediates)
%     'publish_dir'      results/<results_id>/ (paper deliverables)

    % Choose the graphics toolkit before any figure exists.
    qrobustness.compat.setup_graphics();

    p = inputParser;
    addParameter(p, 'max_controllers', []);
    addParameter(p, 'FT', 0.999);
    addParameter(p, 'max_error', 1e-4);
    addParameter(p, 'eta', 1e-6);
    addParameter(p, 'do_sweep', true);
    addParameter(p, 'root', '');
    addParameter(p, 'controller_set', 'problem9_tf15_K32_quasi-newton');
    addParameter(p, 'controller_dir', '');
    addParameter(p, 'results_id', 'lipschitz-margin-matlab');
    addParameter(p, 'build_dir', '');
    addParameter(p, 'publish_dir', '');
    parse(p, varargin{:});
    opt = p.Results;

    if isempty(opt.root)
        this_dir = fileparts(mfilename('fullpath'));
        opt.root = fileparts(fileparts(this_dir));
    end
    addpath(fullfile(opt.root, 'matlab'));

    if isempty(opt.controller_dir)
        ctrl_dir = fullfile(opt.root, 'data', 'controllers', opt.controller_set);
    else
        ctrl_dir = opt.controller_dir;
    end
    mat_path = fullfile(ctrl_dir, 'problem9.mat');
    csv_path = fullfile(ctrl_dir, 'controllers.csv');

    if isempty(opt.build_dir)
        out_build = fullfile(opt.root, 'build');
    else
        out_build = opt.build_dir;
    end
    if isempty(opt.publish_dir)
        out_publish = fullfile(opt.root, 'results', opt.results_id);
    else
        out_publish = opt.publish_dir;
    end
    if ~exist(out_build, 'dir'); mkdir(out_build); end
    if ~exist(out_publish, 'dir'); mkdir(out_publish); end

    problem = qrobustness.load_problem(mat_path);
    controllers = qrobustness.load_controllers(csv_path, opt.max_error);
    if ~isempty(opt.max_controllers)
        controllers = controllers(1:min(opt.max_controllers, numel(controllers)));
    end
    nC = numel(controllers);
    structures = {'H0', 'H1', 'H2'};
    FT = opt.FT;

    fprintf('Case study: %d controllers, FT=%g\n', nC, FT);

    for s = 1:3
        tag = structures{s};
        R.(tag).M = zeros(nC, 1);
        R.(tag).M_minus = zeros(nC, 1);
        R.(tag).M_plus = zeros(nC, 1);
        R.(tag).zeta = zeros(nC, 1);
        R.(tag).L = zeros(nC, 1);
        R.(tag).fid = zeros(nC, 1);
        R.(tag).error = zeros(nC, 1);
        R.(tag).converged = false(nC, 2);
        sweeps.(tag).X = cell(nC, 1);
        sweeps.(tag).Y = cell(nC, 1);
    end

    for n = 1:nC
        c = controllers{n};
        dt = c.tf / c.tau;
        fprintf('Controller %d / %d (fid=%.6g)\n', n, nC, c.fid);

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

            margin = qrobustness.iterative_margin(fid_fn, L, FT, ...
                'mu0', 0, 'eta', opt.eta);

            H_list = qrobustness.perturbed_hamiltonians( ...
                problem.H0, problem.H1, problem.H2, c.u1, c.u2, tag, 0);
            dH = qrobustness.dH_structure( ...
                problem.H0, problem.H1, problem.H2, c.u1, c.u2, tag);
            zeta = qrobustness.differential_sensitivity(H_list, dH, dt, problem.Uf, 32);

            R.(tag).M(n) = margin.M;
            R.(tag).M_minus(n) = margin.M_minus;
            R.(tag).M_plus(n) = margin.M_plus;
            R.(tag).zeta(n) = zeta;
            R.(tag).L(n) = L;
            R.(tag).fid(n) = c.fid;
            R.(tag).error(n) = c.error;
            R.(tag).converged(n, :) = [margin.converged_minus, margin.converged_plus];

            if opt.do_sweep
                span = 1.05 * max(margin.M_minus, margin.M_plus);
                if span <= 0
                    span = 1e-3;
                end
                x = linspace(-span, span, 401);
                [x, F] = qrobustness.fidelity_vs_delta(fid_fn, x);
                sweeps.(tag).X{n} = x;
                sweeps.(tag).Y{n} = 1 - F;
            end
        end
    end

    % Regenerable intermediates -> build/
    save(fullfile(out_build, sprintf('data_consolidated_robustness_%g.mat', FT)), ...
        'R', 'controllers', 'FT', '-v7');

    T = qrobustness.compat.build_margins_table(R, nC);
    margins_csv = sprintf('margins_table_%g.csv', FT);
    qrobustness.compat.write_margins_csv(fullfile(out_build, margins_csv), T);
    copyfile(fullfile(out_build, margins_csv), fullfile(out_publish, margins_csv));
    write_correlations(T, fullfile(out_build, sprintf('correlations_%g.csv', FT)));

    % Working figures -> build/; publish PNGs -> results/<id>/
    if opt.do_sweep
        xlims = {[-8e-3, 8e-3], [-2e-2, 2e-2], [-2e-2, 2e-2]};
        for s = 1:3
            tag = structures{s};
            fig = qrobustness.plot_fidelity_error_sweeps( ...
                sweeps.(tag).X, sweeps.(tag).Y, FT, ...
                'xlabel', sprintf('Perturbation strength \\delta_{%d}', s - 1), ...
                'xlim', xlims{s});
            qrobustness.compat.export_figure(fig, fullfile(out_build, sprintf('%s_all.png', tag)), 300);
            qrobustness.compat.export_figure(fig, fullfile(out_publish, sprintf('%s_all.png', tag)), 300);
            if ~qrobustness.compat.is_octave()
                savefig(fig, fullfile(out_build, sprintf('%s_all.fig', tag)));
            end
            close(fig);
        end
    end

    fig = qrobustness.plot_margins_vs_index(R.H0.error, R.H0.M, R.H1.M, R.H2.M);
    qrobustness.compat.export_figure(fig, fullfile(out_build, 'robustness_margins_fid_err.png'), 300);
    qrobustness.compat.export_figure(fig, fullfile(out_publish, 'robustness_margins_fid_err.png'), 300);
    if ~qrobustness.compat.is_octave()
        savefig(fig, fullfile(out_build, 'robustness_margins_fid_err.fig'));
    end
    close(fig);

    fig = qrobustness.plot_margins_vs_sensitivity( ...
        abs(R.H0.zeta), abs(R.H1.zeta), abs(R.H2.zeta), ...
        R.H0.M, R.H1.M, R.H2.M);
    qrobustness.compat.export_figure(fig, fullfile(out_build, 'robustness_margins_sensitivity.png'), 300);
    qrobustness.compat.export_figure(fig, fullfile(out_publish, 'robustness_margins_sensitivity.png'), 300);
    if ~qrobustness.compat.is_octave()
        savefig(fig, fullfile(out_build, 'robustness_margins_sensitivity.fig'));
    end
    close(fig);

    tex_name = sprintf('correlations_%g.tex', FT);
    write_correlation_tex(T, fullfile(out_build, tex_name));
    copyfile(fullfile(out_build, tex_name), fullfile(out_publish, tex_name));

    focal_name = sprintf('focal_tests_%g.csv', FT);
    write_focal_tests(T, fullfile(out_build, focal_name));
    copyfile(fullfile(out_build, focal_name), fullfile(out_publish, focal_name));

    fprintf('Wrote build artefacts to %s\n', out_build);
    fprintf('Published paper deliverables to %s\n', out_publish);
end

function write_focal_tests(T, path)
    % Cross-check, not a paper claim: Table I is reported descriptively.
    % Kendall's tau_b confirms the reading does not depend on the choice of
    % rank statistic, with a Holm correction across the family of three so
    % the multiplicity is fixed in advance.
    n = numel(T.M_H0);
    taus = zeros(1, 3);
    pvals = zeros(1, 3);
    for j = 0:2
        M = T.(sprintf('M_H%d', j));
        Z = abs(T.(sprintf('zeta_H%d', j)));
        [taus(j+1), pvals(j+1)] = qrobustness.compat.kendall_tau_b(M, Z);
    end
    padj = holm(pvals);
    fid = fopen(path, 'w');
    fprintf(fid, 'comparison,n,kendall_tau_b,kendall_p_two_sided,kendall_p_holm\n');
    for j = 0:2
        fprintf(fid, 'M_H%d_vs_abs_zeta_H%d,%d,%.6f,%.6e,%.6e\n', ...
            j, j, n, taus(j+1), pvals(j+1), padj(j+1));
    end
    fclose(fid);
end

function adj = holm(pvals)
    % Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1).
    m = numel(pvals);
    [~, order] = sort(pvals);
    adj = zeros(1, m);
    running = 0;
    for rank = 1:m
        idx = order(rank);
        running = max(running, (m - rank + 1) * pvals(idx));
        adj(idx) = min(1, running);
    end
end

function write_correlations(T, path)
    % Sensitivities enter as |zeta|: min(M-, M+) is invariant under reversal
    % of the parameter coordinate while zeta changes sign (see
    % write_correlation_tex).
    vars = {'err', 'M_H0', 'M_H1', 'M_H2', 'zeta_H0', 'zeta_H1', 'zeta_H2'};
    names = {'err', 'M_H0', 'M_H1', 'M_H2', 'abs_zeta_H0', 'abs_zeta_H1', 'abs_zeta_H2'};
    X = qrobustness.compat.margins_matrix(T, vars);
    X(:, 5:7) = abs(X(:, 5:7));
    [P, S] = qrobustness.compat.correlation_matrices(X);
    fid = fopen(path, 'w');
    fprintf(fid, 'metric,%s\n', strjoin(names, ','));
    for i = 1:numel(vars)
        fprintf(fid, 'pearson_%s', names{i});
        for j = 1:numel(vars)
            fprintf(fid, ',%.6f', P(i, j));
        end
        fprintf(fid, '\n');
    end
    for i = 1:numel(vars)
        fprintf(fid, 'spearman_%s', names{i});
        for j = 1:numel(vars)
            fprintf(fid, ',%.6f', S(i, j));
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end

function write_correlation_tex(T, path)
    % M_j = min(M_j-, M_j+) is invariant under reversal of the parameter
    % coordinate while zeta_j changes sign, so |zeta_j| is the
    % orientation-invariant local comparator; correlating against signed zeta
    % can hide a relationship by mixing the two signs.  Matches Fig. 3.
    vars = {'err', 'M_H0', 'M_H1', 'M_H2', 'zeta_H0', 'zeta_H1', 'zeta_H2'};
    labels = {'$\varepsilon_0$', '$M_0$', '$M_1$', '$M_2$', '$|\zeta_0|$', '$|\zeta_1|$', '$|\zeta_2|$'};
    X = qrobustness.compat.margins_matrix(T, vars);
    X(:, 5:7) = abs(X(:, 5:7));
    [P, S] = qrobustness.compat.correlation_matrices(X);
    fid = fopen(path, 'w');
    fprintf(fid, '%% Auto-generated: upper Pearson r, lower Spearman rho; |zeta| (see above)\n');
    fprintf(fid, '\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\n');
    fprintf(fid, ' & %s \\\\\n\\midrule\n', strjoin(labels, ' & '));
    for i = 1:7
        fprintf(fid, '%s', labels{i});
        for j = 1:7
            if i == j
                v = 1.0;
            elseif j > i
                v = P(i, j);
            else
                v = S(i, j);
            end
            fprintf(fid, ' & $%.2f$', v);
        end
        fprintf(fid, ' \\\\\n');
    end
    fprintf(fid, '\\bottomrule\n\\end{tabular}\n');
    fclose(fid);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

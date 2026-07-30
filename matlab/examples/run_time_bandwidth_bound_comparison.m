function run_time_bandwidth_bound_comparison(varargin)
%RUN_TIME_BANDWIDTH_BOUND_COMPARISON Compare the Lipschitz margin with the Kosut et al. bound.
%
%   Supplementary analysis (not part of the paper's main results): for every
%   controller and perturbation structure, computes the certified margin M of
%   Algorithm 1 alongside the margin implied by Theorem 1 of arXiv:2507.01215,
%   specialised to this closed-system coherent perturbation model.  See
%   +qrobustness/+kosut/Contents.m and docs/time-bandwidth-bound.md.
%
%   Peer of scripts/run_time_bandwidth_bound_comparison.py; both write the same CSV columns
%   (qrobustness.compat.kosut_csv_headers) so scripts/compare_time_bandwidth_bound.py can
%   cross-check them.
%
%   Name-value options:
%     'max_controllers'  [] = all accepted (61)
%     'FT'               0.999
%     'max_error'        1e-4
%     'eta'              1e-6
%     'literal_theorem'  false; true evaluates their Theorem 1 literally
%                        (F_nom = 1) instead of absorbing eps_0
%     'do_plot'          true
%     'root'             auto-detect repo root
%     'controller_dir'   '' -> data/controllers/problem9_tf15_K32_quasi-newton
%     'publish_dir'      '' -> results/time-bandwidth-bound-matlab/

    % Choose the graphics toolkit before any figure exists.
    qrobustness.compat.setup_graphics();

    p = inputParser;
    addParameter(p, 'max_controllers', []);
    addParameter(p, 'FT', 0.999);
    addParameter(p, 'max_error', 1e-4);
    addParameter(p, 'eta', 1e-6);
    addParameter(p, 'literal_theorem', false);
    addParameter(p, 'do_plot', true);
    addParameter(p, 'root', '');
    addParameter(p, 'controller_dir', '');
    addParameter(p, 'publish_dir', '');
    parse(p, varargin{:});
    opt = p.Results;

    if isempty(opt.root)
        this_dir = fileparts(mfilename('fullpath'));
        opt.root = fileparts(fileparts(this_dir));
    end
    addpath(fullfile(opt.root, 'matlab'));

    if isempty(opt.controller_dir)
        ctrl_dir = fullfile(opt.root, 'data', 'controllers', 'problem9_tf15_K32_quasi-newton');
    else
        ctrl_dir = opt.controller_dir;
    end
    if isempty(opt.publish_dir)
        out_dir = fullfile(opt.root, 'results', 'time-bandwidth-bound-matlab');
    else
        out_dir = opt.publish_dir;
    end
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end

    problem = qrobustness.load_problem(fullfile(ctrl_dir, 'problem9.mat'));
    controllers = qrobustness.load_controllers(fullfile(ctrl_dir, 'controllers.csv'), opt.max_error);
    n = numel(controllers);
    if ~isempty(opt.max_controllers)
        n = min(n, opt.max_controllers);
    end

    tags = {'H0', 'H1', 'H2'};
    headers = qrobustness.compat.kosut_csv_headers();
    Tbl = struct();
    for j = 1:numel(headers)
        Tbl.(headers{j}) = zeros(n, 1);
    end

    for i = 1:n
        c = controllers{i};
        dt = c.tf / c.tau;
        if opt.literal_theorem
            eps0 = 0;
        else
            eps0 = c.error;
        end
        Tbl.controller(i) = i;
        Tbl.fid(i) = c.fid;
        Tbl.err(i) = c.error;
        fprintf('Controller %d/%d fid=%.6g\n', i, n, c.fid);
        for t = 1:numel(tags)
            tag = tags{t};
            switch tag
                case 'H0'
                    C = qrobustness.structure_constant('drift', problem.H0, dt, c.tau);
                case 'H1'
                    C = qrobustness.structure_constant('control', problem.H1, dt, c.tau, c.u1);
                case 'H2'
                    C = qrobustness.structure_constant('control', problem.H2, dt, c.tau, c.u2);
            end
            L = qrobustness.lipschitz_constant(opt.FT, problem.dim, C);
            fid_fn = qrobustness.make_fidelity_fn( ...
                problem.H0, problem.H1, problem.H2, c.u1, c.u2, problem.Uf, dt, tag);
            mres = qrobustness.iterative_margin(fid_fn, L, opt.FT, 'mu0', 0, 'eta', opt.eta);
            M = mres.M;

            H_list = qrobustness.perturbed_hamiltonians( ...
                problem.H0, problem.H1, problem.H2, c.u1, c.u2, tag, 0);
            dH = qrobustness.dH_structure( ...
                problem.H0, problem.H1, problem.H2, c.u1, c.u2, tag);
            rates = qrobustness.kosut.uncertainty_rates(H_list, dH, dt);
            KM = qrobustness.kosut.margin(rates, opt.FT, eps0);

            Tbl.(sprintf('M_%s', tag))(i) = M;
            Tbl.(sprintf('KM_%s', tag))(i) = KM;
            if KM > 0
                Tbl.(sprintf('ratio_%s', tag))(i) = M / KM;
            else
                Tbl.(sprintf('ratio_%s', tag))(i) = Inf;
            end
            % The reference bound evaluated at the certified Lipschitz margin.
            Tbl.(sprintf('KTOb_%s', tag))(i) = qrobustness.kosut.time_bandwidth(rates, M);
            Tbl.(sprintf('Kflb_%s', tag))(i) = qrobustness.kosut.fidelity_bound_at(rates, M);
            % Per-unit-delta uncertainty measures (their Eq. 28).
            Tbl.(sprintf('wunc_%s', tag))(i) = rates.w_unc;
            Tbl.(sprintf('wavg_%s', tag))(i) = rates.w_avg;
            Tbl.(sprintf('wdev_%s', tag))(i) = rates.w_dev;
        end
    end

    csv_path = fullfile(out_dir, sprintf('kosut_comparison_%g.csv', opt.FT));
    qrobustness.compat.write_kosut_csv(csv_path, Tbl);
    fprintf('Wrote %s\n', csv_path);

    if opt.literal_theorem
        mode = 'literal F_nom=1';
    else
        mode = 'eps_0 absorbed';
    end
    fprintf('\nSummary (FT=%g, %s, %d controllers)\n', opt.FT, mode, n);
    fprintf('%6s %12s %12s %14s %18s\n', 'struct', 'median M', 'median M^K', ...
        'median M/M^K', 'max T*Omega_bnd@M');
    for t = 1:numel(tags)
        tag = tags{t};
        fprintf('%6s %12.4e %12.4e %14.2f %18.4e\n', tag, ...
            median(Tbl.(sprintf('M_%s', tag))), ...
            median(Tbl.(sprintf('KM_%s', tag))), ...
            median(Tbl.(sprintf('ratio_%s', tag))), ...
            max(Tbl.(sprintf('KTOb_%s', tag))));
    end
    fprintf('(their bound is vacuous for T*Omega_bnd >= %.4f rad)\n', ...
        qrobustness.kosut.t_omega_max());

    if opt.do_plot
        plot_kosut_scatter(Tbl, tags, opt.FT, ...
            fullfile(out_dir, sprintf('kosut_vs_lipschitz_%g.png', opt.FT)));
    end
end

function plot_kosut_scatter(Tbl, tags, FT, out_path)
%PLOT_KOSUT_SCATTER Log-log scatter of the implied margin against ours.
    colors = [0 0 1; 0 1 0; 1 0 0];
    fig = figure('Visible', 'off');
    hold on;
    lo = Inf; hi = -Inf;
    for t = 1:numel(tags)
        M = Tbl.(sprintf('M_%s', tags{t}));
        K = Tbl.(sprintf('KM_%s', tags{t}));
        scatter(M, K, 14, colors(t, :), 'filled', 'DisplayName', tags{t});
        lo = min([lo; M; K]);
        hi = max([hi; M; K]);
    end
    plot([lo hi], [lo hi], 'k--', 'LineWidth', 0.8, 'DisplayName', 'equality');
    set(gca, 'XScale', 'log', 'YScale', 'log');
    xlabel('Lipschitz margin M (Algorithm 1)');
    ylabel('Kosut et al. implied margin M^K');
    title(sprintf('F_T = %g', FT));
    legend('Location', qrobustness.compat.legend_location());
    grid on;
    qrobustness.apply_plot_style(fig);
    qrobustness.compat.export_figure(fig, out_path);
    close(fig);
    fprintf('Wrote %s\n', out_path);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

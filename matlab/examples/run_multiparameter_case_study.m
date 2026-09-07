% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function run_multiparameter_case_study(varargin)
%RUN_MULTIPARAMETER_CASE_STUDY xQRM joint-parameter margin case study.
%   Peer of scripts/run_multiparameter_case_study.py.  It writes the same
%   CSV schema, with a language-qualified result directory selected by the
%   caller.  The angular stepping rule is the default paper protocol.

    p = inputParser;
    addParameter(p, 'root', '');
    addParameter(p, 'out', '');
    addParameter(p, 'FT', 0.999);
    addParameter(p, 'max_error', 1e-4);
    addParameter(p, 'controllers', 0);
    addParameter(p, 'step', 'angular');
    parse(p, varargin{:});
    opt = p.Results;

    % Bracket refinement, named to match the Python reference's MARGIN_TOL.
    margin_tol = 1e-8;

    if isempty(opt.root)
        here = fileparts(mfilename('fullpath'));
        opt.root = fileparts(fileparts(here));
    end
    addpath(fullfile(opt.root, 'matlab'));
    if isempty(opt.out)
        opt.out = fullfile(opt.root, 'results', 'multiparameter-margin-matlab');
    end
    if ~exist(opt.out, 'dir'); mkdir(opt.out); end

    ctrl = fullfile(opt.root, 'data', 'controllers', ...
        'problem9_tf15_K32_quasi-newton');
    problem = qrobustness.load_problem(fullfile(ctrl, 'problem9.mat'));
    controllers = qrobustness.load_controllers(fullfile(ctrl, 'controllers.csv'), ...
        opt.max_error);
    if opt.controllers > 0
        controllers = controllers(1:min(opt.controllers, numel(controllers)));
    end

    % Three uncertainty parameters: drift and the two controls. Named
    % once here as in the Python peer, which calls it N_PARAMS.
    n_params = 3;
    directions = [qrobustness.multiparam.axis_directions(n_params); ...
                  qrobustness.multiparam.diagonal_directions(n_params)];
    names = {'+e0', '+e1', '+e2', '-e0', '-e1', '-e2', ...
             'diagmmm', 'diagpmm', 'diagmpm', 'diagppm', ...
             'diagmmp', 'diagpmp', 'diagmpp', 'diagppp'};
    headers = {'controller', 'fid', 'err', 'L_H0', 'L_H1', 'L_H2', ...
        'poly_r_H0', 'poly_r_H1', 'poly_r_H2', 'inradius_linf', ...
        'inradius_l2', 'r0_joint', 'r0_H0', 'r0_H1', 'r0_H2'};
    for j = 1:numel(names)
        headers{end + 1} = ['M_' names{j}]; %#ok<AGROW>
        headers{end + 1} = ['Mupper_' names{j}]; %#ok<AGROW>
        headers{end + 1} = ['nev_' names{j}]; %#ok<AGROW>
    end
    values = zeros(numel(controllers), numel(headers));

    for ci = 1:numel(controllers)
        c = controllers{ci};
        dt = c.tf / c.tau;
        specs = {{'drift', problem.H0}, {'control', problem.H1, c.u1}, ...
                 {'control', problem.H2, c.u2}};
        [~, L] = qrobustness.multiparam.structure_constants( ...
            specs, dt, c.tau, opt.FT, problem.dim);
        Hhat_lists = {repmat({problem.H0}, 1, c.tau), ...
            arrayfun(@(u) u * problem.H1, c.u1, 'UniformOutput', false), ...
            arrayfun(@(u) u * problem.H2, c.u2, 'UniformOutput', false)};
        joint = qrobustness.multiparam.joint_gauge(Hhat_lists, dt);
        angular = qrobustness.multiparam.angular_gauge(Hhat_lists, dt);
        fidelity_fn = @(mu) multiparam_fidelity(problem, c, dt, mu);
        F0 = fidelity_fn(zeros(3, 1));
        poly = qrobustness.multiparam.safe_polytope(zeros(3, 1), L, F0, opt.FT);

        row = [ci, c.fid, c.error, L(:).', poly.axis_radii(:).', ...
            poly.inradius_linf, poly.inradius_l2, ...
            qrobustness.timevarying.uniform_margin(L, F0, opt.FT), ...
            qrobustness.timevarying.uniform_margin(L(1), F0, opt.FT), ...
            qrobustness.timevarying.uniform_margin(L(2), F0, opt.FT), ...
            qrobustness.timevarying.uniform_margin(L(3), F0, opt.FT)];
        for di = 1:size(directions, 1)
            d = directions(di, :).';
            kwargs = {'margin_tol', margin_tol, 'return_diagnostics', true, ...
                'L_dir', qrobustness.multiparam.joint_gauge_L_dir( ...
                    joint, d, opt.FT, problem.dim)};
            if strcmpi(opt.step, 'angular')
                kwargs = [kwargs, {'angular_gauge', angular}]; %#ok<AGROW>
            end
            result = qrobustness.multiparam.directional_margin( ...
                fidelity_fn, L, opt.FT, d, kwargs{:});
            row = [row, result.M, result.M_upper, result.n_evals]; %#ok<AGROW>
        end
        values(ci, :) = row;
        fprintf('controller %d/%d inradius_l2=%.3e\n', ...
            ci, numel(controllers), poly.inradius_l2);
    end

    suffix = '';
    if strcmpi(opt.step, 'angular'); suffix = '_angular'; end
    path = fullfile(opt.out, sprintf('multiparam_%g%s.csv', opt.FT, suffix));
    write_csv(path, headers, values);
    fprintf('Wrote %s\n', path);
end

function F = multiparam_fidelity(problem, controller, dt, mu)
    % Fidelity at a joint perturbation mu of the three structures,
    % multiplicative on each as in the main study.
    H = cell(1, controller.tau);
    for k = 1:controller.tau
        H{k} = (1 + mu(1)) * problem.H0 ...
            + (1 + mu(2)) * controller.u1(k) * problem.H1 ...
            + (1 + mu(3)) * controller.u2(k) * problem.H2;
    end
    F = qrobustness.gate_fidelity(qrobustness.propagator(H, dt), problem.Uf);
end

function write_csv(path, headers, values)
    % Write a numeric table with an LF line terminator and 16 significant
    % digits, matching what the Python peer's DictWriter produces.
    fid = fopen(path, 'w');
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '%s\n', strjoin(headers, ','));
    fmt = [repmat('%.16g,', 1, numel(headers) - 1), '%.16g\n'];
    for row = 1:size(values, 1)
        fprintf(fid, fmt, values(row, :));
    end
end

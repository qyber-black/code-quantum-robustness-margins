function run_synthesize_controllers(varargin)
%RUN_SYNTHESIZE_CONTROLLERS Ensemble fidelity maximisation -> results/synth-matlab/.
%
%   Does not modify data/controllers/problem9_tf15_K32_quasi-newton/.
%
%   Name-value:
%     'n_opt'   100
%     'seed'    0
%     'sigma'   1
%     'tf'      15
%     'tau'     32
%     'maxiter' 500
%     'ftol'    1e-12
%     'out'     results/synth-matlab
%     'root'    auto

    p = inputParser;
    addParameter(p, 'n_opt', 100);
    addParameter(p, 'seed', 0);
    addParameter(p, 'sigma', 1.0);
    addParameter(p, 'tf', 15);
    addParameter(p, 'tau', 32);
    addParameter(p, 'maxiter', 500);
    addParameter(p, 'ftol', 1e-12);
    addParameter(p, 'out', '');
    addParameter(p, 'root', '');
    addParameter(p, 'problem_mat', '');
    parse(p, varargin{:});
    opt = p.Results;

    if isempty(opt.root)
        this_dir = fileparts(mfilename('fullpath'));
        opt.root = fileparts(fileparts(this_dir));
    end
    addpath(fullfile(opt.root, 'matlab'));

    paper_ctrl = fullfile(opt.root, 'data', 'controllers', 'problem9_tf15_K32_quasi-newton');
    if isempty(opt.problem_mat)
        opt.problem_mat = fullfile(paper_ctrl, 'problem9.mat');
    end
    if isempty(opt.out)
        opt.out = fullfile(opt.root, 'results', 'synth-matlab');
    end
    if ~exist(opt.out, 'dir'); mkdir(opt.out); end

    dest_mat = fullfile(opt.out, 'problem9.mat');
    copyfile(opt.problem_mat, dest_mat);

    problem = qrobustness.load_problem(dest_mat);
    rows = cell(opt.n_opt, 1);
    errs = zeros(opt.n_opt, 1);

    for i = 1:opt.n_opt
        seed_i = opt.seed + i - 1;
        res = qrobustness.optimize_controller( ...
            problem.H0, problem.H1, problem.H2, problem.Uf, opt.tf, opt.tau, ...
            'sigma', opt.sigma, 'seed', seed_i, ...
            'maxiter', opt.maxiter, 'ftol', opt.ftol);
        rows{i} = res;
        rows{i}.run_id = i;
        rows{i}.tf = opt.tf;
        rows{i}.tau = opt.tau;
        errs(i) = res.error;
        fprintf('[%d/%d] seed=%d fid_init=%.6g fid=%.6g err=%.3e iters=%d\n', ...
            i, opt.n_opt, seed_i, res.fid_init, res.fid, res.error, res.n_iter);
    end

    csv_path = fullfile(opt.out, 'controllers.csv');
    write_controllers_csv(csv_path, rows, opt.tf, opt.tau);

    meta = struct();
    meta.method = 'fminunc-quasi-newton';
    meta.gradient = 'GRAPE';
    meta.seed = opt.seed;
    meta.n_opt = opt.n_opt;
    meta.tf = opt.tf;
    meta.tau = opt.tau;
    meta.sigma = opt.sigma;
    meta.maxiter = opt.maxiter;
    meta.ftol = opt.ftol;
    meta.problem_mat_source = opt.problem_mat;
    meta.analysis_filter = 'eps0 <= 1e-4 (paper; applied by load_controllers, not here)';
    meta.n_accepted_1e_4 = sum(errs <= 1e-4);
    meta.error_min = min(errs);
    meta.error_max = max(errs);
    meta.error_median = median(errs);

    meta_path = fullfile(opt.out, 'meta.json');
    write_meta_json(meta_path, meta);
    fprintf('Wrote %s and %s\n', csv_path, meta_path);
    fprintf('Accepted with eps<=1e-4: %d/%d\n', meta.n_accepted_1e_4, opt.n_opt);
end

function write_controllers_csv(path, rows, tf, tau)
    fid = fopen(path, 'w');
    if fid < 0
        error('Cannot write %s', path);
    end
    for i = 1:numel(rows)
        r = rows{i};
        u = [r.u1(:).'; r.u2(:).'];
        x = u(:);
        fprintf(fid, '9,%d,%.15g,%d,%.16g', r.run_id, tf, tau, r.error);
        for j = 1:numel(x)
            fprintf(fid, ',%.16g', x(j));
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end

function write_meta_json(path, meta)
    fid = fopen(path, 'w');
    fprintf(fid, '{\n');
    fprintf(fid, '  "method": "%s",\n', meta.method);
    fprintf(fid, '  "gradient": "%s",\n', meta.gradient);
    fprintf(fid, '  "seed": %d,\n', meta.seed);
    fprintf(fid, '  "n_opt": %d,\n', meta.n_opt);
    fprintf(fid, '  "tf": %.10g,\n', meta.tf);
    fprintf(fid, '  "tau": %d,\n', meta.tau);
    fprintf(fid, '  "sigma": %.10g,\n', meta.sigma);
    fprintf(fid, '  "maxiter": %d,\n', meta.maxiter);
    fprintf(fid, '  "ftol": %.10g,\n', meta.ftol);
    fprintf(fid, '  "problem_mat_source": "%s",\n', escape_json(meta.problem_mat_source));
    fprintf(fid, '  "analysis_filter": "%s",\n', meta.analysis_filter);
    fprintf(fid, '  "n_accepted_1e-4": %d,\n', meta.n_accepted_1e_4);
    fprintf(fid, '  "error_min": %.16g,\n', meta.error_min);
    fprintf(fid, '  "error_max": %.16g,\n', meta.error_max);
    fprintf(fid, '  "error_median": %.16g\n', meta.error_median);
    fprintf(fid, '}\n');
    fclose(fid);
end

function s = escape_json(s)
    s = strrep(s, '\', '\\');
    s = strrep(s, '"', '\"');
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

function verify_paper_consistency(varargin)
%VERIFY_PAPER_CONSISTENCY Falsifiable checks against a paper results tree.
%
%   verify_paper_consistency()
%   verify_paper_consistency('results_id', 'lipschitz-margin-matlab')
%
% Writes results/<results_id>/verify_paper.md
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root, 'matlab'));

    p = inputParser;
    addParameter(p, 'results_id', 'lipschitz-margin-matlab');
    parse(p, varargin{:});
    results_id = p.Results.results_id;
    results_dir = fullfile(root, 'results', results_id);
    if ~exist(results_dir, 'dir'); mkdir(results_dir); end

    report_md = fullfile(results_dir, 'verify_paper.md');
    report_txt = fullfile(root, 'build', sprintf('verify_paper_%s.txt', results_id));
    if ~exist(fullfile(root, 'build'), 'dir'); mkdir(fullfile(root, 'build')); end

    fid = fopen(report_txt, 'w');
    pass = true;
    FT = 0.999;
    CTRL = fullfile(root, 'data', 'controllers', 'problem9_tf15_K32_quasi-newton');

    logmsg(fid, '=== Paper consistency verification (%s) ===\n', results_id);
    logmsg(fid, 'root=%s\nresults=%s\n\n', root, results_dir);

    %% 1) Case-study inputs
    problem = qrobustness.load_problem(fullfile(CTRL, 'problem9.mat'));
    ctrls = qrobustness.load_controllers(fullfile(CTRL, 'controllers.csv'), 1e-4);
    nC = numel(ctrls);
    logmsg(fid, '[1] Controllers with eps0<=1e-4: %d (paper: 61)\n', nC);
    pass = check(pass, fid, nC == 61);

    tf = ctrls{1}.tf; tau = ctrls{1}.tau; dt = tf / tau;
    logmsg(fid, '[1b] tf=%.6g tau=%d (paper: 15, 32)\n', tf, tau);
    pass = check(pass, fid, abs(tf - 15) < 1e-12 && tau == 32);

    errs = cellfun(@(c) c.error, ctrls);
    logmsg(fid, '[1c] eps0 in [%.3e, %.3e]\n', min(errs), max(errs));
    pass = check(pass, fid, all(errs <= 1e-4 + 1e-15));

    %% 2) Hamiltonians vs paper Heisenberg chain
    N = problem.dim;
    sx = [0 1; 1 0]; sy = [0 -1i; 1i 0]; sz = [1 0; 0 -1];
    op = @(A, ell) kron(kron(eye_pow(ell - 1), A), eye_pow(3 - ell));
    H0_ref = 0.5 * ( ...
        op(sx, 1) * op(sx, 2) + op(sy, 1) * op(sy, 2) + op(sz, 1) * op(sz, 2) + ...
        op(sx, 2) * op(sx, 3) + op(sy, 2) * op(sy, 3) + op(sz, 2) * op(sz, 3));
    H1_ref = 2 * op(sx, 1);
    H2_ref = 2 * op(sy, 1);
    dH = [norm(problem.H0 - H0_ref, 'fro'), ...
          norm(problem.H1 - H1_ref, 'fro'), ...
          norm(problem.H2 - H2_ref, 'fro')];
    logmsg(fid, '[2] ||Hj-paper||_F = [%s], N=%d\n', mat2str(dH, 3), N);
    pass = check(pass, fid, max(dH) < 1e-12 && N == 8);

    %% 3) Nominal fidelity recompute vs CSV
    max_fid_mismatch = 0;
    for k = 1:nC
        c = ctrls{k};
        H_list = qrobustness.perturbed_hamiltonians( ...
            problem.H0, problem.H1, problem.H2, c.u1, c.u2, 'H0', 0);
        F = qrobustness.gate_fidelity(qrobustness.propagator(H_list, dt), problem.Uf);
        max_fid_mismatch = max(max_fid_mismatch, abs((1 - F) - c.error));
    end
    logmsg(fid, '[3] max |eps_recomputed - eps_csv| = %.3e\n', max_fid_mismatch);
    pass = check(pass, fid, max_fid_mismatch < 1e-10);

    %% 4) Margins vs legacy H*_0.999.mat
    csv_path = fullfile(results_dir, 'margins_table_0.999.csv');
    if ~isfile(csv_path)
        % The results id is the make target name, so no string surgery needed.
        logmsg(fid, '[4] MISSING %s (run make %s)\n', csv_path, results_id);
        pass = check(pass, fid, false);
    else
        T = qrobustness.compat.read_margins_csv(csv_path);
        structures = {'H0', 'H1', 'H2'};
        for s = 1:3
            tag = structures{s};
            L = load(fullfile(root, 'data', 'legacy', sprintf('%s_0.999.mat', tag)));
            M_legacy = min(abs(L.margin(:, 1)), abs(L.margin(:, 2)));
            M_new = T.(['M_' tag]);
            Mm = T.(['Mm_' tag]);
            Mp = T.(['Mp_' tag]);
            dM = max(abs(M_new - M_legacy));
            dMinus = max(abs(Mm - abs(L.margin(:, 1))));
            dPlus = max(abs(Mp - abs(L.margin(:, 2))));
            logmsg(fid, '[4] %s max|M-Mleg|=%.3e  |Mm-|abs(leg1)|=%.3e  |Mp-|abs(leg2)|=%.3e\n', ...
                tag, dM, dMinus, dPlus);
            pass = check(pass, fid, dM < 1e-12 && dMinus < 1e-12 && dPlus < 1e-12);
        end
    end

    %% 5) Endpoint fidelity at +/- M (spot check)
    if exist('T', 'var')
        spot = unique([1:5, nC-2:nC]);
        n_bad = 0; max_overshoot = 0; min_slack = inf;
        structures = {'H0', 'H1', 'H2'};
        for ii = 1:numel(spot)
            n = spot(ii);
            c = ctrls{n};
            for s = 1:3
                tag = structures{s};
                M = T.(['M_' tag])(n);
                fid_fn = qrobustness.make_fidelity_fn( ...
                    problem.H0, problem.H1, problem.H2, c.u1, c.u2, problem.Uf, dt, tag);
                for sgn = [-1, 1]
                    F = fid_fn(sgn * M);
                    slack = F - FT;
                    min_slack = min(min_slack, slack);
                    if F < FT - 1e-10
                        n_bad = n_bad + 1;
                        max_overshoot = max(max_overshoot, FT - F);
                    end
                end
            end
        end
        logmsg(fid, '[5] F(+/-M) spot: min(F-FT)=%.3e bad=%d max_overshoot=%.3e\n', ...
            min_slack, n_bad, max_overshoot);
        pass = check(pass, fid, n_bad == 0);
    else
        logmsg(fid, '[5] SKIP (no margins table)\n');
    end

    %% 6) zeta vs central FD + table consistency
    if exist('T', 'var')
        max_rel_fd = 0;
        max_table_dz = 0;
        structures = {'H0', 'H1', 'H2'};
        for n = 1:5
            c = ctrls{n};
            for s = 1:3
                tag = structures{s};
                H_list = qrobustness.perturbed_hamiltonians( ...
                    problem.H0, problem.H1, problem.H2, c.u1, c.u2, tag, 0);
                dH = qrobustness.dH_structure( ...
                    problem.H0, problem.H1, problem.H2, c.u1, c.u2, tag);
                z = qrobustness.differential_sensitivity(H_list, dH, dt, problem.Uf, 32);
                fid_fn = qrobustness.make_fidelity_fn( ...
                    problem.H0, problem.H1, problem.H2, c.u1, c.u2, problem.Uf, dt, tag);
                h = 1e-7;
                z_fd = (fid_fn(h) - fid_fn(-h)) / (2 * h);
                max_rel_fd = max(max_rel_fd, abs(z - z_fd) / max(1e-12, abs(z_fd)));
                max_table_dz = max(max_table_dz, abs(T.(['zeta_' tag])(n) - z));
            end
        end
        logmsg(fid, '[6] max rel|zeta-FD|(h=1e-7)=%.3e  max|table-recompute|=%.3e\n', max_rel_fd, max_table_dz);
        % FD vs analytic \zeta is a soft spot-check (engine-dependent); table recompute is hard.
        if max_rel_fd >= 2e-4
            logmsg(fid, '  NOTE soft FD check exceeded 2e-4 (not failing)\n');
        end
        pass = check(pass, fid, max_table_dz < 1e-10);
    else
        logmsg(fid, '[6] SKIP\n');
    end

    %% 7) Table I in main.tex matches results correlations
    corr_tex = fullfile(results_dir, 'correlations_0.999.tex');
    if ~isfile(corr_tex)
        logmsg(fid, '[7] MISSING %s\n', corr_tex);
        pass = check(pass, fid, false);
    else
        C_code = parse_corr_tex(corr_tex);
        C_paper = parse_corr_from_main(fullfile(fileparts(root), 'main.tex'));
        dC = max(abs(C_code(:) - C_paper(:)));
        logmsg(fid, '[7] max |main.tex TableI - %s/correlations| = %.3e\n', results_id, dC);
        pass = check(pass, fid, dC < 1e-12);
    end

    %% 8) correlations match recomputed from margins table
    if exist('T', 'var') && exist('C_code', 'var')
        vars = {'err', 'M_H0', 'M_H1', 'M_H2', 'zeta_H0', 'zeta_H1', 'zeta_H2'};
        X = qrobustness.compat.margins_matrix(T, vars);
        [P, S] = qrobustness.compat.correlation_matrices(X);
        C_re = eye(7);
        for i = 1:7
            for j = 1:7
                if j > i
                    C_re(i, j) = P(i, j);
                elseif j < i
                    C_re(i, j) = S(i, j);
                end
            end
        end
        dCre = max(max(abs(qrobustness.compat.round_decimals(C_re, 2) - C_code)));
        logmsg(fid, '[8] max |round(recomputed,2) - correlations.tex| = %.3e\n', dCre);
        pass = check(pass, fid, dCre == 0);
    else
        logmsg(fid, '[8] SKIP\n');
    end

    %% 9) L / C_H formulas
    c = ctrls{1};
    C0 = qrobustness.structure_constant('drift', problem.H0, dt, c.tau);
    C1 = qrobustness.structure_constant('control', problem.H1, dt, c.tau, c.u1);
    B_T = sqrt((1 - FT^2) / N);
    L0 = qrobustness.lipschitz_constant(FT, N, C0);
    logmsg(fid, '[9] B_T=%.6g C0=%.6g L0=%.6g; C0_ref=tf*||H0||F=%.6g\n', ...
        B_T, C0, L0, tf * norm(problem.H0, 'fro'));
    pass = check(pass, fid, abs(C0 - tf * norm(problem.H0, 'fro')) < 1e-12);
    pass = check(pass, fid, abs(C1 - dt * norm(c.u1(:), 1) * norm(problem.H1, 'fro')) < 1e-12);

    %% 10) results deliverables present
    need = {'H0_all.png', 'H1_all.png', 'H2_all.png', ...
        'robustness_margins_fid_err.png', 'robustness_margins_sensitivity.png', ...
        'correlations_0.999.tex', 'margins_table_0.999.csv'};
    ok_pub = true;
    for i = 1:numel(need)
        f = fullfile(results_dir, need{i});
        if ~isfile(f)
            logmsg(fid, '[10] MISSING %s\n', f);
            ok_pub = false;
        end
    end
    logmsg(fid, '[10] %s deliverables\n', results_id);
    pass = check(pass, fid, ok_pub);

    %% 11) Algorithm 1 spot-check
    if exist('T', 'var')
        n = 1; c = ctrls{n};
        C = qrobustness.structure_constant('drift', problem.H0, dt, c.tau);
        L = qrobustness.lipschitz_constant(FT, N, C);
        fid_fn = qrobustness.make_fidelity_fn( ...
            problem.H0, problem.H1, problem.H2, c.u1, c.u2, problem.Uf, dt, 'H0');
        res = qrobustness.iterative_margin(fid_fn, L, FT, 'mu0', 0, 'eta', 1e-6);
        logmsg(fid, '[11] ctrl1 H0: M-=%.6g M+=%.6g M=%.6g tableM=%.6g conv=[%d %d]\n', ...
            res.M_minus, res.M_plus, res.M, T.M_H0(1), res.converged_minus, res.converged_plus);
        pass = check(pass, fid, abs(res.M - T.M_H0(1)) < 1e-12);
        pass = check(pass, fid, res.M == min(res.M_minus, res.M_plus));
        pass = check(pass, fid, res.mu_minus < 0 && res.mu_plus > 0);
    else
        logmsg(fid, '[11] SKIP\n');
    end

    logmsg(fid, '\n=== VERDICT: %s ===\n', tern(pass, 'PASS', 'FAIL'));
    fclose(fid);

    txt = fileread(report_txt);
    md = sprintf(['# Paper consistency verification (%s)\n\n' ...
        'Generated by `scripts/verify_paper_consistency.m`.\n\n' ...
        '```\n%s```\n'], results_id, txt);
    fidm = fopen(report_md, 'w');
    fwrite(fidm, md);
    fclose(fidm);
    fprintf('Wrote %s and %s\n', report_txt, report_md);
    if ~pass
        error('verify_paper_consistency:Fail', 'Checks failed; see %s', report_md);
    end
end

function logmsg(fid, varargin)
    fprintf(fid, varargin{:});
    fprintf(varargin{:});
end

function pass = check(pass, fid, cond)
    if cond
        fprintf(fid, '  PASS\n');
        fprintf('  PASS\n');
    else
        fprintf(fid, '  FAIL\n');
        fprintf('  FAIL\n');
        pass = false;
    end
end

function E = eye_pow(n)
    if n <= 0
        E = 1;
    else
        E = eye(2^n);
    end
end

function C = parse_corr_tex(path)
    txt = fileread(path);
    lines = qrobustness.compat.split_lines(txt);
    rows = {};
    for i = 1:numel(lines)
        nums = regexp(lines{i}, '\$([+-]?\d+\.\d+)\$', 'tokens');
        if numel(nums) >= 7
            rows{end+1} = cellfun(@(t) str2double(t{1}), nums(1:7)); %#ok<AGROW>
        end
    end
    C = vertcat(rows{:});
    if ~isequal(size(C), [7, 7])
        error('parse_corr_tex: size %s', mat2str(size(C)));
    end
end

function C = parse_corr_from_main(path)
    tex = fileread(path);
    idx = strfind(tex, '\label{tab:correlations}');
    chunk = tex(idx(1):min(numel(tex), idx(1) + 3000));
    lines = qrobustness.compat.split_lines(chunk);
    rows = {};
    for i = 1:numel(lines)
        nums = regexp(lines{i}, '\$([+-]?\d+\.\d+)\$', 'tokens');
        if numel(nums) >= 7
            rows{end+1} = cellfun(@(t) str2double(t{1}), nums(1:7)); %#ok<AGROW>
        end
        if numel(rows) >= 7
            break;
        end
    end
    C = vertcat(rows{:});
    if ~isequal(size(C), [7, 7])
        error('parse_corr_from_main: size %s', mat2str(size(C)));
    end
end

function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

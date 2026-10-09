% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function run_open_system_case_study(varargin)
%RUN_OPEN_SYSTEM_CASE_STUDY Open-system margins for common-rate dephasing; writes open_margins_<FT>.csv.
%
%   Name-value options:
%     'root'        - repository root (default: detected)
%     'out'         - output directory (default results/lindblad-margin-matlab)
%     'FT'          - fidelity threshold F_T (default 0.999)
%     'max_error'   - nominal error filter (default 1e-4)
%     'controllers' - number of controllers (default 0 = all)
%
%   Peer of scripts/run_open_system_case_study.py.

    p = inputParser;
    addParameter(p, 'root', '');
    addParameter(p, 'out', '');
    addParameter(p, 'FT', 0.999);
    addParameter(p, 'max_error', 1e-4);
    addParameter(p, 'controllers', 0);
    parse(p, varargin{:});
    opt = p.Results;

    % Bracket refinement; same value as MARGIN_TOL in the Python open-system driver.
    margin_tol = 1e-6;
    if isempty(opt.root)
        opt.root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    end
    addpath(fullfile(opt.root, 'matlab'));
    if isempty(opt.out)
        opt.out = fullfile(opt.root, 'results', 'lindblad-margin-matlab');
    end
    if ~exist(opt.out, 'dir'); mkdir(opt.out); end

    ctrl_dir = fullfile(opt.root, 'data', 'controllers', ...
        'problem9_tf15_K32_quasi-newton');
    problem = qrobustness.load_problem(fullfile(ctrl_dir, 'problem9.mat'));
    controllers = qrobustness.load_controllers( ...
        fullfile(ctrl_dir, 'controllers.csv'), opt.max_error);
    if opt.controllers > 0
        controllers = controllers(1:min(opt.controllers, numel(controllers)));
    end
    Ggamma = dephasing_generator(problem.n_qubits);
    dn = qrobustness.lindblad.diamond_norm(Ggamma);
    headers = {'controller', 'fid', 'err', 'F_pro_0', 'L_gamma', ...
        'M_gamma', 'M_gamma_upper', 'r0_gamma', 'gamma_star', ...
        'conservatism_M', 'conservatism_r0', 'n_evals'};
    rows = zeros(numel(controllers), numel(headers));

    for ci = 1:numel(controllers)
        c = controllers{ci};
        dt = c.tf / c.tau;
        % Per controller: tf is a per-row field.
        Lgamma = qrobustness.lindblad.rate_lipschitz(dn.value, c.tf);
        H = cell(1, c.tau);
        for k = 1:c.tau
            H{k} = problem.H0 + c.u1(k) * problem.H1 + c.u2(k) * problem.H2;
        end
        GH = cellfun(@qrobustness.lindblad.hamiltonian_superop, H, ...
            'UniformOutput', false);
        Fpro = @(gamma) fidelity_at(GH, Ggamma, gamma, dt, problem.Uf);
        F0 = Fpro(0);
        ft_pro = opt.FT^2;
        result = qrobustness.lindblad.open_margin(Fpro, Lgamma, ft_pro, ...
            'margin_tol', margin_tol, 'return_diagnostics', true);
        M = result.M_plus;
        gamma_star = crossing(Fpro, ft_pro, M);
        rows(ci, :) = [ci, c.fid, c.error, F0, Lgamma, M, ...
            result.M_upper_plus, (F0 - ft_pro) / Lgamma, gamma_star, ...
            gamma_star / M, gamma_star / ((F0 - ft_pro) / Lgamma), ...
            result.n_evals];
        fprintf('controller %d/%d M_gamma=%.3e gamma*=%.3e\n', ...
            ci, numel(controllers), M, gamma_star);
    end
    path = fullfile(opt.out, sprintf('open_margins_%g.csv', opt.FT));
    write_csv(path, headers, rows);
    fprintf('Wrote %s\n', path);
end

function F = fidelity_at(GH, Ggamma, gamma, dt, Uf)
    % Process fidelity at one dissipation rate.
    G = cellfun(@(base) base + gamma * Ggamma, GH, 'UniformOutput', false);
    F = qrobustness.lindblad.process_fidelity( ...
        qrobustness.lindblad.channel(G, dt), Uf);
end

function value = crossing(fn, threshold, lo)
    % Bisect the threshold crossing above a certified margin; returns the lower (safe) end of the bracket.
    % Peer of scripts/_drivers.true_crossing.
    hi = max(10 * lo, 1e-6);
    while fn(hi) >= threshold && hi < 1e3
        lo = hi;
        hi = 10 * hi;
    end
    for iteration = 1:60
        mid = 0.5 * (lo + hi);
        if fn(mid) >= threshold; lo = mid; else; hi = mid; end
        if (hi - lo) / hi < 1e-10; break; end
    end
    value = lo;
end

function G = dephasing_generator(n)
    % Sum of the single-qubit dephasing dissipators acting on n qubits.
    sz = [1, 0; 0, -1];
    G = 0;
    for q = 1:n
        V = 1;
        for j = 1:n
            V = kron(V, ternary(j == q, sz, eye(2)));
        end
        G = G + qrobustness.lindblad.dissipator(V);
    end
end

function output = ternary(condition, yes, no)
    % Inline conditional, which the language does not supply.
    if condition; output = yes; else; output = no; end
end

function write_csv(path, headers, rows)
    % Write a numeric table using the shared CSV format (LF, 16 digits).
    fid = fopen(path, 'w');
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '%s\n', strjoin(headers, ','));
    fmt = [repmat('%.16g,', 1, numel(headers) - 1), '%.16g\n'];
    for row = 1:size(rows, 1); fprintf(fid, fmt, rows(row, :)); end
end

function test_consistency_matlab()
%TEST_CONSISTENCY_MATLAB Compare live MATLAB values to Python's committed table.
%
%   The reference is results/lipschitz-margin-python/margins_table_<FT>.csv,
%   not a recorded fixture. A fixture asserts that some previous answer was
%   correct, which it never established; two independent implementations
%   agreeing is evidence, and if they move together the committed tree shows
%   it in git. make test-parity does the full 61-controller comparison; this
%   is the subset that runs inside the MATLAB suite.

    this_dir = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(this_dir));
    addpath(fullfile(root, 'matlab'));

    FT = 0.999;
    eta = 1e-6;
    rtol = 1e-8;
    atol = 1e-10;
    % Smallest nominal error, an asymmetric-margin case, the largest
    % nominal error: the anchors the drivers use.
    indices = [1 59 61];
    structures = {'H0', 'H1', 'H2'};

    table_path = fullfile(root, 'results', 'lipschitz-margin-python', ...
        sprintf('margins_table_%g.csv', FT));
    if ~isfile(table_path)
        % A skip, not a failure: make maintainer-clean removes every
        % generated result on purpose, and running the tests before
        % rebuilding them is a legitimate order.
        fprintf('SKIP  test_consistency_matlab (no %s; run make paper-QRM-margins)\n', ...
            'results/lipschitz-margin-python/margins_table_0.999.csv');
        return
    end
    ref = read_margins_table(table_path);

    CTRL = fullfile(root, 'data', 'controllers', 'problem9_tf15_K32_quasi-newton');
    problem = qrobustness.load_problem(fullfile(CTRL, 'problem9.mat'));
    controllers = qrobustness.load_controllers(fullfile(CTRL, 'controllers.csv'), 1e-4);

    n = 0;
    for idx = indices
        k = find(ref.controller == idx);
        assert(numel(k) == 1, 'controller %d absent from the peer table', idx);
        c = controllers{idx};
        dt = c.tf / c.tau;
        for s = 1:numel(structures)
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
                'mu0', 0, 'eta', eta, 'margin_tol', 1e-8);
            H_list = qrobustness.perturbed_hamiltonians( ...
                problem.H0, problem.H1, problem.H2, c.u1, c.u2, tag, 0);
            dH = qrobustness.dH_structure(problem.H0, problem.H1, problem.H2, ...
                c.u1, c.u2, tag);
            zeta = qrobustness.differential_sensitivity(H_list, dH, dt, problem.Uf, 32);

            assert_close(fid_fn(0), ref.fid(k), rtol, atol, sprintf('%s[%d] fid', tag, idx));
            assert_close(margin.M, ref.(['M_' tag])(k), rtol, atol, ...
                sprintf('%s[%d] M', tag, idx));
            assert_close(margin.M_minus, ref.(['Mm_' tag])(k), rtol, atol, ...
                sprintf('%s[%d] M-', tag, idx));
            assert_close(margin.M_plus, ref.(['Mp_' tag])(k), rtol, atol, ...
                sprintf('%s[%d] M+', tag, idx));
            assert_close(zeta, ref.(['zeta_' tag])(k), rtol, atol, ...
                sprintf('%s[%d] zeta', tag, idx));
            assert(logical(margin.converged_minus), '%s[%d] converged_minus', tag, idx);
            assert(logical(margin.converged_plus), '%s[%d] converged_plus', tag, idx);
            n = n + 1;
        end
    end
    assert(n == numel(indices) * numel(structures), 'expected %d records, got %d', ...
        numel(indices) * numel(structures), n);
    fprintf('PASS  test_consistency_matlab (%d records)\n', n);
end

function t = read_margins_table(path)
%READ_MARGINS_TABLE Numeric CSV with a header, as a struct of columns.
%   A struct rather than a table: `table` lives in an Octave Forge package
%   and this suite runs on core Octave.
    fid = fopen(path, 'r');
    header = strsplit(strtrim(fgetl(fid)), ',');
    data = textscan(fid, repmat('%f', 1, numel(header)), 'Delimiter', ',');
    fclose(fid);
    t = struct();
    for i = 1:numel(header)
        t.(strtrim(header{i})) = data{i};
    end
end

function assert_close(a, b, rtol, atol, name)
    % Mixed relative/absolute comparison, naming the field on failure.
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

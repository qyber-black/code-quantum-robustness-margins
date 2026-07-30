function test_compat_csv()
%TEST_COMPAT_CSV Round-trip margins CSV and controller load via compat helpers.

    this_dir = fileparts(mfilename('fullpath'));
    root_dir = fileparts(fileparts(this_dir));
    addpath(fullfile(root_dir, 'matlab'));

    CTRL = fullfile(root_dir, 'data', 'controllers', 'problem9_tf15_K32_quasi-newton');
    csv_path = fullfile(CTRL, 'controllers.csv');
    controllers = qrobustness.load_controllers(csv_path, 1e-4);
    assert(numel(controllers) == 61);

    missing = fullfile(root_dir, 'build', 'does_not_exist_controllers.csv');
    threw = false;
    try
        qrobustness.compat.read_numeric_csv(missing);
    catch
        threw = true;
    end
    assert(threw, 'expected missing-file error');

    nC = 3;
    R.H0.fid = [0.999; 0.998; 0.997];
    R.H0.error = 1 - R.H0.fid;
    R.H0.M = [0.01; 0.02; 0.03];
    R.H0.M_minus = R.H0.M;
    R.H0.M_plus = R.H0.M + 1e-6;
    R.H0.zeta = [1e-4; -2e-4; 3e-4];
    for tag = {'H1', 'H2'}
        t = tag{1};
        R.(t).M = R.H0.M * 1.1;
        R.(t).M_minus = R.(t).M;
        R.(t).M_plus = R.(t).M + 2e-6;
        R.(t).zeta = -R.H0.zeta;
    end

    T = qrobustness.compat.build_margins_table(R, nC);
    out_dir = fullfile(root_dir, 'build');
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end
    out_csv = fullfile(out_dir, 'test_compat_margins.csv');
    qrobustness.compat.write_margins_csv(out_csv, T);
    T2 = qrobustness.compat.read_margins_csv(out_csv);

    headers = qrobustness.compat.margins_csv_headers();
    for j = 1:numel(headers)
        h = headers{j};
        assert(max(abs(T.(h) - T2.(h))) < 1e-12, 'mismatch on %s', h);
    end

    vars = {'err', 'M_H0', 'M_H1', 'M_H2', 'zeta_H0', 'zeta_H1', 'zeta_H2'};
    X = qrobustness.compat.margins_matrix(T2, vars);
    [P, S] = qrobustness.compat.correlation_matrices(X);
    assert(isequal(size(P), [7, 7]) && isequal(size(S), [7, 7]));
    assert(max(abs(diag(P) - 1)) < 1e-12);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

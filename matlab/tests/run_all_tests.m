function results = run_all_tests()
%RUN_ALL_TESTS Execute MATLAB unit tests for +qrobustness.

    this_dir = fileparts(mfilename('fullpath'));
    root_dir = fileparts(fileparts(this_dir));
    addpath(fullfile(root_dir, 'matlab'));

    tests = {
        @test_propagator_fidelity
        @test_lipschitz_structure
        @test_perfect_fidelity_zeta
        @test_iterative_margin_synthetic
        @test_margin_solvers_selectable
        @test_threshold_error
        @test_load_case_study_smoke
        @test_compat_csv
        @test_compat_graphics
        @test_iterative_margin_case_study
        @test_optimize_controller
        @test_dU_dmu_exact
        @test_error_control
        @test_kosut_bound
    };

    n = numel(tests);
    passed = 0;
    for k = 1:n
        name = func2str(tests{k});
        try
            tests{k}();
            fprintf('PASS  %s\n', name);
            passed = passed + 1;
        catch ME
            fprintf('FAIL  %s\n  %s\n', name, ME.message);
            rethrow(ME);
        end
    end
    fprintf('%d / %d tests passed\n', passed, n);
    results = struct('passed', passed, 'total', n);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

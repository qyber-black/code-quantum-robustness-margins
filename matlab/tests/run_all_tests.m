function results = run_all_tests()
%RUN_ALL_TESTS Execute MATLAB unit tests for +qrobustness.
%   Every test runs, whatever the ones before it did: stopping at the first
%   failure hides how many others would also have failed, which is the
%   number you need when deciding whether a change broke one thing or
%   everything. The failures are reported together at the end and then
%   raised, so the exit code still fails the build.
%
%   Returns a struct with fields passed, failed and total.

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
        @test_traceless_and_status
        @test_lengthspace
        @test_lindblad
        @test_multiparam
        @test_timevarying
    };

    n = numel(tests);
    passed = 0;
    failures = {};
    for k = 1:n
        name = func2str(tests{k});
        try
            tests{k}();
            fprintf('PASS  %s\n', name);
            passed = passed + 1;
        catch ME
            fprintf('FAIL  %s\n  %s\n', name, ME.message);
            failures{end + 1} = sprintf('%s: %s', name, ME.message); %#ok<AGROW>
        end
    end

    failed = numel(failures);
    fprintf('\n%d passed, %d failed of %d tests\n', passed, failed, n);
    if failed > 0
        fprintf('\nFailed:\n');
        for k = 1:failed
            fprintf('  %s\n', failures{k});
        end
    end
    results = struct('passed', passed, 'failed', failed, 'total', n);
    if failed > 0
        error('qrobustness:test:Failures', '%d of %d tests failed', failed, n);
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

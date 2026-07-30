function test_error_control()
% Error control: every approximated quantity must carry a usable certificate.
%
% Two quantities in the package are not exact to roundoff:
%   kosut.uncertainty_rates -> w_dev, a supremum recovered from samples
%   iterative_margin        -> M, terminated on a FIDELITY band eta
% Both must be conservative in a stated direction and reach a requested
% precision when asked.  Mirrors python/tests/test_error_control.py.

    this_dir = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(this_dir));
    CTRL = fullfile(root, 'data', 'controllers', 'problem9_tf15_K32_quasi-newton');
    problem = qrobustness.load_problem(fullfile(CTRL, 'problem9.mat'));
    controllers = qrobustness.load_controllers(fullfile(CTRL, 'controllers.csv'));
    FT = 0.999;
    structures = {'H0', 'H1', 'H2'};

    % ---- Kosut uncertainty rates -------------------------------------
    for ci = 1:4
        c = controllers{ci};
        dt = c.tf / c.tau;
        H_list = cell(1, c.tau);
        for k = 1:c.tau
            H_list{k} = problem.H0 + c.u1(k) * problem.H1 + c.u2(k) * problem.H2;
        end
        for si = 1:numel(structures)
            tag = structures{si};
            dH = qrobustness.dH_structure(problem.H0, problem.H1, problem.H2, ...
                c.u1, c.u2, tag);
            r = qrobustness.kosut.uncertainty_rates(H_list, dH, dt);

            % Both rigorous certificates must hold.
            assert(r.w_dev >= r.w_dev_bracket_lo - 1e-12 && ...
                   r.w_dev <= r.w_dev_bracket_hi + 1e-12, ...
                '%s ctrl%d: w_dev outside isospectral bracket', tag, ci);
            assert(r.w_dev <= r.w_dev_certified + 1e-12, ...
                '%s ctrl%d: w_dev exceeds Lipschitz certificate', tag, ci);
            assert(r.w_dev_bracket_lo >= 0, 'bracket_lo must be non-negative');

            % Sampling can only under-estimate a supremum.
            rs = qrobustness.kosut.uncertainty_rates(H_list, dH, dt, [], 17, [], [], false);
            assert(rs.w_dev <= r.w_dev + 1e-12, ...
                '%s ctrl%d: sampled sup exceeded polished sup', tag, ci);

            if ci == 1
                % n_quad is inert: <Htil> is closed form.
                r4 = qrobustness.kosut.uncertainty_rates(H_list, dH, dt, 4);
                r512 = qrobustness.kosut.uncertainty_rates(H_list, dH, dt, 512);
                assert(r4.w_avg == r512.w_avg, 'n_quad changed w_avg');
                assert(r.dev_converged, '%s: w_dev refinement did not converge', tag);
                % The seed grid must follow the Bohr bandwidth, not a constant.
                assert(r.dev_resolved, '%s: bandwidth not resolved', tag);
                assert(r.dev_cycles_max > 0, '%s: bad bandwidth', tag);
                assert(r.dev_samples_per_cycle >= 16, ...
                    '%s: only %g samples/cycle', tag, r.dev_samples_per_cycle);
            end
        end
    end

    % Under-estimating w_dev biases the comparison in favour of the bound: pin the direction.
    c = controllers{1};
    dt = c.tf / c.tau;
    H_list = cell(1, c.tau);
    for k = 1:c.tau
        H_list{k} = problem.H0 + c.u1(k) * problem.H1 + c.u2(k) * problem.H2;
    end
    dH = qrobustness.dH_structure(problem.H0, problem.H1, problem.H2, c.u1, c.u2, 'H1');
    r_pol = qrobustness.kosut.uncertainty_rates(H_list, dH, dt);
    r_smp = qrobustness.kosut.uncertainty_rates(H_list, dH, dt, [], 17, [], [], false);
    assert(r_smp.w_dev < r_pol.w_dev, 'expected sampling to fall short');
    m_smp = qrobustness.kosut.margin(r_smp, FT, c.error);
    m_pol = qrobustness.kosut.margin(r_pol, FT, c.error);
    assert(m_smp > m_pol, 'under-estimated w_dev should inflate the margin');

    % A longer interval means more Bohr cycles, so the grid must grow with it.
    % This is the case a fixed grid gets wrong.
    dt_long = 20 * c.tf / c.tau;
    derived = qrobustness.kosut.uncertainty_rates(H_list, dH, dt_long);
    fixed = qrobustness.kosut.uncertainty_rates(H_list, dH, dt_long, [], 17, [], [], false);
    reference = qrobustness.kosut.uncertainty_rates(H_list, dH, dt_long, [], 40001, [], 40001, false);
    assert(derived.dev_cycles_max > 10, 'expected a stress case');
    assert(derived.dev_resolved, 'bandwidth not resolved on the long interval');
    assert(derived.n_dev_used > 17, 'grid did not grow with the bandwidth');
    assert(derived.w_dev >= reference.w_dev - 1e-12, ...
        'polished value fell below a sampled reference');
    err_derived = abs(derived.w_dev - reference.w_dev) / reference.w_dev;
    err_fixed = abs(fixed.w_dev - reference.w_dev) / reference.w_dev;
    assert(err_derived < err_fixed / 100, ...
        'bandwidth-derived grid (%g) not much better than fixed (%g)', err_derived, err_fixed);

    % ---- Margin ------------------------------------------------------
    for ci = 1:3
        c = controllers{ci};
        dt = c.tf / c.tau;
        for si = 1:numel(structures)
            tag = structures{si};
            switch tag
                case 'H0'
                    C = qrobustness.structure_constant('drift', problem.H0, dt, c.tau);
                case 'H1'
                    C = qrobustness.structure_constant('control', problem.H1, dt, c.tau, c.u1);
                otherwise
                    C = qrobustness.structure_constant('control', problem.H2, dt, c.tau, c.u2);
            end
            L = qrobustness.lipschitz_constant(FT, problem.dim, C);
            fn = qrobustness.make_fidelity_fn(problem.H0, problem.H1, problem.H2, ...
                c.u1, c.u2, problem.Uf, dt, tag);

            base = qrobustness.iterative_margin(fn, L, FT);
            for tol = [1e-6, 1e-10]
                r = qrobustness.iterative_margin(fn, L, FT, 'margin_tol', tol);
                assert(strcmp(r.reason_minus, 'bracketed') && ...
                       strcmp(r.reason_plus, 'bracketed'), ...
                    '%s ctrl%d: expected a valid bracket', tag, ci);
                assert(r.margin_uncertainty / r.M <= tol * 1.000001, ...
                    '%s ctrl%d: margin_tol=%g not achieved (got %g)', ...
                    tag, ci, tol, r.margin_uncertainty / r.M);
                % Refinement only tightens.
                assert(r.M >= base.M - 1e-15, '%s ctrl%d: margin_tol loosened M', tag, ci);
            end

            % The reported bracket must be a valid bracket.
            r = qrobustness.iterative_margin(fn, L, FT, 'margin_tol', 1e-10);
            assert(fn(r.M) >= FT && fn(-r.M) >= FT, ...
                '%s ctrl%d: F < FT at the reported margin', tag, ci);
            assert(min(fn(r.M_upper_plus), fn(-r.M_upper_minus)) < FT, ...
                '%s ctrl%d: F >= FT at M_upper, not a bracket', tag, ci);
        end
    end

    % The default eta=1e-6 leaves a material relative error in M.
    c = controllers{1};
    dt = c.tf / c.tau;
    C = qrobustness.structure_constant('control', problem.H1, dt, c.tau, c.u1);
    L = qrobustness.lipschitz_constant(FT, problem.dim, C);
    fn = qrobustness.make_fidelity_fn(problem.H0, problem.H1, problem.H2, ...
        c.u1, c.u2, problem.Uf, dt, 'H1');
    r_coarse = qrobustness.iterative_margin(fn, L, FT);
    r_fine = qrobustness.iterative_margin(fn, L, FT, 'margin_tol', 1e-12);
    coarse = r_coarse.M;
    fine = r_fine.M;
    rel = abs(fine - coarse) / fine;
    assert(rel > 1e-5 && rel < 1e-2, ...
        'expected ~5e-4 relative eta-induced error in M, got %g', rel);

    % Certificate class must be reported, not inferred from the method string.
    meths = {'algorithm1', 'lipschitz_brent', 'lipschitz_toms748', 'doubling'};
    expect = {'segment', 'segment', 'segment', 'endpoint'};
    C = qrobustness.structure_constant('drift', problem.H0, dt, c.tau);
    L = qrobustness.lipschitz_constant(FT, problem.dim, C);
    fn = qrobustness.make_fidelity_fn(problem.H0, problem.H1, problem.H2, ...
        c.u1, c.u2, problem.Uf, dt, 'H0');
    for mi = 1:numel(meths)
        r = qrobustness.iterative_margin(fn, L, FT, 'method', meths{mi});
        assert(strcmp(r.certificate, expect{mi}), ...
            '%s: expected certificate %s, got %s', meths{mi}, expect{mi}, r.certificate);
    end

    % margin_tol must be positive.
    ok = false;
    try
        qrobustness.iterative_margin(fn, L, FT, 'margin_tol', 0);
    catch
        ok = true;
    end
    assert(ok, 'expected margin_tol=0 to be rejected');
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

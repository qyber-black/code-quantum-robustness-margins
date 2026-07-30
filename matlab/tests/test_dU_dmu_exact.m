function test_dU_dmu_exact()
% Exact closed-form segment derivative vs finite differences and quadrature.
%
% For piecewise-constant controls dU/dmu is exact in the eigenbasis of the
% interval Hamiltonian, so the Gauss-Legendre path only approximates what
% qrobustness.dU_dmu_exact computes in closed form.  Pin the closed form down
% directly, then assert the two paths agree on the real case-study data.

    dt = 0.4688;
    rng(20260730);

    % Cases: generic, exactly degenerate spectrum, zero, and scalar.
    [Q, ~] = qr(randn(4) + 1i * randn(4));
    Hdeg = Q * diag([1 1 1 2]) * Q';
    cases = struct( ...
        'generic',    {rand_herm(8)}, ...
        'degenerate', {(Hdeg + Hdeg') / 2}, ...
        'zero',       {zeros(4)}, ...
        'scalar',     {3 * eye(5)});
    names = fieldnames(cases);

    for i = 1:numel(names)
        H = cases.(names{i});
        n = size(H, 1);
        dH = rand_herm(n);
        [V, lam] = qrobustness.segment_eig(H);
        dU = qrobustness.dU_dmu_exact(V, lam, dH, dt);

        assert(all(isfinite(dU(:))), '%s: dU has non-finite entries', names{i});
        assert(norm(V' * V - eye(n)) < 1e-13, '%s: eigenvectors not unitary', names{i});
        assert(norm(qrobustness.segment_propagator(V, lam, dt) - expm(-1i * dt * H)) < 1e-12, ...
            '%s: eig-based propagator disagrees with expm', names{i});

        % Central difference in mu, at the roundoff floor of eps = 1e-6.
        ep = 1e-6;
        fd = (expm(-1i * dt * (H + ep * dH)) - expm(-1i * dt * (H - ep * dH))) / (2 * ep);
        rel = norm(dU - fd, 'fro') / norm(dU, 'fro');
        assert(rel < 1e-8, '%s: dU vs central difference rel err %g', names{i}, rel);
    end

    % dH = H commutes, so dU/dmu = -1i*dt*H*expm(-1i*dt*H) analytically.
    H = rand_herm(6);
    [V, lam] = qrobustness.segment_eig(H);
    dU = qrobustness.dU_dmu_exact(V, lam, H, dt);
    ref = -1i * dt * H * expm(-1i * dt * H);
    rel = norm(dU - ref, 'fro') / norm(ref, 'fro');
    assert(rel < 1e-13, 'Commuting case rel err %g', rel);

    % Unknown method is rejected.
    ok = false;
    try
        qrobustness.parse_dU_options('method', 'simpson');
    catch
        ok = true;
    end
    assert(ok, 'Expected parse_dU_options to reject an unknown method');

    % Exact vs quadrature on the real case-study data.
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    ctrl = fullfile(root, 'data', 'controllers', 'problem9_tf15_K32_quasi-newton');
    problem = qrobustness.load_problem(fullfile(ctrl, 'problem9.mat'));
    controllers = qrobustness.load_controllers(fullfile(ctrl, 'controllers.csv'));
    controllers = controllers(1:min(6, numel(controllers)));

    structures = {'H0', 'H1', 'H2'};
    for si = 1:numel(structures)
        for ci = 1:numel(controllers)
            c = controllers{ci};
            dtc = c.tf / c.tau;
            H_list = cell(1, c.tau);
            for k = 1:c.tau
                H_list{k} = problem.H0 + c.u1(k) * problem.H1 + c.u2(k) * problem.H2;
            end
            dH_list = qrobustness.dH_structure(problem.H0, problem.H1, problem.H2, ...
                c.u1, c.u2, structures{si});
            z_exact = qrobustness.differential_sensitivity(H_list, dH_list, dtc, problem.Uf);
            z_quad = qrobustness.differential_sensitivity(H_list, dH_list, dtc, problem.Uf, ...
                'method', 'quadrature', 'n_quad', 48);
            rel = abs(z_exact - z_quad) / max(abs(z_exact), 1e-12);
            assert(rel < 1e-7, '%s controller %d: zeta exact vs quadrature rel err %g', ...
                structures{si}, ci, rel);
        end
    end

    % Gradients agree, and F matches the quadrature path to roundoff.
    c = controllers{1};
    dtc = c.tf / c.tau;
    [Fe, g1e, g2e] = qrobustness.fidelity_and_gradient(problem.H0, problem.H1, problem.H2, ...
        c.u1, c.u2, problem.Uf, dtc);
    [Fq, g1q, g2q] = qrobustness.fidelity_and_gradient(problem.H0, problem.H1, problem.H2, ...
        c.u1, c.u2, problem.Uf, dtc, 'method', 'quadrature', 'n_quad', 48);
    assert(abs(Fe - Fq) < 1e-12, 'F exact vs quadrature diff %g', abs(Fe - Fq));
    assert(max(abs(g1e - g1q)) / max(abs(g1e)) < 1e-7, 'g1 exact vs quadrature mismatch');
    assert(max(abs(g2e - g2q)) / max(abs(g2e)) < 1e-7, 'g2 exact vs quadrature mismatch');

    % A positional n_quad must not select quadrature.
    H_list = cell(1, c.tau);
    for k = 1:c.tau
        H_list{k} = problem.H0 + c.u1(k) * problem.H1 + c.u2(k) * problem.H2;
    end
    dH_list = qrobustness.dH_structure(problem.H0, problem.H1, problem.H2, c.u1, c.u2, 'H1');
    z4 = qrobustness.differential_sensitivity(H_list, dH_list, dtc, problem.Uf, 4);
    z32 = qrobustness.differential_sensitivity(H_list, dH_list, dtc, problem.Uf, 32);
    assert(z4 == z32, 'Positional n_quad changed the exact result');
    z4q = qrobustness.differential_sensitivity(H_list, dH_list, dtc, problem.Uf, ...
        'method', 'quadrature', 'n_quad', 4);
    assert(abs(z4q - z32) > 1e-12 * abs(z32), ...
        '4-node quadrature should differ from exact; positional test is vacuous');
end

function H = rand_herm(n)
    A = randn(n) + 1i * randn(n);
    H = (A + A') / 2;
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

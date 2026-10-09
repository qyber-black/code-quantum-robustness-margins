% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function test_openstates()
%TEST_OPENSTATES Open-system state margins: closed limit of the speed and parity with the Python fixture.
%   The fixture is written by scripts/gen_states_parity.py.

    % Closed limit: ||-1i[B, .]||_diamond = lambda_max - lambda_min, bracketed without assuming it: the equal superposition of the
    % extreme eigenvectors is an explicit input (lower bound), and the solver-free value is an independent upper bound.
    B = [0.3, 0.1 - 0.2i, 0; 0.1 + 0.2i, -0.5, 0.05; 0, 0.05, 0.2];
    S = qrobustness.lindblad.hamiltonian_superop(B);
    [V, Dg] = eig(B);
    [w, order] = sort(real(diag(Dg)));
    V = V(:, order);
    spread = w(end) - w(1);
    psi = (V(:, 1) + V(:, end)) / sqrt(2);
    out = reshape(S * reshape(psi * psi', [], 1), 3, 3);
    witness = sum(abs(eig((out + out') / 2)));
    assert(abs(witness / spread - 1) < 1e-12);
    r = qrobustness.lindblad.hamiltonian_dnorm(S);
    assert(r.value_certified >= spread && r.value_certified <= spread * (1 + 1e-10));
    assert(r.value_certified <= getfield(qrobustness.lindblad.diamond_norm(S), 'value_certified')); %#ok<GFLD>
    C = qrobustness.states.state_speed({B, B}, 0.4);
    D = qrobustness.openstates.open_speed({S, S}, 0.4, [], true);
    assert(D / 2 >= C && abs(D / 2 / C - 1) < 1e-10);
    assert(isempty(qrobustness.lindblad.hamiltonian_part(qrobustness.lindblad.dissipator(diag([1, -1, 0])))));
    assert_error(@() qrobustness.lindblad.hamiltonian_dnorm(qrobustness.lindblad.dissipator(diag([1, -1, 0]))), 'qrobustness:lindblad:notHamiltonian', ...
                 'not Hamiltonian');

    f = states_fixture();
    o = f.openstates;
    SZ = diag([1, -1, 0]);
    LOW = [0, 1, 0; 0, 0, 1; 0, 0, 0];
    rho0 = f.psi0 * f.psi0';
    gen = @(sv, mu) cellfun(@(h, a) qrobustness.lindblad.hamiltonian_superop(h + sv * a) + (0.3 + mu) * qrobustness.lindblad.dissipator(SZ) ...
                            + 0.1 * qrobustness.lindblad.dissipator(LOW), f.H, f.A, 'UniformOutput', false);
    rho = @(sv, mu) qrobustness.openstates.evolve_density(gen(sv, mu), f.dt, rho0);
    rel = @(a, b) abs(a - b) <= 1e-9 * max(abs(b), 1e-12) + 1e-12;
    D = qrobustness.openstates.open_speed(repmat({qrobustness.lindblad.dissipator(SZ)}, 1, numel(f.H)), f.dt, @(G) o.dnorm);
    assert(rel(D, o.D));
    assert(rel(qrobustness.openstates.trace_distance(rho(0, 0), rho(0.5, 0.2)), o.trace_distance));
    assert(rel(getfield(qrobustness.lindblad.hamiltonian_dnorm(qrobustness.lindblad.hamiltonian_superop(f.B{1})), 'value_certified'), ...
               o.ham_dnorm)); %#ok<GFLD>
    supers = cellfun(@qrobustness.lindblad.hamiltonian_superop, f.B, 'UniformOutput', false);
    assert(rel(qrobustness.openstates.open_speed(supers, f.dt, [], true), o.D_exact));
end

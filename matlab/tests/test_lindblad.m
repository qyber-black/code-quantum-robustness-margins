% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function test_lindblad()
%TEST_LINDBLAD Open-system layer: superoperators and the diamond norm.
%   The diamond norm is checked against the analytic value 2n of the
%   paper's lem:2n, which is the strongest check available: it does not
%   depend on the Python implementation being right.

    sz = [1 0; 0 -1];
    sm = [0 1; 0 0];
    I2 = eye(2);
    L = @qrobustness.lindblad;

    % Hamiltonian superoperator annihilates the identity: [H, I] = 0.
    H = [0.3 0.7-0.2i; 0.7+0.2i -0.5];
    Sh = qrobustness.lindblad.hamiltonian_superop(H);
    assert(norm(Sh * reshape(I2, [], 1)) < 1e-12);

    % A dissipator is trace annihilating: Tr D[V](rho) = 0 for all rho.
    D = qrobustness.lindblad.dissipator(sm);
    for t = 1:4
        R = randn(2) + 1i * randn(2);
        out = reshape(D * R(:), 2, 2);
        assert(abs(trace(out)) < 1e-12);
    end

    % The unitary superoperator has process fidelity 1 against itself.
    U = expm(-1i * H);
    Su = qrobustness.lindblad.unitary_superop(U);
    assert(abs(qrobustness.lindblad.process_fidelity(Su, U) - 1) < 1e-12);
    assert(abs(qrobustness.lindblad.average_gate_fidelity(Su, U) - 1) < 1e-12);

    % Choi of the identity channel is the unnormalised maximally
    % entangled projector, of trace N.
    Sid = qrobustness.lindblad.unitary_superop(I2);
    Jid = qrobustness.lindblad.choi_matrix(Sid);
    assert(abs(trace(Jid) - 2) < 1e-12);
    assert(norm(Jid - Jid', 'fro') < 1e-12);

    % Diamond norm: sum_q D[sigma_z^(q)] has diamond norm exactly 2n
    % (paper lem:2n). Same for sigma_- under that normalisation.
    for n = 1:3
        tot = [];
        for q = 1:n
            V = 1;
            for m = 1:n
                if m == q
                    V = kron(V, sz);
                else
                    V = kron(V, I2);
                end
            end
            Dq = qrobustness.lindblad.dissipator(V);
            if isempty(tot)
                tot = Dq;
            else
                tot = tot + Dq;
            end
        end
        r = qrobustness.lindblad.diamond_norm(tot);
        % An upper bound must not fall below the true value.
        assert(r.value >= 2 * n - 1e-9);
        assert(abs(r.value - 2 * n) < 1e-6);
        assert(r.feas_shift >= 0);
        assert(strcmp(r.status, 'solver_free'));
    end

    % Amplitude damping at n = 1: also exactly 2.
    ramp = qrobustness.lindblad.diamond_norm( ...
        qrobustness.lindblad.dissipator(sm));
    assert(ramp.value >= 2 - 1e-9);
    assert(abs(ramp.value - 2) < 1e-6);

    % --- channel: ordered product of interval propagators --------------
    % A single interval reduces to expm(dt*G).
    G = qrobustness.lindblad.dissipator(sm);
    S1 = qrobustness.lindblad.channel({G}, 0.3);
    assert(norm(S1 - expm(0.3 * G), 'fro') < 1e-12);
    % Two intervals compose in time order: later acts on the left.
    Gb = qrobustness.lindblad.hamiltonian_superop(H);
    S2 = qrobustness.lindblad.channel({G, Gb}, 0.25);
    assert(norm(S2 - expm(0.25 * Gb) * expm(0.25 * G), 'fro') < 1e-12);
    % A zero generator gives the identity channel.
    Z = zeros(size(G));
    assert(norm(qrobustness.lindblad.channel({Z, Z}, 0.5) - eye(size(G, 1)), ...
                'fro') < 1e-12);
    threw = false;
    try
        qrobustness.lindblad.channel({}, 0.1);
    catch
        threw = true;
    end
    assert(threw);   % an empty interval list has no meaning

    % --- open_margin: the one-sided rate domain ------------------------
    % The wrapper exists to impose omega = [0, Inf): a decoherence rate
    % below zero is not a physical certificate, and the generic iteration
    % would happily step there.
    FT = 0.999;
    fid = @(g) 1 - (1 - FT) * (g / 0.02)^2;
    r = qrobustness.lindblad.open_margin(fid, 50, FT, 'eta', 1e-10);
    assert(r.mu_minus >= -1e-12);
    assert(r.M_plus > 0);
    % An explicit omega must still win over the default.
    r2 = qrobustness.lindblad.open_margin(fid, 50, FT, 'eta', 1e-10, ...
                                          'omega', [-Inf, Inf]);
    assert(r2.mu_minus < 0);

    % Homogeneity: scaling the map scales the norm.
    r1 = qrobustness.lindblad.diamond_norm(D);
    r2 = qrobustness.lindblad.diamond_norm(3.0 * D);
    assert(abs(r2.value - 3 * r1.value) < 1e-6 * max(1, r1.value));
end

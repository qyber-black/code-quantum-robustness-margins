function test_propagator_fidelity()
    % Identity evolution: H=0 => U=I, fidelity 1 against I
    N = 2;
    H_list = {zeros(N), zeros(N)};
    U = qrobustness.propagator(H_list, 0.1);
    assert(norm(U - eye(N), 'fro') < 1e-12);
    F = qrobustness.gate_fidelity(U, eye(N));
    assert(abs(F - 1) < 1e-12);

    % Pauli-Z rotation matches analytic
    Hz = [1 0; 0 -1] / 2;
    dt = 0.3;
    U = qrobustness.propagator({Hz}, dt);
    Uref = expm(-1i * dt * Hz);
    assert(norm(U - Uref, 'fro') < 1e-12);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

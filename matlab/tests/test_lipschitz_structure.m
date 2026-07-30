function test_lipschitz_structure()
    H0 = [0 1; 1 0];
    H1 = [1 0; 0 -1];
    dt = 0.25;
    tau = 4;
    tf = tau * dt;
    controls = [0.5, -0.2, 0.1, 0.3];

    C0 = qrobustness.structure_constant('drift', H0, dt, tau);
    assert(abs(C0 - tf * norm(H0, 'fro')) < 1e-14);

    C1 = qrobustness.structure_constant('control', H1, dt, tau, controls);
    assert(abs(C1 - dt * norm(controls, 1) * norm(H1, 'fro')) < 1e-14);

    FT = 0.999;
    N = 2;
    L = qrobustness.lipschitz_constant(FT, N, C0);
    B = sqrt((1 - FT^2) / N);
    assert(abs(L - B * C0) < 1e-14);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

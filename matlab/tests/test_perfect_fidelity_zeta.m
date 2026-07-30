function test_perfect_fidelity_zeta()
    % When U = Uf, fidelity is 1 and zeta must vanish (Corollary).
    N = 2;
    H = [0 1; 1 0];
    dt = 0.2;
    Uf = expm(-1i * dt * H);
    H_list = {H};
    dH_list = {H};  % any Hermitian structure
    zeta = qrobustness.differential_sensitivity(H_list, dH_list, dt, Uf, 48);
    assert(abs(zeta) < 1e-9, 'Expected vanishing sensitivity at F=1, got %g', zeta);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

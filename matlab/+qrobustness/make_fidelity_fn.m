function fn = make_fidelity_fn(H0, H1, H2, u1, u2, Uf, dt, structure)
%MAKE_FIDELITY_FN Function handle F = fn(delta) for a structure.

    fn = @(delta) local_fid(delta, H0, H1, H2, u1, u2, Uf, dt, structure);
end

function F = local_fid(delta, H0, H1, H2, u1, u2, Uf, dt, structure)
    H_list = qrobustness.perturbed_hamiltonians(H0, H1, H2, u1, u2, structure, delta);
    U = qrobustness.propagator(H_list, dt);
    F = qrobustness.gate_fidelity(U, Uf);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

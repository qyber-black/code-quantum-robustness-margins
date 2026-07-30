function F = fidelity_bound(T_omega_bnd)
%FIDELITY_BOUND F_lb of their Eq. 30 given T*Omega_bnd.
%   F_lb = max(1 - 0.5*(exp((T*Omega_bnd/2)^2) - 1)^2, 0), clamped to 0 at and
%   beyond qrobustness.kosut.t_omega_max().

    if any(T_omega_bnd < 0)
        error('qrobustness:kosut:NegTOb', 'T_omega_bnd must be non-negative.');
    end
    ymax = qrobustness.kosut.t_omega_max();
    F = max(1 - 0.5 * (exp((T_omega_bnd / 2).^2) - 1).^2, 0);
    F(T_omega_bnd >= ymax) = 0;
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

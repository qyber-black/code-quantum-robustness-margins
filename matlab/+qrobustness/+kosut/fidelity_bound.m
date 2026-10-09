function F = fidelity_bound(T_omega_bnd)
%FIDELITY_BOUND Lower bound F_lb = max(1 - (exp((T*Omega_bnd/2)^2) - 1)^2/2, 0).
%   T_omega_bnd - T*Omega_bnd, non-negative; F_lb = 0 at and beyond
%                 qrobustness.kosut.t_omega_max()
%
%   Peer of python/src/qrobustness/kosut.py:fidelity_bound.

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

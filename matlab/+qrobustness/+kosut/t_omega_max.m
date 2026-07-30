function y = t_omega_max()
%T_OMEGA_MAX Largest T*Omega_bnd with F_lb > 0 (their Eq. 32), in radians.
%   y = 2*sqrt(log(1+sqrt(2))) ~= 1.8776.  Beyond this their bound is vacuous.

    y = 2 * sqrt(log(1 + sqrt(2)));
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

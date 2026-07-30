function y = time_bandwidth(rates, delta)
%TIME_BANDWIDTH T*Omega_bnd of their Eq. 29 at perturbation delta.
%   Uses linearity of their Eq. 28 in delta:
%     T*Omega_bnd(delta) = sqrt(a*delta^2 + b*|delta|),
%     a = T^2*w_unc*w_dev,  b = 4*T*w_avg.
%
%   See also QROBUSTNESS.KOSUT.UNCERTAINTY_RATES.

    d = abs(delta);
    a = rates.T^2 * rates.w_unc * rates.w_dev;
    b = 4 * rates.T * rates.w_avg;
    y = sqrt(a .* d .* d + b .* d);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

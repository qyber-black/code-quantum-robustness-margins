function y = threshold_time_bandwidth(FT, nominal_error)
%THRESHOLD_TIME_BANDWIDTH T*Omega_bnd at which their F_lb equals the threshold.
%   Inverts their Eq. 30 in closed form:
%     F_lb = F_eff  <=>  T*Omega_bnd = 2*sqrt(log(1 + sqrt(2*(1 - F_eff))))
%   with F_eff = FT + nominal_error.  The nominal_error term absorbs the
%   nominal fidelity deficit, since their Theorem 1 assumes F_nom = 1
%   (default 0, i.e. their theorem taken literally).
%
%   Returns 0 when F_eff >= 1, i.e. no perturbation is certifiable.

    if nargin < 2 || isempty(nominal_error); nominal_error = 0; end
    if ~(FT > 0 && FT < 1)
        error('qrobustness:kosut:BadFT', 'FT must satisfy 0 < FT < 1.');
    end
    if nominal_error < 0
        error('qrobustness:kosut:BadEps', 'nominal_error must be non-negative.');
    end
    eps_t = 1 - FT - nominal_error;
    if eps_t <= 0
        y = 0;
        return;
    end
    y = 2 * sqrt(log(1 + sqrt(2 * eps_t)));
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

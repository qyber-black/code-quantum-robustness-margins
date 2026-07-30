function M = margin(rates, FT, nominal_error)
%MARGIN Perturbation margin implied by their Theorem 1.
%   M = MARGIN(rates, FT) is the largest |delta| for which their F_lb >= FT,
%   i.e. the analogue of qrobustness.iterative_margin(...).M obtained from
%   their bound.  M = MARGIN(rates, FT, nominal_error) instead certifies
%   F_lb >= FT + nominal_error (their Theorem 1 assumes F_nom = 1).
%
%   Since T*Omega_bnd is monotone in |delta|, this inverts
%   a*delta^2 + b*delta = y^2 in closed form, with
%   y = qrobustness.kosut.threshold_time_bandwidth(FT, nominal_error).
%
%   Returns 0 if no positive perturbation is certifiable, and Inf if the
%   perturbation does not enter the bound at all (w_unc*w_dev = 0, w_avg = 0).

    if nargin < 3 || isempty(nominal_error); nominal_error = 0; end
    y = qrobustness.kosut.threshold_time_bandwidth(FT, nominal_error);
    if y <= 0
        M = 0;
        return;
    end
    a = rates.T^2 * rates.w_unc * rates.w_dev;
    b = 4 * rates.T * rates.w_avg;
    y2 = y * y;
    if a <= 0 && b <= 0
        M = Inf;
    elseif a <= 0
        M = y2 / b;
    else
        M = (-b + sqrt(b * b + 4 * a * y2)) / (2 * a);
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

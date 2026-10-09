function M = margin(rates, FT, nominal_error, absorption, uncertainty)
%MARGIN Margin M^K implied by the Kosut-Lidar-Rabitz bound.
%   M = qrobustness.kosut.margin(rates, FT, nominal_error, absorption, uncertainty)
%
%   rates         - from qrobustness.kosut.uncertainty_rates
%   FT            - fidelity threshold F_T
%   nominal_error - nominal error, absorbed via effective_threshold (default 0)
%   absorption    - 'angular' (default) or 'additive'
%   uncertainty   - 'constant' (default; M^K, constant |delta| <= M only) or
%                   'trajectory' (M^K_tv)
%   M             - largest |delta| with F_lb >= the effective threshold; 0 if
%                   none, Inf if the perturbation does not enter the bound
%
%   Peer of python/src/qrobustness/kosut.py:margin.

    if nargin < 3 || isempty(nominal_error); nominal_error = 0; end
    if nargin < 4 || isempty(absorption); absorption = 'angular'; end
    if nargin < 5 || isempty(uncertainty); uncertainty = 'constant'; end
    y = qrobustness.kosut.threshold_time_bandwidth(FT, nominal_error, absorption);
    if y <= 0
        M = 0;
        return
    end
    [w_avg, w_dev] = qrobustness.kosut.select_rates(rates, uncertainty);
    a = rates.T^2 * rates.w_unc * w_dev;
    b = 4 * rates.T * w_avg;
    y2 = y * y;
    if a <= 0 && b <= 0
        M = Inf;
    else
        % Positive root of a*m^2 + b*m = y2 in rationalised form, stable when a*y2 << b^2.
        M = 2 * y2 / (b + sqrt(b * b + 4 * a * y2));
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

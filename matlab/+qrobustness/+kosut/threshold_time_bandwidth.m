function y = threshold_time_bandwidth(FT, nominal_error, absorption)
%THRESHOLD_TIME_BANDWIDTH T*Omega_bnd at which F_lb equals the effective threshold.
%   FT            - fidelity threshold F_T
%   nominal_error - nominal error (default 0)
%   absorption    - 'angular' (default) or 'additive'; see effective_threshold
%   y             - 2*sqrt(log(1 + sqrt(2*(1 - F_eff)))); 0 when F_eff >= 1
%
%   Peer of python/src/qrobustness/kosut.py:threshold_time_bandwidth.

    if nargin < 2 || isempty(nominal_error); nominal_error = 0; end
    if nargin < 3 || isempty(absorption); absorption = 'angular'; end
    F_eff = qrobustness.kosut.effective_threshold(FT, nominal_error, absorption);
    eps_t = 1 - F_eff;
    if eps_t <= 0
        y = 0;
        return
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

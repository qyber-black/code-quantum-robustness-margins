function F_eff = effective_threshold(FT, nominal_error, absorption)
%EFFECTIVE_THRESHOLD Achieved-gate fidelity threshold implied by F_T on the target.
%   FT            - fidelity threshold F_T on the target gate
%   nominal_error - nominal error eps_0 = 1 - F(mu0), 0 <= eps_0 <= 1 (default 0)
%   absorption    - 'angular' (default): cos(arccos F_T - arccos(1 - eps_0)),
%                   or 1 if the nominal angle exhausts the budget;
%                   'additive': F_T + eps_0, which is not sufficient
%   F_eff         - threshold on the fidelity to the achieved nominal gate
%
%   Peer of python/src/qrobustness/kosut.py:effective_threshold.

    if nargin < 2 || isempty(nominal_error); nominal_error = 0; end
    if nargin < 3 || isempty(absorption); absorption = 'angular'; end
    if ~(FT > 0 && FT < 1)
        error('qrobustness:kosut:BadFT', 'FT must satisfy 0 < FT < 1.');
    end
    if ~(nominal_error >= 0 && nominal_error <= 1)
        % Reject eps_0 > 1 rather than clamp it.
        error('qrobustness:kosut:BadEps', ...
              'nominal_error must satisfy 0 <= nominal_error <= 1.');
    end
    switch absorption
        case 'additive'
            F_eff = min(FT + nominal_error, 1);
        case 'angular'
            theta_T = acos(FT);
            theta_nom = acos(1 - nominal_error);
            if theta_nom >= theta_T
                F_eff = 1;
            else
                F_eff = cos(theta_T - theta_nom);
            end
        otherwise
            error('qrobustness:kosut:BadAbsorption', ...
                  'absorption must be ''angular'' or ''additive''.');
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

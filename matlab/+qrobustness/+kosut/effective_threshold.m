function F_eff = effective_threshold(FT, nominal_error, absorption)
%EFFECTIVE_THRESHOLD Achieved-gate fidelity threshold implied by FT on the target.
%   Theorem 1 of the reference bounds the fidelity to the ACHIEVED nominal
%   gate, |Tr(U_S' * U)|/N, whereas the certificate is stated against the
%   TARGET.  Since arccos of the gate fidelity is the angle between the
%   corresponding Choi states, it satisfies the triangle inequality, and the
%   sufficient condition on the achieved-gate fidelity is
%
%     F_achieved >= cos( arccos(FT) - arccos(1 - nominal_error) )
%
%   (absorption = 'angular', the default).  When the nominal angle exhausts
%   the budget, arccos(1-eps0) >= arccos(FT), no perturbation is certifiable
%   and 1 is returned so the margin is zero.
%
%   absorption = 'additive' returns FT + nominal_error.  It is NOT sufficient
%   for the target-gate threshold (it is looser than the angular value
%   whenever nominal_error > 0) and is retained only to reproduce previously
%   published numbers.

    if nargin < 2 || isempty(nominal_error); nominal_error = 0; end
    if nargin < 3 || isempty(absorption); absorption = 'angular'; end
    if ~(FT > 0 && FT < 1)
        error('qrobustness:kosut:BadFT', 'FT must satisfy 0 < FT < 1.');
    end
    if ~(nominal_error >= 0 && nominal_error <= 1)
        % 1 - eps_0 is a fidelity, so eps_0 > 1 is not a physical input;
        % reject it rather than clamp it silently.
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

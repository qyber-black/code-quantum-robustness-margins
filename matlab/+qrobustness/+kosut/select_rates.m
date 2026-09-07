% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function [w_avg, w_dev] = select_rates(rates, uncertainty)
%SELECT_RATES (w_avg, w_dev) for the requested uncertainty class.
%   'constant' scales the measures of the structure delta*Hhat with a fixed
%   delta -- valid for constant (fixed-direction) perturbations only.
%   'trajectory' substitutes the certified worst-case bounds over all
%   measurable trajectories |delta(t)| <= |delta|
%   (w_avg_traj = mean_k ||Hhat^(k)||, w_dev_traj = w_unc + w_avg_traj): a
%   sign-modulated trajectory can defeat the coherent averaging behind the
%   small w_avg, so the constant-delta margin is NOT a supremum-norm
%   time-varying margin (adversarial counterexamples exist), whereas the
%   trajectory variant is.
%
%   Peer of python/src/qrobustness/kosut.py:_select_rates.

    if nargin < 2 || isempty(uncertainty); uncertainty = 'constant'; end
    uncertainty = lower(char(uncertainty));
    switch uncertainty
        case 'constant'
            w_avg = rates.w_avg;
            w_dev = rates.w_dev;
        case 'trajectory'
            if ~isfield(rates, 'w_avg_traj') || ~isfield(rates, 'w_dev_traj') ...
                    || ~isfinite(rates.w_avg_traj) || ~isfinite(rates.w_dev_traj)
                error('qrobustness:kosut:selectRates', ...
                    'trajectory rates unavailable; recompute uncertainty_rates()');
            end
            w_avg = rates.w_avg_traj;
            w_dev = rates.w_dev_traj;
        otherwise
            error('qrobustness:kosut:selectRates', ...
                'uncertainty must be ''constant'' or ''trajectory''');
    end
end

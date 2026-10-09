% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function [w_avg, w_dev] = select_rates(rates, uncertainty)
%SELECT_RATES (w_avg, w_dev) for an uncertainty class.
%   rates       - from qrobustness.kosut.uncertainty_rates
%   uncertainty - 'constant' (default): w_avg, w_dev, valid for constant delta
%                 only; 'trajectory': w_avg_traj, w_dev_traj, valid for all
%                 trajectories |delta(t)| <= |delta|
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

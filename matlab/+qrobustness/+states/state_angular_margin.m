% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function result = state_angular_margin(fidelity_fn, C, FT, varargin)
%STATE_ANGULAR_MARGIN Margin of a state fidelity F(mu) = |<chi|psi(mu)>|.
%   fidelity_fn - handle mu -> F
%   C           - state speed C^st, > 0
%   FT          - fidelity threshold F_T
%   result      - as qrobustness.iterative_margin
%
%   Name-value options:
%     'angular' - true (default): safe radius (arccos F_T - arccos F)/C;
%                 false: Lipschitz radius (F - F_T)/(sqrt(1 - F_T^2) C)
%   Other options are passed to qrobustness.iterative_margin.
%
%   Peer of python/src/qrobustness/states.py:state_angular_margin.

    if C <= 0
        error('qrobustness:states:speed', 'Speed constant C must be positive');
    end
    angular = true;
    rest = {};
    for i = 1:2:numel(varargin)
        if strcmpi(varargin{i}, 'angular')
            angular = varargin{i + 1};
        else
            rest(end + 1:end + 2) = varargin(i:i + 1); %#ok<AGROW>
        end
    end
    L = qrobustness.states.state_lipschitz_constant(FT, C);
    if angular
        thetaT = acos(FT);
        rest(end + 1:end + 2) = {'safe_radius_fn', @(F) max(0, (thetaT - acos(min(F, 1))) / C)};
    end
    result = qrobustness.iterative_margin(fidelity_fn, L, FT, rest{:});
end

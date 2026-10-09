% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function t = trajectory_radius(budget, speed)
%TRAJECTORY_RADIUS Radius budget/speed for time-varying perturbations.
%   budget - angle budget, > 0 (arccos F_T - arccos F0 for a state fidelity)
%   speed  - speed bound (Inf radius if 0)
%   t      - struct with fields r, budget, speed
%
%   Peer of python/src/qrobustness/states.py:trajectory_radius.

    if budget <= 0
        error('qrobustness:states:budget', 'The budget must be positive (nominal value beyond the threshold)');
    end
    if speed > 0
        r = budget / speed;
    else
        r = Inf;
    end
    t = struct('r', r, 'budget', budget, 'speed', speed);
end

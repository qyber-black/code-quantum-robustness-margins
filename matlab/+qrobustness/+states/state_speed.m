% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function C = state_speed(Hhat_list, dt)
%STATE_SPEED State speed C^st = dt sum_k ||\hat H^(k)||_c (per unit parameter).
%   Hhat_list - cell array of interval structures \hat H^(k)
%   dt        - interval length
%
%   Peer of python/src/qrobustness/states.py:state_speed.

    C = 0;
    for k = 1:numel(Hhat_list)
        C = C + qrobustness.states.half_spread(Hhat_list{k});
    end
    C = dt * C;
end

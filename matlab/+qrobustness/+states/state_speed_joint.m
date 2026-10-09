% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function c = state_speed_joint(Hhat_lists, dt, x)
%STATE_SPEED_JOINT Joint state gauge dt sum_k ||sum_j x_j \hat H_j^(k)||_c.
%   Hhat_lists - cell{p}{tau} of structure matrices
%   dt         - interval length
%   x          - parameter direction
%   c          - at most sum_j |x_j| state_speed(\hat H_j)
%
%   Peer of python/src/qrobustness/states.py:state_speed_joint.

    if numel(Hhat_lists) ~= numel(x)
        error('qrobustness:states:direction', 'Need one direction component per structure');
    end
    c = 0;
    for k = 1:numel(Hhat_lists{1})
        G = zeros(size(Hhat_lists{1}{k}));
        for j = 1:numel(x)
            G = G + x(j) * Hhat_lists{j}{k};
        end
        c = c + qrobustness.states.half_spread(G);
    end
    c = dt * c;
end

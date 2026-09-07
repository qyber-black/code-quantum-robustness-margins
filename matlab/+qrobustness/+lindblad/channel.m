% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function S = channel(G_list, dt)
%CHANNEL Ordered product of piecewise-constant Lindblad propagators.
%   Peer of python/src/qrobustness/lindblad.py:channel.

    if isempty(G_list)
        error('qrobustness:lindblad:channel', 'G_list must not be empty.');
    end
    n = size(G_list{1}, 1);
    S = eye(n);
    for k = 1:numel(G_list)
        S = expm(dt * G_list{k}) * S;
    end
end

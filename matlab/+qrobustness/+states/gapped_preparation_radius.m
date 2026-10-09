% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function r = gapped_preparation_radius(budget, gap, prep_spread, evolution_speed)
%GAPPED_PREPARATION_RADIUS Angular radius for a gapped, parameter-dependent preparation.
%   budget          - angle budget
%   gap             - spectral gap, > 0
%   prep_spread     - s = ||dH||_c of the preparation Hamiltonian derivative
%   evolution_speed - evolution speed C^st (default 0)
%   r               - largest r with r*(s/(gap - 2 s r) + C^st) <= budget
%
%   Peer of python/src/qrobustness/states.py:gapped_preparation_radius.

    if nargin < 4 || isempty(evolution_speed)
        evolution_speed = 0;
    end
    if ~(gap > 0)
        error('qrobustness:states:gap', 'The gap must be positive');
    end
    if budget <= 0
        r = 0;
        return
    end
    s = prep_spread;
    C = evolution_speed;
    if s < 0 || C < 0
        error('qrobustness:states:speed', 'Speeds must be non-negative');
    end
    if s == 0 && C == 0
        r = Inf;
        return
    end
    b = s + C * gap + 2 * s * budget;
    disc = max(b * b - 8 * s * C * budget * gap, 0);
    r = 2 * budget * gap / (b + sqrt(disc));
end

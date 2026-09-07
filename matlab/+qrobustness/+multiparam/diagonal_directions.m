% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function D = diagonal_directions(p)
%DIAGONAL_DIRECTIONS All 2^p diagonal directions, Euclidean-normalised.
%
%   Peer of python/src/qrobustness/multiparam.py:diagonal_directions.

    % Row order must match Python's itertools.product((-1, 1), repeat=p),
    % which varies the LAST coordinate fastest: mmm, mmp, mpm, ... Any
    % positional comparison of the two implementations -- a CSV column
    % order, a parity check -- mis-pairs the directions otherwise, while
    % every individual row still looks correct.
    nv = 2^p;
    D = zeros(nv, p);
    for v = 0:(nv - 1)
        for j = 1:p
            if bitand(bitshift(v, -(p - j)), 1)
                D(v + 1, j) = 1;
            else
                D(v + 1, j) = -1;
            end
        end
    end
    D = D / sqrt(p);
end

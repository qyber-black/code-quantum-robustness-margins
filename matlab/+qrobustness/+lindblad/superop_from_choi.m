% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function S = superop_from_choi(J, N)
%SUPEROP_FROM_CHOI Inverse of choi_matrix, again pure reindexing.
%
%   S = qrobustness.lindblad.superop_from_choi(J) recovers the
%   superoperator whose Choi matrix is J. Every entry is moved, none is
%   computed, so choi_matrix followed by this returns the stored bits of
%   the original and not merely a close value. That round trip is what
%   the verified diamond-norm bound relies on when it claims to speak
%   about the represented superoperator.
%
%   Peer of python/src/qrobustness/lindblad.py:superop_from_choi.

    if nargin < 2
        N = round(sqrt(size(J, 1)));
    end
    S = reshape(permute(reshape(J, [N N N N]), [2 4 1 3]), [N * N, N * N]);
end

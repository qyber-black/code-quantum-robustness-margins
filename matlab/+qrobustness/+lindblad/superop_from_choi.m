% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function S = superop_from_choi(J, N)
%SUPEROP_FROM_CHOI Superoperator from its Choi matrix; exact inverse of choi_matrix.
%   J - Choi matrix (output kron input)
%   N - Hilbert-space dimension (default sqrt(size(J, 1)))
%
%   Peer of python/src/qrobustness/lindblad.py:superop_from_choi.

    if nargin < 2
        N = round(sqrt(size(J, 1)));
    end
    S = reshape(permute(reshape(J, [N N N N]), [2 4 1 3]), [N * N, N * N]);
end

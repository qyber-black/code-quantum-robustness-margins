% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function J = choi_matrix(S)
%CHOI_MATRIX Choi matrix J = sum_ij Phi(E_ij) kron E_ij (output kron input).
%
%   Pure reindexing: every entry of J is a stored entry of S, moved
%   without arithmetic. With column stacking,
%   Phi(E_ij)(a,b) = S(a + b*N, i + j*N), which the definition places at
%   J(a*N + i, b*N + j), so the whole map is one permutation of the
%   four-index view. That exactness is what lets the verified bound be
%   stated about the represented superoperator rather than about a
%   recomputation of it; superop_from_choi inverts it exactly.
%
%   The earlier form multiplied S against each matrix unit and
%   accumulated, which is arithmetic on floating data.
%
%   Peer of python/src/qrobustness/lindblad.py:choi_matrix.

    N = round(sqrt(size(S, 1)));
    J = reshape(permute(reshape(S, [N N N N]), [3 1 4 2]), [N * N, N * N]);
end

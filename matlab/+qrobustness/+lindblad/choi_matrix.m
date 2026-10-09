% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function J = choi_matrix(S)
%CHOI_MATRIX Choi matrix J = sum_ij Phi(E_ij) kron E_ij (output kron input).
%   S - column-stacked superoperator; J is an exact reindexing of S
%
%   Peer of python/src/qrobustness/lindblad.py:choi_matrix.

    N = round(sqrt(size(S, 1)));
    J = reshape(permute(reshape(S, [N N N N]), [3 1 4 2]), [N * N, N * N]);
end

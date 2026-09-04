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
%   Peer of python/src/qrobustness/lindblad.py:choi_matrix.

    N = round(sqrt(size(S, 1)));
    J = zeros(N * N, N * N);
    for i = 1:N
        for j = 1:N
            Eij = zeros(N, N);
            Eij(i, j) = 1;
            out = reshape(S * Eij(:), N, N);
            J = J + kron(out, Eij);
        end
    end
end

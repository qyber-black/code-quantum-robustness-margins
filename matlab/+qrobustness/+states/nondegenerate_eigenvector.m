% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function [psi, E, gap] = nondegenerate_eigenvector(H, index)
%NONDEGENERATE_EIGENVECTOR Nondegenerate eigenvector of a Hermitian matrix, its eigenvalue and gap.
%   H     - Hermitian matrix
%   index - level in ascending order, 1 = ground state (default 1)
%   psi   - eigenvector
%   E     - eigenvalue
%   gap   - distance to the rest of the spectrum; errors if degenerate
%
%   Peer of python/src/qrobustness/states.py:nondegenerate_eigenvector (which counts from 0).

    if nargin < 2 || isempty(index)
        index = 1;
    end
    [V, D] = eig((H + H') / 2);
    [w, order] = sort(real(diag(D)));
    V = V(:, order);
    others = w([1:index - 1, index + 1:end]);
    if isempty(others)
        gap = Inf;
    else
        gap = min(abs(others - w(index)));
    end
    if gap <= 64 * eps * max(1, max(abs(w)))
        error('qrobustness:states:degenerate', 'The eigenvalue is degenerate; the gap must be positive');
    end
    psi = V(:, index);
    E = w(index);
end

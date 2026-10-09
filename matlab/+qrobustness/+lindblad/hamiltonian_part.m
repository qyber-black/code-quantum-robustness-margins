% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function B = hamiltonian_part(S, rtol)
%HAMILTONIAN_PART Traceless Hermitian B with S = -1i*[B, .], or [] if S is not of that form.
%   S    - column-stacked superoperator
%   rtol - relative Frobenius tolerance for rebuilding S from B (default 1e-12)
%
%   Peer of python/src/qrobustness/lindblad.py:hamiltonian_part.

    if nargin < 2 || isempty(rtol)
        rtol = 1e-12;
    end
    n2 = size(S, 1);
    N = round(sqrt(n2));
    if N * N ~= n2 || size(S, 2) ~= n2
        error('qrobustness:lindblad:shape', 'S must be a square superoperator of size N^2');
    end
    blocksum = zeros(N);
    for a = 1:N
        idx = (a - 1) * N + (1:N);
        blocksum = blocksum + S(idx, idx);
    end
    B = 1i * blocksum / N;
    B = (B + B') / 2;
    scale = norm(S, 'fro');
    if scale == 0
        return
    end
    if norm(S - qrobustness.lindblad.hamiltonian_superop(B), 'fro') > rtol * scale
        B = [];
    end
end

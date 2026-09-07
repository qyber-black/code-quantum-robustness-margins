% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function S = hamiltonian_superop(H)
%HAMILTONIAN_SUPEROP Superoperator of -1i*[H, .] (column-stacked).
%
%   Peer of python/src/qrobustness/lindblad.py:hamiltonian_superop.

    N = size(H, 1);
    I = eye(N);
    S = -1i * (kron(I, H) - kron(H.', I));
end

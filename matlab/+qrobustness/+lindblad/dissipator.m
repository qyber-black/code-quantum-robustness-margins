% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function S = dissipator(V)
%DISSIPATOR Lindblad dissipator D[V] at unit rate (column-stacked).
%
%   Peer of python/src/qrobustness/lindblad.py:dissipator.

    N = size(V, 1);
    I = eye(N);
    VdV = V' * V;
    S = kron(conj(V), V) - 0.5 * (kron(I, VdV) + kron(VdV.', I));
end

function [V, lam] = segment_eig(H)
%SEGMENT_EIG Hermitian eigendecomposition of an interval Hamiltonian.
%   [V, lam] = qrobustness.segment_eig(H)
%
%   H   - interval Hamiltonian; symmetrised as (H + H')/2 first, so that eig
%         returns a unitary V
%   V   - unitary eigenvectors, H = V*diag(lam)*V'
%   lam - real eigenvalues
%
%   Peer of python/src/qrobustness/core.py:segment_eig.

    Hs = (H + H') / 2;
    [V, D] = eig(Hs);
    lam = real(diag(D));
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

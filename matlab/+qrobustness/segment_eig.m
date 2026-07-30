function [V, lam] = segment_eig(H)
%SEGMENT_EIG Hermitian eigendecomposition of a segment Hamiltonian.
%   [V, lam] = SEGMENT_EIG(H) returns a unitary V and real eigenvalues lam
%   with H = V*diag(lam)*V'.
%
%   H is symmetrised first.  eig only dispatches to the Hermitian LAPACK
%   path -- and only then guarantees a unitary V -- for exactly Hermitian
%   input, so the symmetrisation is required, not cosmetic.

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

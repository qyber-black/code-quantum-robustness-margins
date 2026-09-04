% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function F = process_fidelity(S, Uf)
%PROCESS_FIDELITY F_pro = Tr(Sf' S)/N^2 with unitary target Uf.
%
%   Peer of python/src/qrobustness/lindblad.py:process_fidelity.

    N = size(Uf, 1);
    Sf = qrobustness.lindblad.unitary_superop(Uf);
    F = real(trace(Sf' * S)) / N^2;
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function F = average_gate_fidelity(S, Uf)
%AVERAGE_GATE_FIDELITY Average gate fidelity (N F^pro + 1)/(N + 1).
%   S  - channel superoperator (column-stacked)
%   Uf - unitary target
%
%   Peer of python/src/qrobustness/lindblad.py:average_gate_fidelity.

    N = size(Uf, 1);
    F = (N * qrobustness.lindblad.process_fidelity(S, Uf) + 1) / (N + 1);
end

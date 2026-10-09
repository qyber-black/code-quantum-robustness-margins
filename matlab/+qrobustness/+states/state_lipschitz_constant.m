% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function L = state_lipschitz_constant(FT, C)
%STATE_LIPSCHITZ_CONSTANT Lipschitz constant sqrt(1 - F_T^2) C of F = |<chi|psi(mu)>| on F > F_T.
%   FT - fidelity threshold F_T
%   C  - state speed C^st
%
%   Peer of python/src/qrobustness/states.py:state_lipschitz_constant.

    if ~(FT > 0 && FT < 1)
        error('qrobustness:states:threshold', 'Require 0 < FT < 1');
    end
    L = sqrt(1 - FT^2) * C;
end

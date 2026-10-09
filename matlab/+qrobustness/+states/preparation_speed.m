% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function c = preparation_speed(dH, psi0, gap, bound)
%PREPARATION_SPEED Bound on the preparation speed C_prep = ||P_perp dpsi0/dmu||.
%   dH    - derivative of the preparation Hamiltonian
%   psi0  - nondegenerate eigenvector
%   gap   - its spectral gap, > 0
%   bound - 'sigma' (default): sigma_psi0(dH)/gap; 'spread': ||dH||_c/gap
%
%   Peer of python/src/qrobustness/states.py:preparation_speed.

    if nargin < 4 || isempty(bound)
        bound = 'sigma';
    end
    if ~(gap > 0)
        error('qrobustness:states:gap', 'The gap must be positive');
    end
    switch bound
        case 'spread'
            c = qrobustness.states.half_spread(dH) / gap;
        case 'sigma'
            v = dH * psi0;
            m = psi0' * v;
            c = norm(v - m * psi0) / gap;
        otherwise
            error('qrobustness:states:bound', 'bound must be ''sigma'' or ''spread''');
    end
end

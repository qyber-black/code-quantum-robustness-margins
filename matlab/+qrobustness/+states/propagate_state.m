% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function psi = propagate_state(H_list, dt, psi0)
%PROPAGATE_STATE State after the piecewise-constant evolution.
%   H_list - cell array of interval Hamiltonians
%   dt     - interval length
%   psi0   - initial state
%
%   Peer of python/src/qrobustness/states.py:propagate_state.

    psi = psi0(:);
    for k = 1:numel(H_list)
        H = (H_list{k} + H_list{k}') / 2;
        [V, D] = eig(H);
        psi = V * (exp(-1i * real(diag(D)) * dt) .* (V' * psi));
    end
end

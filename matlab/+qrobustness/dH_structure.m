function dH_list = dH_structure(H0, H1, H2, u1, u2, structure)
%DH_STRUCTURE Per-interval derivative dH^(k)/dmu for a multiplicative structure.
%   H0, H1, H2 - drift and control Hamiltonians
%   u1, u2     - controls per interval
%   structure  - 'H0' (dH = H0), 'H1' (dH = u1(k) H1) or 'H2' (dH = u2(k) H2)
%   dH_list    - cell array of dH^(k)/dmu, one per interval
%
%   Peer of python/src/qrobustness/core.py:dH_structure.

    tau = numel(u1);
    dH_list = cell(1, tau);
    switch upper(structure)
        case 'H0'
            for k = 1:tau
                dH_list{k} = H0;
            end
        case 'H1'
            for k = 1:tau
                dH_list{k} = u1(k) * H1;
            end
        case 'H2'
            for k = 1:tau
                dH_list{k} = u2(k) * H2;
            end
        otherwise
            error('qrobustness:dH:Structure', 'structure must be H0, H1, or H2.');
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

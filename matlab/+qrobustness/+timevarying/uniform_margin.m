% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function r0 = uniform_margin(L, F, FT)
%UNIFORM_MARGIN Lipschitz radius r_0 = (F - F_T)/sum_j L_j for time-varying perturbations.
%   L  - per-parameter constants L_j
%   F  - nominal fidelity
%   FT - fidelity threshold F_T
%
%   Every trajectory with sum_j L_j ||delta_j||_inf <= F - F_T keeps F >= F_T.
%
%   Peer of python/src/qrobustness/timevarying.py:uniform_margin.

    if ~(F > FT)
        error('qrobustness:timevarying:surplus', 'Require F > FT');
    end
    r0 = (F - FT) / sum(L(:));
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function r0 = uniform_margin(L, F, FT)
%UNIFORM_MARGIN Certified uniform margin against time-varying uncertainty.
%   Every measurable trajectory delta_j(t) with
%   sum_j L_j ||delta_j||_inf <= F - FT keeps the fidelity at or above FT.
%
%   Numerically this is the first Lipschitz step of the scalar iteration,
%   but it is exposed separately because its CERTIFICATE SEMANTICS differ:
%   it holds for all measurable trajectories in sup-norm, where the
%   iterated margin certifies constant perturbations only.
%
%   Peer of python/src/qrobustness/timevarying.py:uniform_margin.

    if ~(F > FT)
        error('qrobustness:timevarying:surplus', 'Require F > FT');
    end
    r0 = (F - FT) / sum(L(:));
end

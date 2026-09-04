% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function [ell, budget] = fs_margin_joint(Hhat_lists, dt, F0, FT)
%FS_MARGIN_JOINT Exact Choi-speed certificate for a joint box budget.
%   ell(m) is the worst path length over |delta_j(t)| <= m_j.  Every
%   m satisfying ell(m) <= budget is certified safe.  Peer of
%   python/src/qrobustness/timevarying.py:fs_margin_joint.

    gauge = qrobustness.lengthspace.path_gauge( ...
        qrobustness.lengthspace.interval_grams(Hhat_lists, true, true), dt);
    ell = @(m) qrobustness.lengthspace.path_gauge_C_box(gauge, m);
    budget = qrobustness.lengthspace.angle_budget(F0, FT);
end

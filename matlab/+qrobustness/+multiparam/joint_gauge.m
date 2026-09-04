% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function g = joint_gauge(Hhat_lists, dt)
%JOINT_GAUGE The combined-structure gauge C_joint and its certified region.
%   C_joint(x) = sum_k Delta ||sum_j x_j Hhat_j^(k)||_F captures
%   cancellations between structures that the parameter-wise triangle
%   inequality (the cross-polytope) discards, and detects exact null
%   directions. Raw Frobenius grams; master-lemma case (b).
%
%   Returns a path-gauge struct; the arithmetic lives in
%   qrobustness.lengthspace, this adds the B_T conversion through
%   joint_gauge_L_dir and joint_gauge_inradius_certified.
%
%   Peer of python/src/qrobustness/multiparam.py:joint_gauge.

    grams = qrobustness.lengthspace.interval_grams(Hhat_lists);
    g = qrobustness.lengthspace.path_gauge(grams, dt);
end

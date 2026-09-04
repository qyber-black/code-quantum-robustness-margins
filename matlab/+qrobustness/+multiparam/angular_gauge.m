% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function g = angular_gauge(Hhat_lists, dt)
%ANGULAR_GAUGE The static Choi-angular gauge.
%   The strongest zero-evaluation static region of the paper. A constant
%   displacement is a trajectory, so the exact Choi-speed certificate
%   applies with C_FS(x) = dt sum_k sqrt(x' Q^(k) x) over traceless
%   normalised interval Grams, against the fidelity ANGLE budget
%   acos(FT) - acos(F_nu). Contains the joint Lipschitz gauge region and
%   is insensitive to identity components (pure global phase).
%   Master-lemma case (a-ii).
%
%   Peer of python/src/qrobustness/multiparam.py:angular_gauge.

    grams = qrobustness.lengthspace.interval_grams(Hhat_lists, true, true);
    g = qrobustness.lengthspace.path_gauge(grams, dt);
end

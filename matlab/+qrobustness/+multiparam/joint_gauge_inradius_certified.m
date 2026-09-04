% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function r = joint_gauge_inradius_certified(g, surplus, FT, N)
%JOINT_GAUGE_INRADIUS_CERTIFIED Certified Euclidean inradius of the region.
%
%   Peer of JointGauge.inradius_certified.

    L = qrobustness.lipschitz_constant(FT, N, ...
        qrobustness.lengthspace.path_gauge_alpha_cs(g));
    r = qrobustness.lengthspace.margin_from(surplus, L);
end

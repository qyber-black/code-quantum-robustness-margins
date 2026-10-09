% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function mu = polytope_boundary_point(P, d)
%POLYTOPE_BOUNDARY_POINT Point on the cross-polytope boundary in direction d.
%   P - from qrobustness.multiparam.safe_polytope
%   d - direction
%
%   Peer of python/src/qrobustness/multiparam.py:SafePolytope.boundary_point.

    d = d(:);
    s = P.surplus / dot(P.L, abs(d));
    mu = P.centre + s * d;
end

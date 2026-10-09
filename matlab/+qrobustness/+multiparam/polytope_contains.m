% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function tf = polytope_contains(P, mu)
%POLYTOPE_CONTAINS True if mu lies in the certified cross-polytope.
%   P  - from qrobustness.multiparam.safe_polytope
%   mu - parameter vector
%
%   Peer of python/src/qrobustness/multiparam.py:SafePolytope.contains.

    tf = dot(P.L, abs(mu(:) - P.centre)) <= P.surplus;
end

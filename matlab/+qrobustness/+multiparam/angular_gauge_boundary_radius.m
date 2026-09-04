% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function r = angular_gauge_boundary_radius(g, d, F_nu, FT)
%ANGULAR_GAUGE_BOUNDARY_RADIUS Certified radius along d under the angle budget.
%
%   Peer of AngularGauge.boundary_radius.

    r = qrobustness.lengthspace.path_gauge_radius(g, d, ...
        qrobustness.multiparam.angular_gauge_budget(F_nu, FT));
end

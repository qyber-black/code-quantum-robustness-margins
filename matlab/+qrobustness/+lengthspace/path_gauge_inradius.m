% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function r = path_gauge_inradius(g, budget)
%PATH_GAUGE_INRADIUS Certified Euclidean inradius budget / alpha_cs.
%
%   Peer of PathGauge.inradius.

    r = qrobustness.lengthspace.margin_from(budget, ...
        qrobustness.lengthspace.path_gauge_alpha_cs(g));
end

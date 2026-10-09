% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function g = angular_gauge(Hhat_lists, dt)
%ANGULAR_GAUGE Static angular gauge C^stat_FS(x) = dt sum_k sqrt(x' Q^(k) x).
%   Hhat_lists - cell{p}{tau} of structure matrices
%   dt         - interval length
%   g          - path-gauge struct over traceless, normalised Grams Q^(k)
%
%   Peer of python/src/qrobustness/multiparam.py:angular_gauge.

    grams = qrobustness.lengthspace.interval_grams(Hhat_lists, true, true);
    g = qrobustness.lengthspace.path_gauge(grams, dt);
end

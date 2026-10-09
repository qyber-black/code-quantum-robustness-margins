% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function g = path_gauge(grams, dt)
%PATH_GAUGE Quadratic path gauge C(x) = dt sum_k sqrt(x' G^(k) x).
%   grams - cell{tau} of Gram matrices G^(k)
%   dt    - interval length
%   g     - struct with fields grams and dt, used by path_gauge_C,
%           path_gauge_C_box, path_gauge_radius, path_gauge_alpha_cs and
%           path_gauge_inradius
%
%   Peer of python/src/qrobustness/lengthspace.py:PathGauge.

    g = struct('grams', {grams}, 'dt', dt);
end

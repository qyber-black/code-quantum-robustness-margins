% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function g = path_gauge(grams, dt)
%PATH_GAUGE A quadratic path gauge C(x) = dt sum_k sqrt(x' G^(k) x).
%   Returns a struct with fields grams (cell) and dt; operate on it with
%   path_gauge_C, path_gauge_C_box, path_gauge_radius, path_gauge_alpha_cs
%   and path_gauge_inradius.
%
%   A struct rather than a classdef: the .m sources run under both MATLAB
%   and Octave, and Octave's classdef support is not complete enough to
%   rely on here.
%
%   Peer of python/src/qrobustness/lengthspace.py:PathGauge.

    g = struct('grams', {grams}, 'dt', dt);
end

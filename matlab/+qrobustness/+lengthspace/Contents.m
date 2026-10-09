% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
% +LENGTHSPACE  Shared pieces of the gauge-budget certificates.
%   Traceless per-interval Gram matrices G^(k), the angle budget
%   arccos F_T - arccos F_0, the quadratic path gauge
%   C(x) = dt sum_k sqrt(x' G^(k) x) with its box bound and certified
%   inradius, and refinement of interval lists onto a finer grid.
%
%   Gauge
%     interval_grams       - per-interval Gram matrices of the structures
%     path_gauge           - the gauge struct
%     path_gauge_C         - gauge value C(x)
%     path_gauge_C_box     - maximum of C over the box |x_j| <= m_j
%     path_gauge_alpha_cs  - certified upper bound on C over the unit sphere
%     path_gauge_radius    - certified radius budget/C(d) along d
%     path_gauge_inradius  - certified Euclidean inradius budget/alpha_cs
%
%   Budgets and grids
%     angle_budget         - arccos F_T - arccos F_0
%     margin_from          - budget/speed, Inf at zero speed
%     refine               - split each interval into q equal sub-intervals
%
%   Peer of python/src/qrobustness/lengthspace.py.

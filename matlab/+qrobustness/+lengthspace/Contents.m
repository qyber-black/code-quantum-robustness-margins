% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
% +LENGTHSPACE  The path gauge shared by every trajectory certificate.
%
%   One construction underlies the joint, angular and time-varying
%   certificates: a perturbation trajectory has a LENGTH in the space of
%   propagators, and a gauge that bounds that length per unit budget turns
%   a fidelity surplus into a certified radius. Putting it here once is why
%   QROBUSTNESS.MULTIPARAM and QROBUSTNESS.TIMEVARYING agree by
%   construction rather than by inspection.
%
%   The gauge is quadratic, C(x) = dt sum_k sqrt(x' G^(k) x), with the
%   per-interval Gram matrices G^(k) built from the structure lists. It is
%   positively homogeneous of degree one, which is what makes a radial
%   certificate meaningful, and its Cauchy-Schwarz bound gives a certified
%   Euclidean inradius without sampling any direction.
%
%   Structures are CENTRED before the Grams are formed: the trace part of a
%   perturbation is a global phase the fidelity cannot see, so leaving it in
%   would inflate every constant.
%
%   Gauge
%     interval_grams       - per-interval Gram matrices of the structures
%     path_gauge           - the gauge object
%     path_gauge_C         - the gauge value along a direction
%     path_gauge_C_box     - worst case over the box |x_j| <= m_j
%     path_gauge_alpha_cs  - certified bound on the gauge over the sphere
%     path_gauge_radius    - certified radius budget/C(d) along d
%     path_gauge_inradius  - certified Euclidean inradius budget/alpha_cs
%
%   Budgets and grids
%     angle_budget         - acos(FT) - acos(F0), the angular case
%     margin_from          - budget/speed, with Inf at zero speed
%     refine               - split each interval into q equal sub-intervals
%
%   Peer of python/src/qrobustness/lengthspace.py.

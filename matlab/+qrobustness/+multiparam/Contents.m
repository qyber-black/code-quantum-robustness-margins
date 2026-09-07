% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
% +MULTIPARAM  Joint certificates over several uncertain parameters.
%
%   With p structures uncertain at once, the certified safe set is a region
%   in R^p rather than an interval, and this package builds three nested
%   ones, in increasing order of what they certify and of what they cost:
%
%   1. The free cross-polytope, sum_j L_j |mu_j| <= F - FT (SAFE_POLYTOPE):
%      closed form, no off-nominal fidelity evaluation at all.
%   2. The joint and angular gauge regions (JOINT_GAUGE, ANGULAR_GAUGE):
%      also closed form, but they see cancellations BETWEEN structures that
%      the separable polytope cannot, so they are larger.
%   3. Directional iteration (DIRECTIONAL_MARGIN): Algorithm 1 along a ray,
%      which resolves the true crossing at the cost of evaluations.
%
%   The angular safe step never falls below the Lipschitz one, so stepping
%   with the angular gauge is never worse and usually needs no more
%   evaluations.
%
%   Regions
%     structure_constants              - per-parameter constants (C, L)
%     safe_polytope                    - the free cross-polytope
%     polytope_contains                - membership test
%     polytope_boundary_point          - boundary point along a direction
%     joint_gauge                      - combined-structure gauge region
%     joint_gauge_L_dir                - sharp directional constant
%     joint_gauge_inradius_certified   - certified Euclidean inradius
%     angular_gauge                    - static Choi-angular gauge
%     angular_gauge_budget             - the angle budget it spends
%     angular_gauge_boundary_radius    - certified radius along d
%     angular_gauge_inradius_certified - certified Euclidean inradius
%
%   Directions and rays
%     axis_directions      - the 2p signed coordinate directions
%     diagonal_directions  - all 2^p normalised diagonals
%     make_ray_fn          - restrict a fidelity function to a ray
%     directional_margin   - certified margin along a ray
%
%   Peer of python/src/qrobustness/multiparam.py.

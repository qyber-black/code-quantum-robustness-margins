% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
% +MULTIPARAM  Certified regions and directional margins for p simultaneous structures.
%   Per-parameter constants L_j = B_T C_j, the separable cross-polytope
%   sum_j L_j |mu_j - nu_j| <= F_nu - F_T, the combined-structure gauge
%   C_joint, the static angular gauge C^stat_FS, and directional margins
%   along rays via qrobustness.iterative_margin.
%
%   Regions
%     structure_constants              - per-parameter constants C_j, L_j
%     safe_polytope                    - certified cross-polytope
%     polytope_contains                - membership test
%     polytope_boundary_point          - boundary point along a direction
%     joint_gauge                      - combined-structure gauge C_joint
%     joint_gauge_L_dir                - directional constant B_T C_joint(d)
%     joint_gauge_inradius_certified   - certified Euclidean inradius
%     angular_gauge                    - static angular gauge C^stat_FS
%     angular_gauge_budget             - angle budget arccos F_T - arccos F_nu
%     angular_gauge_boundary_radius    - certified radius along d
%     angular_gauge_inradius_certified - certified Euclidean inradius
%
%   Directions and rays
%     axis_directions      - the 2p signed coordinate directions
%     diagonal_directions  - all 2^p normalised diagonals
%     make_ray_fn          - restrict a fidelity function to a ray
%     directional_margin   - margin along a ray
%
%   Peer of python/src/qrobustness/multiparam.py.

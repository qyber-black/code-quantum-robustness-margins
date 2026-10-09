% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
% +STATES  State-fidelity margins for pure states under structured Hamiltonian perturbations.
%   The half spread ||X||_c, the state speed C^st and its joint gauge, the
%   preparation speed C_prep of a gapped eigenvector, state propagation,
%   fidelity and Fubini-Study angle, the angular (or Lipschitz) state margin,
%   and uniform trajectory radii.
%
%   Functions
%     half_spread                 - ||X||_c = (lambda_max - lambda_min)/2
%     state_speed                 - state speed C^st
%     state_speed_joint           - joint gauge of several structures
%     propagate_state             - piecewise-constant evolution of a state
%     state_fidelity              - |<chi|psi>|
%     fs_angle                    - Fubini-Study angle, accurate for small angles
%     state_lipschitz_constant    - sqrt(1 - F_T^2) C^st
%     state_angular_margin        - state-fidelity margin
%     trajectory_radius           - radius for time-varying perturbations
%     nondegenerate_eigenvector   - eigenvector, eigenvalue and gap of a nondegenerate level
%     preparation_speed           - bound sigma/gap or ||dH||_c/gap on C_prep
%     gapped_preparation_radius   - radius for a gapped, parameter-dependent preparation
%
%   Peer of python/src/qrobustness/states.py.

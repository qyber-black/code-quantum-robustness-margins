% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
% +LINDBLAD  Open-system certificates in Liouville space.
%
%   Everything here works on column-stacked superoperators: a Lindblad
%   generator propagates a vectorised density matrix, the channel is the
%   ordered product of the per-interval propagators, and the figure of merit
%   is the process fidelity against a unitary target. The closed-system
%   threshold FT transfers as FT^2, since the process fidelity of a unitary
%   channel is the square of the gate fidelity.
%
%   The certificate is the same shape as the closed-system one: a Lipschitz
%   constant in the uncertain rate, from the diamond norm of the structure
%   generator, turns a fidelity surplus into a certified radius, and
%   OPEN_MARGIN iterates outward from there over the one-sided rate domain.
%
%   DIAMOND_NORM here is the SOLVER-FREE bound, not an SDP. The Watrous
%   program is a minimisation, so every feasible point is already an upper
%   bound and only tightening needs a solver; neither MATLAB nor Octave has
%   a portable SDP solver, so the solver-free method takes the plain name.
%   Its Python peer is DIAMOND_NORM_FREE, and Python's DIAMOND_NORM is the
%   cvxpy SDP, which has no peer here.
%
%   Superoperators
%     hamiltonian_superop   - superoperator of -1i*[H, .]
%     dissipator            - Lindblad dissipator D[V] at unit rate
%     unitary_superop       - superoperator of the conjugation U . U'
%     channel               - ordered product over the intervals
%     choi_matrix           - Choi matrix, output kron input
%     superop_from_choi     - inverse of choi_matrix (exact reindexing)
%
%   Fidelity and certificates
%     process_fidelity      - Tr(Sf' S)/N^2 against a unitary target
%     average_gate_fidelity - (N F_pro + 1)/(N + 1)
%     diamond_norm          - solver-free diamond-norm upper bound
%     rate_lipschitz        - 0.5 * t_f * dnorm, the constant-rate case
%     open_margin           - certified margin for a scalar rate
%
%   Peer of python/src/qrobustness/lindblad.py.

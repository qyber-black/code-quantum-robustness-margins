% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
% +LINDBLAD  Open-system (Lindblad) robustness margins on the process fidelity F^pro.
%   Column-stacking superoperator builders, the piecewise-constant channel,
%   F^pro, Choi conversions, verified diamond-norm upper bounds (solver-free
%   and closed form), the constant L_j = 0.5 t_f ||G||_diamond, and scalar
%   open-system margins. diamond_norm here is the solver-free bound; the
%   Python SDP variant has no counterpart.
%
%   Superoperators
%     hamiltonian_superop   - superoperator of -1i*[H, .]
%     dissipator            - Lindblad dissipator D[V] at unit rate
%     unitary_superop       - superoperator of the conjugation U . U'
%     channel               - ordered product over the intervals
%     choi_matrix           - Choi matrix, output kron input
%     superop_from_choi     - inverse of choi_matrix (exact reindexing)
%
%   Fidelity and margins
%     process_fidelity      - F^pro = Tr(Sf' S)/N^2 against a unitary target
%     average_gate_fidelity - (N F^pro + 1)/(N + 1)
%     diamond_norm          - verified solver-free diamond-norm upper bound
%     hamiltonian_part      - B with S = -1i*[B, .], or [] otherwise
%     hamiltonian_dnorm     - verified lambda_max(B) - lambda_min(B) for -1i*[B, .]
%     rate_lipschitz        - L = 0.5 t_f dnorm, constant-rate case
%     open_margin           - margin for a scalar rate on [0, Inf)
%
%   Peer of python/src/qrobustness/lindblad.py.

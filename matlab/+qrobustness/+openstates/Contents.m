% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
% +OPENSTATES  Open-system state-fidelity margins under piecewise-constant Lindblad generators.
%   The speed constant D = sum_k dt_k ||Ghat_k||_diamond from verified
%   diamond-norm upper bounds, density-operator evolution, the trace distance,
%   and the margin of F(mu) = <chi|rho(mu)|chi> with Lipschitz constant D/2.
%   Superoperators are column-stacked as in qrobustness.lindblad.
%
%   Functions
%     open_speed                       - speed constant D
%     evolve_density                   - density operator after the channel
%     trace_distance                   - ||rho - sigma||_1 / 2
%     open_state_fidelity_margin       - margin of <chi|rho(mu)|chi>
%
%   Peer of python/src/qrobustness/openstates.py.

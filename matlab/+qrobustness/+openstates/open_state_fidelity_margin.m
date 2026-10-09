% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function result = open_state_fidelity_margin(fidelity_fn, D, FT, varargin)
%OPEN_STATE_FIDELITY_MARGIN Margin of F(mu) = <chi|rho(mu)|chi> with Lipschitz constant D/2.
%   fidelity_fn - handle mu -> F
%   D           - speed constant from open_speed, > 0
%   FT          - fidelity threshold F_T
%   result      - as qrobustness.iterative_margin
%
%   Name-value options are passed to qrobustness.iterative_margin; pass
%   'omega' to keep every generator valid.
%
%   Peer of python/src/qrobustness/openstates.py:open_state_fidelity_margin.

    if D <= 0
        error('qrobustness:openstates:speed', 'Speed constant D must be positive');
    end
    result = qrobustness.iterative_margin(fidelity_fn, 0.5 * D, FT, varargin{:});
end

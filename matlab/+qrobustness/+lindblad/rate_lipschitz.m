% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function L = rate_lipschitz(dnorm_value, t_f)
%RATE_LIPSCHITZ Lipschitz constant of the process fidelity in a Lindblad rate.
%
%   L = QROBUSTNESS.LINDBLAD.RATE_LIPSCHITZ(DNORM_VALUE, T_F) returns
%   0.5 * T_F * DNORM_VALUE, the time-independent case of
%   QROBUSTNESS.LINDBLAD.OPEN_STRUCTURE_CONSTANTS: a structure generator
%   applied at constant strength over the whole gate contributes
%   0.5 * t_f * dnorm(G), the factor one half being the process fidelity's
%   sensitivity to a channel deviation measured in the diamond norm.
%
%   Pass dn.value for the solved diamond norm or dn.value_certified for the
%   verified upper bound; which of the two is appropriate is the caller's
%   decision, not this function's.
%
%   Peer of qrobustness.lindblad.rate_lipschitz in Python.

    L = 0.5 * t_f * dnorm_value;
end

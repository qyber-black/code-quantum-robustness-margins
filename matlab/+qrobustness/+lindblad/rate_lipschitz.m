% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function L = rate_lipschitz(dnorm_value, t_f)
%RATE_LIPSCHITZ Lipschitz constant L = 0.5 t_f dnorm of F^pro in a constant Lindblad rate.
%   dnorm_value - diamond norm of the structure generator (dn.value_certified
%                 for the verified bound)
%   t_f         - gate time
%
%   Peer of python/src/qrobustness/lindblad.py:rate_lipschitz.

    L = 0.5 * t_f * dnorm_value;
end

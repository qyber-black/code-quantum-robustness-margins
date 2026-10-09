% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function t = trace_distance(rho, sigma)
%TRACE_DISTANCE Trace distance ||rho - sigma||_1 / 2.
%
%   Peer of python/src/qrobustness/openstates.py:trace_distance.

    D = rho - sigma;
    t = 0.5 * sum(abs(eig((D + D') / 2)));
end

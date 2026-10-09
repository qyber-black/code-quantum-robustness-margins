% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function h = half_spread(H)
%HALF_SPREAD Half spread ||H||_c = (lambda_max - lambda_min)/2 of a Hermitian matrix.
%
%   Peer of python/src/qrobustness/states.py:half_spread.

    w = sort(real(eig((H + H') / 2)));
    h = (w(end) - w(1)) / 2;
end

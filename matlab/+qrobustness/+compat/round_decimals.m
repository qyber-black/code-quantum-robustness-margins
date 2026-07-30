function Y = round_decimals(X, n)
%ROUND_DECIMALS Round to n decimal places (Octave-safe).
    if nargin < 2
        n = 0;
    end
    if qrobustness.compat.is_octave()
        f = 10^n;
        Y = round(X * f) / f;
    else
        Y = round(X, n);
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

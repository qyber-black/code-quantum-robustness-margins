function loc = legend_location()
%LEGEND_LOCATION Portable 'Location' value for legend.
%   loc = LEGEND_LOCATION() returns 'best' under MATLAB and 'northeast' under
%   Octave, which has not implemented 'best' and substitutes 'northeast' with
%   a warning.  Requesting the substitute directly gives the same placement
%   without the warning.

    if qrobustness.compat.is_octave()
        loc = 'northeast';
    else
        loc = 'best';
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

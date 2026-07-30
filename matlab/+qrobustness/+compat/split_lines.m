function lines = split_lines(txt)
%SPLIT_LINES Split text into lines (Octave-safe alternative to splitlines).
    if exist('splitlines', 'builtin') || (exist('splitlines', 'file') == 2 && ~qrobustness.compat.is_octave())
        lines = splitlines(txt);
        return;
    end
    if isempty(txt)
        lines = {''};
        return;
    end
    parts = strsplit(txt, {'\r\n', '\n', '\r'});
    if ischar(parts)
        lines = {parts};
    else
        lines = parts;
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

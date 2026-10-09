function assert_error(fh, identifier, what)
%ASSERT_ERROR Assert that a call fails with a given error identifier.
%   fh         - function handle called with no arguments
%   identifier - required error identifier
%   what       - description of the condition, used in the failure message
%
%   See also ERROR.

    ok = false;
    try
        fh();
    catch err
        if ~strcmp(err.identifier, identifier)
            error('qrobustness:test:WrongError', ...
                  'expected %s for %s, got %s (%s)', ...
                  identifier, what, err.identifier, err.message);
        end
        ok = true;
    end
    assert(ok, 'expected %s for %s, no error raised', identifier, what);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

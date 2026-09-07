function assert_error(fh, identifier, what)
%ASSERT_ERROR Assert that a call fails with a specific error identifier.
%   ASSERT_ERROR(FH, IDENTIFIER, WHAT) calls FH() and requires it to raise
%   an error whose identifier is IDENTIFIER. WHAT names the condition under
%   test and appears in the failure message.
%
%   The identifier is the point. A bare try/catch passes when the call
%   fails for ANY reason -- a renamed function, a typo in the test, a
%   missing path -- which is precisely the regression such a test is meant
%   to catch, so it reports green exactly when it should report red. The
%   Python peers assert on the exception type and message for the same
%   reason.
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

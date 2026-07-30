function test_threshold_error()
    fidelity_fn = @(mu) 0.95;
    threw = false;
    try
        qrobustness.iterative_margin(fidelity_fn, 1.0, 0.99, 'mu0', 0);
    catch
        threw = true;
    end
    assert(threw, 'Expected error when FT >= F(mu0)');
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

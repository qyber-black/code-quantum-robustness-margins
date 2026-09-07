% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function ray = make_ray_fn(fidelity_fn, mu0, d)
%MAKE_RAY_FN Restrict a multi-parameter fidelity function to mu0 + s d.
%
%   Peer of python/src/qrobustness/multiparam.py:make_ray_fn.

    mu0 = mu0(:);
    d = d(:);
    ray = @(s) fidelity_fn(mu0 + s * d);
end

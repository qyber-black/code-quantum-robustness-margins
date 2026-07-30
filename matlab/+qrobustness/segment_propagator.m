function U = segment_propagator(V, lam, dt)
%SEGMENT_PROPAGATOR expm(-1i*dt*H) from the eigendecomposition of H.
%   U = SEGMENT_PROPAGATOR(V, lam, dt) with [V, lam] = qrobustness.segment_eig(H).

    U = V * diag(exp(-1i * dt * lam)) * V';
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

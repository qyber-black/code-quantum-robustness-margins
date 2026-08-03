function Hc = traceless(Hhat)
%TRACELESS Remove the trace part: Hbar = H - (Tr H / N) I.
%   The trace-amplitude fidelity is invariant under a global phase, and the
%   trace part of a perturbation structure contributes only such a phase to the
%   propagator.  Centring therefore leaves the fidelity -- and hence the margin
%   -- unchanged while making ||Hbar||_F <= ||H||_F, so it can only tighten the
%   Lipschitz constant (paper, Sec. IV).

    [n, m] = size(Hhat);
    if n ~= m
        error('qrobustness:traceless:Square', ...
            'structure matrix must be square.');
    end
    if norm(Hhat - Hhat', 'fro') > 1e-10 * max(1, norm(Hhat, 'fro'))
        error('qrobustness:traceless:Hermitian', ...
            'structure matrix must be Hermitian.');
    end
    Hc = Hhat - (trace(Hhat) / n) * eye(n);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

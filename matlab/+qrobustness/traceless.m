function Hc = traceless(Hhat)
%TRACELESS Remove the trace part: Hbar = H - (Tr H / N) I.
%   The trace-amplitude fidelity is invariant under a global phase, and the
%   trace part of a perturbation structure contributes only such a phase to the
%   propagator.  Centring therefore leaves the fidelity -- and hence the margin
%   -- unchanged while making ||Hbar||_F <= ||H||_F, so it can only tighten the
%   Lipschitz constant (paper, Sec. IV).
%
%   Peer of python/src/qrobustness/core.py.  Both sides now have one
%   definition, validating and shared; the reference briefly carried a
%   second, unvalidated copy in lengthspace.py, which this file never did.

    [n, m] = size(Hhat);
    if n ~= m
        error('qrobustness:traceless:Square', ...
            'structure matrix must be square.');
    end
    % Elementwise, matching numpy.allclose in the reference's core.traceless:
    % |H - H'| <= atol + rtol*|H'|, with the same HERMITIAN_RTOL and
    % HERMITIAN_ATOL. A Frobenius-norm ratio was used here before and is a
    % different test: it accepted a large-norm matrix carrying a small
    % absolute asymmetry that the reference rejects, so the two engines
    % disagreed on which structures were valid.
    hermitian_rtol = 1e-10;
    hermitian_atol = 1e-12;
    if any(any(abs(Hhat - Hhat') > hermitian_atol + hermitian_rtol * abs(Hhat')))
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

function dU = dU_dmu_exact(V, lam, dH, dt)
%DU_DMU_EXACT Closed-form derivative of expm(-1i*dt*H) along dH.
%   dU = qrobustness.dU_dmu_exact(V, lam, dH, dt)
%
%   V, lam - eigendecomposition of H from qrobustness.segment_eig
%   dH     - derivative dH/dmu on the interval
%   dt     - interval length
%   dU     - -1i*dt * int_0^1 expm(-1i*dt*H*(1-s)) dH expm(-1i*dt*H*s) ds,
%            evaluated as a divided difference in the eigenbasis of H
%
%   Peer of python/src/qrobustness/core.py:dU_dmu_exact.

    lam = lam(:);
    X = 0.5 * dt * (lam.' - lam);            % X(m,n), real, antisymmetric
    ph = exp(-0.5i * dt * lam);
    P = ph * ph.';                           % exp(-0.5i*dt*(lam_m + lam_n))

    S = ones(size(X));
    nz = (X ~= 0);
    S(nz) = sin(X(nz)) ./ X(nz);

    Phi = P .* S;
    Vh = V';
    dU = -1i * dt * (V * ((Vh * dH * V) .* Phi) * Vh);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

function dU = dU_dmu_exact(V, lam, dH, dt)
%DU_DMU_EXACT Exact derivative of expm(-1i*dt*H) for constant Hermitian H.
%   dU = DU_DMU_EXACT(V, lam, dH, dt) with [V, lam] = qrobustness.segment_eig(H)
%   returns
%
%     dU/dmu = -1i*dt * int_0^1 expm(-1i*dt*H*(1-s)) dH expm(-1i*dt*H*s) ds
%
%   in closed form.  H is constant on the interval (piecewise-constant
%   controls), so the integral is a divided difference in the eigenbasis of H.
%   With a = -1i*dt*(lam_n - lam_m) and X = 0.5*dt*(lam_n - lam_m),
%
%     (exp(a) - 1)/a = exp(a/2) * sin(X)/X.
%
%   a is purely imaginary, so this form has no cancellation and needs no
%   magnitude threshold: only the literal X == 0 entries (the diagonal and
%   any exact degeneracies) require masking.
%
%   (V, lam) may be reused across several dH for the same interval.
%
%   See also QROBUSTNESS.SEGMENT_EIG, QROBUSTNESS.DU_DMU_QUAD.

    lam = lam(:);
    X = 0.5 * dt * (lam.' - lam);            % X(m,n), real, antisymmetric
    ph = exp(-0.5i * dt * lam);
    P = ph * ph.';                           % exp(-0.5i*dt*(lam_m + lam_n))

    S = ones(size(X));
    nz = (X ~= 0);
    S(nz) = sin(X(nz)) ./ X(nz);

    Phi = P .* S;
    dU = -1i * dt * (V * ((V' * dH * V) .* Phi) * V');
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

function dU = dU_dmu_integral(H, dH, dt, nodes, weights)
%DU_DMU_INTEGRAL Gauss-Legendre approximation of the interval derivative.
%   dU = qrobustness.dU_dmu_integral(H, dH, dt, nodes, weights)
%
%   H, dH          - interval Hamiltonian and dH/dmu
%   dt             - interval length
%   nodes, weights - from qrobustness.gauss_legendre_01
%   dU             - -1i*dt * int_0^1 expm(-1i*dt*H*(1-s)) dH expm(-1i*dt*H*s) ds
%
%   Peer of python/src/qrobustness/core.py:dU_dmu_integral.

    dU = zeros(size(H));
    for j = 1:numel(nodes)
        s = nodes(j);
        A = expm(-1i * dt * H * (1 - s));
        B = expm(-1i * dt * H * s);
        dU = dU + weights(j) * (A * dH * B);
    end
    dU = -1i * dt * dU;
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

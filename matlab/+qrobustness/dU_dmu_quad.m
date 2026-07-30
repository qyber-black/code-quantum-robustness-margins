function dU = dU_dmu_quad(H, dH, dt, nodes, weights)
%DU_DMU_QUAD Gauss-Legendre approximation of the segment derivative.
%   dU = DU_DMU_QUAD(H, dH, dt, nodes, weights) approximates
%
%     dU/dmu = -1i*dt * int_0^1 expm(-1i*dt*H*(1-s)) dH expm(-1i*dt*H*s) ds
%
%   with nodes/weights from qrobustness.gauss_legendre_01.  Retained as an
%   alternative to and cross-check on qrobustness.dU_dmu_exact, which is the
%   default: for piecewise-constant controls the integral is exact in closed
%   form, so this path is not needed for accuracy.
%
%   See also QROBUSTNESS.DU_DMU_EXACT, QROBUSTNESS.GAUSS_LEGENDRE_01.

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

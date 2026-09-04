% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function m = fs_margin(Hhat_list, dt, F0, FT, r0)
%FS_MARGIN Closed-form time-varying margin via the Fubini-Study angle.
%   Returns a struct with fields r_fs, r0, r, speed, theta_0, theta_T, F0
%   and speed_halfspread.
%
%   The perturbed propagator relative to the nominal, W = U_S' U, obeys
%   dW/dt = -i delta(t) Htil(t) W exactly. The normalised Choi state of W
%   stays maximally entangled, so its Fubini-Study speed is not merely
%   bounded but EXACT: on interval k it is
%   |delta| ||Hhatbar^(k)||_F / sqrt(N) with Hhatbar the traceless part.
%   The path-length bound and the triangle inequality then give
%
%       r_fs = (acos FT - acos F0) / s,   s = dt sum_k ||Hhatbar||_F/sqrt(N).
%
%   No sampling and no expansion; identity components of the structure
%   (global phase) contribute exactly zero, and the nominal deficit enters
%   as the angle theta_0 = acos F0. Dominance r_fs >= r0 always holds.
%
%   Peer of python/src/qrobustness/timevarying.py:fs_margin.

    if nargin < 5 || isempty(r0)
        r0 = 0;
    end
    if ~(FT > 0 && FT <= 1) || ~(F0 > 0 && F0 <= 1)
        error('qrobustness:timevarying:range', 'Require 0 < FT, F0 <= 1');
    end
    if ~(F0 > FT)
        error('qrobustness:timevarying:surplus', 'Require F0 > FT');
    end

    N = size(Hhat_list{1}, 1);
    s = 0;
    hs = 0;
    for k = 1:numel(Hhat_list)
        Hk = Hhat_list{k};
        s = s + norm(qrobustness.traceless(Hk), 'fro');
        ev = sort(real(eig((Hk + Hk') / 2)));
        hs = hs + 0.5 * (ev(end) - ev(1));
    end
    speed = dt * s / sqrt(N);
    speed_hs = dt * hs;

    theta_T = acos(FT);
    theta_0 = acos(min(F0, 1.0));
    if speed > 0
        r_fs = (theta_T - theta_0) / speed;
    else
        r_fs = Inf;
    end

    m = struct('r_fs', r_fs, 'r0', r0, 'r', max(r0, r_fs), ...
               'speed', speed, 'theta_0', theta_0, 'theta_T', theta_T, ...
               'F0', F0, 'speed_halfspread', speed_hs);
end

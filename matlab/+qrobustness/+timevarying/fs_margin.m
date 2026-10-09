% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function m = fs_margin(Hhat_list, dt, F0, FT, r0)
%FS_MARGIN Fubini-Study radius r_FS for time-varying perturbations.
%   Hhat_list - cell array of interval structures \hat H^(k)
%   dt        - interval length
%   F0        - nominal fidelity, F_T < F0 <= 1
%   FT        - fidelity threshold F_T
%   r0        - Lipschitz radius r_0 (default 0)
%   m         - struct with fields
%                 r_fs  - (arccos F_T - theta_0)/speed
%                 r0, r - r_0 and max(r_0, r_fs)
%                 speed - dt sum_k ||\hat H^(k)||_F/sqrt(N), traceless parts
%                 theta_0, theta_T - arccos F0, arccos F_T
%                 F0, speed_halfspread - F0 and dt sum_k half-spread of \hat H^(k)
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

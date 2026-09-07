% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function r = margin(H_list, dH_list, dt, FT, varargin)
%MARGIN Implied margin of arXiv:2509.08481 Theorem 2.1 (their Eq. 14).
%   Largest budget certified by that theorem for this controller.
%
%   Name-value options:
%     'nominal_error' (default 0)             absorbed as an angle
%     'uncertainty'   (default 'independent') 'independent' (trajectory
%                     class) or 'systematic' (constant class)
%     'rates'         (default [])            precomputed Kosut
%                     interaction-picture measures, systematic only
%
%   Returns a struct with fields m, uncertainty, s_T, a, b, w_max,
%   w_mean, gamma, magnus_ok and vacuous.
%
%   Peer of python/src/qrobustness/berberich.py:margin.

    p = inputParser;
    addParameter(p, 'nominal_error', 0);
    addParameter(p, 'uncertainty', 'independent');
    addParameter(p, 'rates', []);
    parse(p, varargin{:});
    unc = lower(char(p.Results.uncertainty));
    eps0 = p.Results.nominal_error;
    rates = p.Results.rates;

    if ~any(strcmp(unc, {'independent', 'systematic'}))
        error('qrobustness:berberich:uncertainty', ...
            'Unknown uncertainty=%s.', unc);
    end
    if ~(FT > 0 && FT < 1)
        error('qrobustness:berberich:FT', 'FT must satisfy 0 < FT < 1');
    end

    tau = numel(H_list);
    w = zeros(1, tau);
    for k = 1:tau
        w(k) = norm(dH_list{k}, 2);
    end
    w_max = max(w);
    w_mean = mean(w);

    theta_0 = acos(min(1.0, 1.0 - eps0));
    budget = acos(FT) - theta_0;
    if budget <= 0 || w_max <= 0
        r = struct('m', 0, 'uncertainty', unc, 's_T', 0, 'a', NaN, ...
                   'b', NaN, 'w_max', w_max, 'w_mean', w_mean, ...
                   'gamma', NaN, 'magnus_ok', true, 'vacuous', true);
        return
    end
    s_T = sin(budget);

    % X(m) = tau*((tau-1)/2 * delta(m)^2 + Gbar(m)), delta(m) = dt*m*w_max.
    a_depth = tau * (tau - 1) / 2 * (dt * w_max)^2;
    if strcmp(unc, 'independent')
        a = a_depth;
        b = tau * dt * w_mean;
    else
        if isempty(rates)
            rates = qrobustness.kosut.uncertainty_rates(H_list, dH_list, dt);
        end
        T = tau * dt;
        a = a_depth + dt^2 * sum(w.^2) / 2;
        b = T * rates.w_avg;
    end
    m = 2 * s_T / (b + sqrt(b * b + 4 * a * s_T));
    delta = dt * m * w_max;
    if delta > 0
        gamma = (b * m + (a - a_depth) * m * m) / (tau * delta);
    else
        gamma = NaN;
    end

    r = struct('m', m, 'uncertainty', unc, 's_T', s_T, 'a', a, 'b', b, ...
               'w_max', w_max, 'w_mean', w_mean, 'gamma', gamma, ...
               'magnus_ok', delta < pi, 'vacuous', false);
end

function [tau, pval] = kendall_tau_b(x, y)
%KENDALL_TAU_B Kendall's tau_b and its two-sided asymptotic p-value.
%   Hand-rolled rather than delegating to corr(...,'Type','Kendall') or the
%   Octave statistics package, so MATLAB, Octave and the Python reference
%   (scipy.stats.kendalltau, variant='b', method='asymptotic') agree bit for
%   bit.  Ties are handled by the standard tau_b corrections.

    x = x(:);
    y = y(:);
    n = numel(x);
    if numel(y) ~= n
        error('qrobustness:kendall:Size', 'x and y must have equal length.');
    end

    % Concordant minus discordant pairs.
    S = 0;
    for i = 1:n-1
        s = sign(x(i) - x(i+1:n)) .* sign(y(i) - y(i+1:n));
        S = S + sum(s);
    end

    n0 = n * (n - 1) / 2;
    [t1, t1b, t1c] = tie_terms(x);
    [t2, t2b, t2c] = tie_terms(y);

    denom = sqrt((n0 - t1) * (n0 - t2));
    if denom <= 0
        tau = NaN;
        pval = NaN;
        return;
    end
    tau = S / denom;

    % Asymptotic variance of S with tie corrections.
    v0 = n * (n - 1) * (2 * n + 5);
    v = (v0 - t1c - t2c) / 18 ...
        + t1b * t2b / (9 * n * (n - 1) * (n - 2)) ...
        + t1 * t2 / (2 * n * (n - 1));
    z = S / sqrt(v);
    % Two-sided: 2*(1 - Phi(|z|)) = erfc(|z|/sqrt(2)).
    pval = erfc(abs(z) / sqrt(2));
end

function [t, tb, tc] = tie_terms(v)
    u = unique(v);
    t = 0; tb = 0; tc = 0;
    for i = 1:numel(u)
        c = sum(v == u(i));
        t = t + c * (c - 1) / 2;
        tb = tb + c * (c - 1) * (c - 2);
        tc = tc + c * (c - 1) * (2 * c + 5);
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

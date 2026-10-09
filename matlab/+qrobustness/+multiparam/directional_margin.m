% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function result = directional_margin(fidelity_fn, L, FT, d, varargin)
%DIRECTIONAL_MARGIN Margin along the ray mu0 + s d.
%   fidelity_fn - handle mu -> F for a parameter vector mu
%   L           - per-parameter constants L_j
%   FT          - fidelity threshold F_T
%   d           - direction
%   result      - as qrobustness.iterative_margin, in units of s (the
%                 parameter excursion is M*d); method 'zero_gauge' if the
%                 gauge vanishes along d
%
%   Name-value options:
%     'mu0'           - ray origin (default zeros)
%     'L_dir'         - directional constant, e.g. B_T C_joint(d) (default []
%                       uses sum_j L_j |d_j|)
%     'angular_gauge' - angular gauge from angular_gauge; if set, the safe
%                       radius is (arccos F_T - arccos F)/C^stat_FS(d)
%                       (default [])
%   Other options are passed to qrobustness.iterative_margin.
%
%   Peer of python/src/qrobustness/multiparam.py:directional_margin.

    p = inputParser;
    p.KeepUnmatched = true;
    addParameter(p, 'mu0', []);
    addParameter(p, 'L_dir', []);
    addParameter(p, 'angular_gauge', []);
    parse(p, varargin{:});

    d = d(:);
    L = L(:);
    mu0 = p.Results.mu0;
    if isempty(mu0)
        mu0 = zeros(size(d));
    end
    L_dir = p.Results.L_dir;
    if isempty(L_dir)
        L_dir = dot(L, abs(d));
    end

    extra = p.Unmatched;
    names = fieldnames(extra);
    pass = {};
    for i = 1:numel(names)
        pass{end + 1} = names{i};      %#ok<AGROW>
        pass{end + 1} = extra.(names{i});  %#ok<AGROW>
    end

    ag = p.Results.angular_gauge;
    if ~isempty(ag)
        zero_gauge = qrobustness.lengthspace.path_gauge_C(ag, d) == 0;
    else
        zero_gauge = L_dir == 0;
    end
    if zero_gauge
        % The combined structure vanishes on every interval along d: the whole admissible ray is certified, up to the nearer edge
        % of omega.
        omega = [-Inf, Inf];
        if isfield(extra, 'omega')
            omega = extra.omega;
        end
        M_zero = min(abs(omega(1)), abs(omega(2)));
        result = struct('M_minus', M_zero, 'M_plus', M_zero, 'M', M_zero, 'converged_minus', true, 'converged_plus', true, ...
                        'mu_minus', -M_zero, 'mu_plus', M_zero, 'method', 'zero_gauge', 'status_minus', 'zero_gauge', ...
                        'status_plus', 'zero_gauge', 'certificate', 'segment', 'reason_minus', 'zero_gauge', ...
                        'reason_plus', 'zero_gauge', 'M_upper_minus', Inf, 'M_upper_plus', Inf, 'M_upper', Inf, ...
                        'margin_uncertainty', Inf, 'n_unresolved', 0, 'safeguard_minus', false, 'safeguard_plus', false);
        return
    end
    if ~isempty(ag) && ~any(strcmp(names, 'safe_radius_fn'))
        C_FS = qrobustness.lengthspace.path_gauge_C(ag, d);
        acFT = acos(FT);
        srf = @(F) qrobustness.lengthspace.margin_from( ...
            acFT - acos(min(F, 1.0)), C_FS);
        pass{end + 1} = 'safe_radius_fn';
        pass{end + 1} = srf;
    end

    ray = qrobustness.multiparam.make_ray_fn(fidelity_fn, mu0, d);
    result = qrobustness.iterative_margin(ray, L_dir, FT, pass{:});
end

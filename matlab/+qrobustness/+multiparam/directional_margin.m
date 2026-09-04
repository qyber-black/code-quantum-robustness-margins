% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function result = directional_margin(fidelity_fn, L, FT, d, varargin)
%DIRECTIONAL_MARGIN Certified margin along the ray mu0 + s d.
%   Calls qrobustness.iterative_margin with the directional Lipschitz
%   constant, so every directional margin inherits its error control
%   (margin_tol brackets, certificate classes) unchanged.
%
%   Name-value options:
%     'mu0'           (default zeros)  ray origin
%     'L_dir'         (default [])     sharper joint-gauge constant
%                     B_T C_joint(d); otherwise the separable relaxation
%                     sum_j L_j |d_j| is used
%     'angular_gauge' (default [])     path-gauge struct; step and certify
%                     with the Choi-angular safe radius
%                     (acos FT - acos F)/C_FS(d) instead of the Lipschitz
%                     surplus rule. By full-gauge dominance the angular
%                     step is never smaller, so the same crossing is
%                     resolved with fewer evaluations.
%   All further options pass through to iterative_margin.
%
%   The returned margins are in units of s: the certified parameter
%   excursion is M*d.
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
        pass{end+1} = names{i};      %#ok<AGROW>
        pass{end+1} = extra.(names{i});  %#ok<AGROW>
    end

    ag = p.Results.angular_gauge;
    if ~isempty(ag) && ~any(strcmp(names, 'safe_radius_fn'))
        C_FS = qrobustness.lengthspace.path_gauge_C(ag, d);
        acFT = acos(FT);
        srf = @(F) qrobustness.lengthspace.margin_from( ...
            acFT - acos(min(F, 1.0)), C_FS);
        pass{end+1} = 'safe_radius_fn';
        pass{end+1} = srf;
    end

    ray = qrobustness.multiparam.make_ray_fn(fidelity_fn, mu0, d);
    result = qrobustness.iterative_margin(ray, L_dir, FT, pass{:});
end

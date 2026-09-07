% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function P = safe_polytope(centre, L, F, FT)
%SAFE_POLYTOPE Certified cross-polytope sum_j L_j |mu_j - centre_j| <= surplus.
%   Every point of the polytope satisfies F >= FT. Returns a struct with
%   fields centre, L, surplus, and the derived axis_radii,
%   inradius_linf, inradius_l2.
%
%   Peer of python/src/qrobustness/multiparam.py:safe_polytope.

    if ~(F > FT)
        error('qrobustness:multiparam:surplus', 'Require F > FT at the centre');
    end
    centre = centre(:);
    L = L(:);
    surplus = F - FT;
    P = struct('centre', centre, 'L', L, 'surplus', surplus, ...
               'axis_radii', surplus ./ L, ...
               'inradius_linf', surplus / sum(L), ...
               'inradius_l2', surplus / norm(L, 2));
end

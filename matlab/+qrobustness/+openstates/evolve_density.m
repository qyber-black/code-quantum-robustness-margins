% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function rho = evolve_density(G_list, dt, rho0)
%EVOLVE_DENSITY Density operator after the channel prod_k expm(dt G_k).
%   G_list - cell array of interval generators (column-stacked)
%   dt     - interval length
%   rho0   - initial density operator
%
%   Peer of python/src/qrobustness/openstates.py:evolve_density.

    N = size(rho0, 1);
    rho = reshape(qrobustness.lindblad.channel(G_list, dt) * rho0(:), N, N);
end

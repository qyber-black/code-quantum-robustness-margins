% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function a = path_gauge_alpha_cs(g)
%PATH_GAUGE_ALPHA_CS Certified upper bound on C(d) over the unit sphere |d|_2 = 1.
%   g - from qrobustness.lengthspace.path_gauge
%   a - sqrt(t_f lambda_max(sum_k dt G^(k))), by Cauchy-Schwarz over intervals
%
%   Peer of python/src/qrobustness/lengthspace.py:PathGauge.alpha_cs.

    p = size(g.grams{1}, 1);
    Gsum = zeros(p, p);
    for k = 1:numel(g.grams)
        Gsum = Gsum + g.dt * g.grams{k};
    end
    tf = g.dt * numel(g.grams);
    lam = max(eig((Gsum + Gsum') / 2));
    a = sqrt(max(0, tf * lam));
end

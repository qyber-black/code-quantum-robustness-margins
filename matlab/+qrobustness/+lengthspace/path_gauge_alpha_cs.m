% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function a = path_gauge_alpha_cs(g)
%PATH_GAUGE_ALPHA_CS Certified upper bound on max_{|d|_2 = 1} C(d).
%   Cauchy-Schwarz over intervals: C(d) <= sqrt(t_f * d' (sum_k dt G^(k)) d),
%   so the sphere maximum is bounded by sqrt(t_f * lambda_max(...)) -- one
%   eigenvalue computation, certified though possibly conservative. A
%   sphere-sampled value would only be an estimate.
%
%   Peer of PathGauge.alpha_cs.

    p = size(g.grams{1}, 1);
    Gsum = zeros(p, p);
    for k = 1:numel(g.grams)
        Gsum = Gsum + g.dt * g.grams{k};
    end
    tf = g.dt * numel(g.grams);
    lam = max(eig((Gsum + Gsum') / 2));
    a = sqrt(max(0, tf * lam));
end

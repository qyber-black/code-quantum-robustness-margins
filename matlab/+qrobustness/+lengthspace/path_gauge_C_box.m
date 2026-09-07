% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function v = path_gauge_C_box(g, m)
%PATH_GAUGE_C_BOX Worst case of C over the box |x_j| <= m_j.
%   A convex function is maximised over a box at a sign vertex; the 2^p
%   vertices are enumerated exactly.
%
%   Peer of PathGauge.C_box.

    m = m(:);
    p = numel(m);
    nv = 2^p;
    z = zeros(nv, p);
    for v = 0:(nv - 1)
        for j = 1:p
            if bitand(bitshift(v, -(j - 1)), 1)
                z(v + 1, j) = m(j);
            else
                z(v + 1, j) = -m(j);
            end
        end
    end
    total = 0;
    for k = 1:numel(g.grams)
        q = sum((z * g.grams{k}) .* z, 2);
        total = total + sqrt(max(0, max(q)));
    end
    v = g.dt * total;
end

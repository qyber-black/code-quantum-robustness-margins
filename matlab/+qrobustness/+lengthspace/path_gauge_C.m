% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function v = path_gauge_C(g, x)
%PATH_GAUGE_C The gauge value; positively homogeneous of degree one.
%
%   Peer of PathGauge.C.

    x = x(:);
    v = 0;
    for k = 1:numel(g.grams)
        v = v + sqrt(max(0, x' * g.grams{k} * x));
    end
    v = g.dt * v;
end

function [delta_grid, F] = fidelity_vs_delta(fidelity_fn, delta_grid)
%FIDELITY_VS_DELTA Dense sweep F(delta) for plotting (not a certificate).
    F = zeros(size(delta_grid));
    for k = 1:numel(delta_grid)
        F(k) = fidelity_fn(delta_grid(k));
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

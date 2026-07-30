function [nodes, weights] = gauss_legendre_01(n)
%GAUSS_LEGENDRE_01 Nodes and weights for integral on [0,1].

    % Golub--Welsch on [-1,1], then affine map to [0,1]
    beta = (1:n-1) ./ sqrt(4*(1:n-1).^2 - 1);
    T = diag(beta, 1) + diag(beta, -1);
    [V, D] = eig(T);
    x = diag(D);
    w = 2 * (V(1, :).^2);
    nodes = 0.5 * (x + 1);
    weights = 0.5 * w(:);
    % Sort by node
    [nodes, idx] = sort(nodes);
    weights = weights(idx);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

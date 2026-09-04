% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function grams = interval_grams(Hhat_lists, make_traceless, normalise)
%INTERVAL_GRAMS Per-interval Gram matrices of the structure lists.
%   Hhat_lists     : cell{p}{tau} of structure matrices
%   make_traceless : use the traceless part of each structure
%   normalise      : divide by the Hilbert dimension N
%
%   G^(k)_ij = Re Tr(A_i^(k)' A_j^(k)).  Raw grams build the joint Lipschitz
%   gauge C_joint; traceless normalised grams build the Choi-angular and
%   trajectory gauges (the exact Choi speed).
%
%   Peer of python/src/qrobustness/lengthspace.py:interval_grams.

    if nargin < 2 || isempty(make_traceless), make_traceless = false; end
    if nargin < 3 || isempty(normalise), normalise = false; end

    p = numel(Hhat_lists);
    tau = numel(Hhat_lists{1});
    N = size(Hhat_lists{1}{1}, 1);

    grams = cell(1, tau);
    for k = 1:tau
        A = cell(1, p);
        for j = 1:p
            A{j} = Hhat_lists{j}{k};
            if make_traceless
                A{j} = qrobustness.traceless(A{j});
            end
        end
        G = zeros(p, p);
        for i = 1:p
            for j = i:p
                g = real(trace(A{i}' * A{j}));
                if normalise
                    g = g / N;
                end
                G(i, j) = g;
                G(j, i) = g;
            end
        end
        grams{k} = G;
    end
end

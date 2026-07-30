function [P, S] = correlation_matrices(X)
%CORRELATION_MATRICES Pearson and Spearman correlation matrices.
%   X is observations-by-variables (rows = controllers).
    if qrobustness.compat.is_octave()
        P = corr(X);
        S = spearman(X);
    else
        P = corr(X, 'Type', 'Pearson');
        S = corr(X, 'Type', 'Spearman');
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

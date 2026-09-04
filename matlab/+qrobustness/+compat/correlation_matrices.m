function [P, S] = correlation_matrices(X)
%CORRELATION_MATRICES Pearson and Spearman correlation matrices.
%   X is observations-by-variables (rows = controllers).
%
%   Pearson describes linear association and Spearman monotone rank
%   association; both are descriptive.  Kendall's tau_b is available as
%   qrobustness.compat.kendall_tau_b and is used as a cross-check that the
%   reading does not depend on the choice of rank statistic.
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

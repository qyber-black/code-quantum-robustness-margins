function write_correlation_tex(T, path)
%WRITE_CORRELATION_TEX Pearson/Spearman table with an explicit triangle key.
    % M_j = min(M_j-, M_j+) is invariant under reversal of the parameter
    % coordinate while zeta_j changes sign, so |zeta_j| is the
    % orientation-invariant local comparator; correlating against signed zeta
    % can hide a relationship by mixing the two signs.  Matches Fig. 3.
    vars = {'err', 'M_H0', 'M_H1', 'M_H2', 'zeta_H0', 'zeta_H1', 'zeta_H2'};
    labels = {'$\varepsilon_0$', '$M_0$', '$M_1$', '$M_2$', '$|\zeta_0|$', '$|\zeta_1|$', '$|\zeta_2|$'};
    X = qrobustness.compat.margins_matrix(T, vars);
    X(:, 5:7) = abs(X(:, 5:7));
    [P, S] = qrobustness.compat.correlation_matrices(X);
    fid = fopen(path, 'w');
    fprintf(fid, '%% Auto-generated: upper Pearson r, lower Spearman rho; |zeta| (see above)\n');
    fprintf(fid, '\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\n');
    fprintf(fid, ' & %s \\\\\n\\midrule\n', strjoin(labels, ' & '));
    for i = 1:7
        fprintf(fid, '%s', labels{i});
        for j = 1:7
            if i == j
                v = 1.0;
            elseif j > i
                v = P(i, j);
            else
                v = S(i, j);
            end
            fprintf(fid, ' & $%.2f$', v);
        end
        fprintf(fid, ' \\\\\n');
    end
    fprintf(fid, '\\midrule\n');
    fprintf(fid, '%s\n', '\multicolumn{8}{@{}l@{}}{\footnotesize Cells above this table''s main diagonal (1.00): Pearson $r$.} \\');
    fprintf(fid, '%s\n', '\multicolumn{8}{@{}l@{}}{\footnotesize Cells below this table''s main diagonal (1.00): Spearman $\rho$.} \\');
    fprintf(fid, '\\bottomrule\n\\end{tabular}\n');
    fclose(fid);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
% +COMPAT  MATLAB/Octave portability shims and shared CSV schemas.
%
%   Nothing here is part of the certificate machinery. Two kinds of thing
%   live in this package, and both exist so the peers can be held to the
%   Python reference rather than to each other:
%
%   1. Language shims. Octave and MATLAB differ on graphics toolkits,
%      legend placement, string splitting, rounding and the rank
%      statistics, so the drivers call these rather than branching on
%      IS_OCTAVE at every site.
%   2. The result-file schemas. Column names and order are fixed here, in
%      one place, because the Python peer writes the same files and a
%      cross-engine comparison that disagreed about column order would be
%      comparing the wrong numbers.
%
%   Portability
%     is_octave            - true when running under GNU Octave
%     setup_graphics       - select the graphics toolkit used for export
%     export_figure        - write a figure to PNG at a given resolution
%     legend_location      - portable 'Location' value for legend
%     split_lines          - split text into lines
%     round_decimals       - round to n decimal places
%     kendall_tau_b        - Kendall's tau_b and its two-sided p-value
%     correlation_matrices - Pearson and Spearman correlation matrices
%
%   Result-file schemas
%     margins_csv_headers  - column names for margins_table_*.csv
%     kosut_csv_headers    - column names for kosut_comparison_*.csv
%     read_margins_csv     - load margins_table_*.csv as column vectors
%     read_numeric_csv     - load an all-numeric, headerless CSV
%     write_margins_csv    - write the margins table with fixed headers
%     write_kosut_csv      - write the Kosut table with fixed headers
%     build_margins_table  - assemble margins columns from case-study results
%     margins_matrix       - stack named columns into a matrix

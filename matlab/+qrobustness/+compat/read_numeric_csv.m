function data = read_numeric_csv(csv_path)
%READ_NUMERIC_CSV Load an all-numeric CSV (no header) as a double matrix.
    if ~exist(csv_path, 'file')
        error('qrobustness:compat:MissingFile', 'CSV not found: %s', csv_path);
    end
    if exist('readmatrix', 'file') == 2 && ~qrobustness.compat.is_octave()
        data = readmatrix(csv_path);
    else
        data = dlmread(csv_path, ',');
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

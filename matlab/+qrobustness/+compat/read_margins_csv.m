function T = read_margins_csv(csv_path)
%READ_MARGINS_CSV Load margins_table CSV into a struct of column vectors.
    if ~exist(csv_path, 'file')
        error('qrobustness:compat:MissingFile', 'CSV not found: %s', csv_path);
    end
    headers = qrobustness.compat.margins_csv_headers();
    fid = fopen(csv_path, 'r');
    if fid < 0
        error('qrobustness:compat:ReadFail', 'Cannot read %s', csv_path);
    end
    header_line = fgetl(fid);
    fclose(fid);
    if ~ischar(header_line)
        error('qrobustness:compat:EmptyCSV', 'Empty CSV: %s', csv_path);
    end
    file_headers = strsplit(strtrim(header_line), ',');
    if numel(file_headers) ~= numel(headers) || ~isequal(file_headers, headers)
        error('qrobustness:compat:BadHeader', ...
            'Unexpected margins CSV headers in %s', csv_path);
    end
    data = dlmread(csv_path, ',', 1, 0);
    if size(data, 2) ~= numel(headers)
        error('qrobustness:compat:BadShape', ...
            'Expected %d columns, got %d in %s', numel(headers), size(data, 2), csv_path);
    end
    T = struct();
    for j = 1:numel(headers)
        T.(headers{j}) = data(:, j);
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

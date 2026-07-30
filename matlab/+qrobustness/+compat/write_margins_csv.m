function write_margins_csv(path, T)
%WRITE_MARGINS_CSV Write margins table struct/table to CSV with fixed headers.
%   T may be a MATLAB table or a struct with column fields as column vectors.
    headers = qrobustness.compat.margins_csv_headers();
    n = numel(T.controller);
    fid = fopen(path, 'w');
    if fid < 0
        error('qrobustness:compat:WriteFail', 'Cannot write %s', path);
    end
    fprintf(fid, '%s\n', strjoin(headers, ','));
    for i = 1:n
        vals = zeros(1, numel(headers));
        for j = 1:numel(headers)
            vals(j) = T.(headers{j})(i);
        end
        fprintf(fid, '%.15g', vals(1));
        for j = 2:numel(vals)
            fprintf(fid, ',%.15g', vals(j));
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

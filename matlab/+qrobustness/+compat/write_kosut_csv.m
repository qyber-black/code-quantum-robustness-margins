function write_kosut_csv(path, T)
%WRITE_KOSUT_CSV Write Kosut-comparison table struct to CSV with fixed headers.
    headers = qrobustness.compat.kosut_csv_headers();
    n = numel(T.controller);
    fid = fopen(path, 'w');
    if fid < 0
        error('qrobustness:compat:WriteFail', 'Cannot write %s', path);
    end
    fprintf(fid, '%s\n', strjoin(headers, ','));
    for i = 1:n
        for j = 1:numel(headers)
            v = T.(headers{j})(i);
            if j > 1
                fprintf(fid, ',');
            end
            fprintf(fid, '%.15g', v);
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

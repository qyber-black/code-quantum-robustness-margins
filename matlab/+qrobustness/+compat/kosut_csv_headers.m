function headers = kosut_csv_headers()
%KOSUT_CSV_HEADERS Column names for kosut_comparison_*.csv (MATLAB and Python).
%   Keep in sync with scripts/run_time_bandwidth_bound_comparison.py (CSV_HEADERS).
    headers = {'controller', 'fid', 'err'};
    tags = {'H0', 'H1', 'H2'};
    per = {'M', 'KM', 'ratio', 'KTOb', 'Kflb', 'wunc', 'wavg', 'wdev'};
    for t = 1:numel(tags)
        for j = 1:numel(per)
            headers{end+1} = sprintf('%s_%s', per{j}, tags{t}); %#ok<AGROW>
        end
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

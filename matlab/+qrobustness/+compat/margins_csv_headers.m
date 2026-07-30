function headers = margins_csv_headers()
%MARGINS_CSV_HEADERS Column names for margins_table_*.csv.
    headers = { ...
        'controller', 'fid', 'err', ...
        'M_H0', 'Mm_H0', 'Mp_H0', 'zeta_H0', ...
        'M_H1', 'Mm_H1', 'Mp_H1', 'zeta_H1', ...
        'M_H2', 'Mm_H2', 'Mp_H2', 'zeta_H2'};
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

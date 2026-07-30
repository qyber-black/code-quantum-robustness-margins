function T = build_margins_table(R, nC)
%BUILD_MARGINS_TABLE Assemble margins struct columns from case-study results R.
    T = struct();
    T.controller = (1:nC)';
    T.fid = R.H0.fid;
    T.err = R.H0.error;
    structures = {'H0', 'H1', 'H2'};
    for s = 1:3
        tag = structures{s};
        T.(['M_' tag]) = R.(tag).M;
        T.(['Mm_' tag]) = R.(tag).M_minus;
        T.(['Mp_' tag]) = R.(tag).M_plus;
        T.(['zeta_' tag]) = R.(tag).zeta;
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

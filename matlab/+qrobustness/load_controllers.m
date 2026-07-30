function controllers = load_controllers(csv_path, max_error)
%LOAD_CONTROLLERS Load controller CSV and filter by nominal error.
%   CSV columns: id, ?, tf, tau, error, then interleaved u1,u2 samples
%   max_error default 1e-4 (paper: keep 61 of 100)

    if nargin < 2 || isempty(max_error)
        max_error = 1e-4;
    end
    data = qrobustness.compat.read_numeric_csv(csv_path);
    data = sortrows(data, 5);
    data = data(data(:, 5) <= max_error, :);

    n = size(data, 1);
    controllers = cell(n, 1);
    for k = 1:n
        c = struct();
        c.tf = data(k, 3);
        c.tau = data(k, 4);
        c.error = data(k, 5);
        c.fid = 1 - c.error;
        u = reshape(data(k, 6:end), 2, c.tau);
        c.u1 = u(1, :);
        c.u2 = u(2, :);
        controllers{k} = c;
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

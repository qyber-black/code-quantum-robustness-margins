function problem = load_problem(mat_path)
%LOAD_PROBLEM Load problem9-style MAT file.
%   Returns struct with fields H0, H1, H2, Uf, n_qubits, dim.
%
%   problem.N in the MAT file is the number of qubits; the Hilbert space
%   dimension is 2^N.  Both are returned under unambiguous names.

    if nargin < 1 || isempty(mat_path)
        error('qrobustness:load_problem:Path', 'mat_path is required.');
    end
    S = load(mat_path);
    if ~isfield(S, 'problem')
        error('qrobustness:load_problem:Field', 'Expected variable ''problem''.');
    end
    p = S.problem;
    problem = struct();
    problem.H0 = p.H{1};
    problem.H1 = p.H{2};
    problem.H2 = p.H{3};
    problem.Uf = p.UT;
    problem.n_qubits = double(p.N);
    problem.dim = 2^problem.n_qubits;
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

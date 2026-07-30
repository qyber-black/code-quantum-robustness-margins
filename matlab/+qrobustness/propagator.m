function U = propagator(H_list, dt)
%PROPAGATOR Ordered product of piecewise-constant unitaries.
%   U = qrobustness.propagator(H_list, dt)
%   H_list : cell array of Hermitian Hamiltonians H{k} for interval k
%   dt     : pulse duration Delta
%   U      : U(tf) = U{tau} * ... * U{1}

    if isempty(H_list)
        error('qrobustness:propagator:Empty', 'H_list must be non-empty.');
    end
    U = eye(size(H_list{1}));
    for k = 1:numel(H_list)
        U = expm(-1i * dt * H_list{k}) * U;
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

function H_list = perturbed_hamiltonians(H0, H1, H2, u1, u2, structure, delta)
%PERTURBED_HAMILTONIANS Build H{k} for multiplicative structure perturbation.
%   structure : 'H0', 'H1', or 'H2'
%   delta     : mu - mu0 (multiplicative factor on the named Hamiltonian)

    tau = numel(u1);
    if numel(u2) ~= tau
        error('qrobustness:perturbed:Length', 'u1 and u2 must have equal length.');
    end
    H_list = cell(1, tau);
    for k = 1:tau
        switch upper(structure)
            case 'H0'
                H_list{k} = H0 * (1 + delta) + u1(k) * H1 + u2(k) * H2;
            case 'H1'
                H_list{k} = H0 + u1(k) * H1 * (1 + delta) + u2(k) * H2;
            case 'H2'
                H_list{k} = H0 + u1(k) * H1 + u2(k) * H2 * (1 + delta);
            otherwise
                error('qrobustness:perturbed:Structure', ...
                    'structure must be H0, H1, or H2.');
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

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function [C, L] = structure_constants(specs, dt, tau, FT, N)
%STRUCTURE_CONSTANTS Per-parameter constants (C, L) with L_j = B_T C_j.
%   specs : cell array; each entry {'drift', Hhat} or
%           {'control', Hhat, controls}, exactly the cases of
%           qrobustness.structure_constant.
%
%   Peer of python/src/qrobustness/multiparam.py:structure_constants.

    p = numel(specs);
    C = zeros(p, 1);
    L = zeros(p, 1);
    for j = 1:p
        spec = specs{j};
        switch lower(spec{1})
            case 'drift'
                C(j) = qrobustness.structure_constant('drift', spec{2}, dt, tau);
            case 'control'
                C(j) = qrobustness.structure_constant('control', spec{2}, ...
                                                      dt, tau, spec{3});
            otherwise
                error('qrobustness:multiparam:kind', ...
                    'Unknown structure kind %s.', spec{1});
        end
        L(j) = qrobustness.lipschitz_constant(FT, N, C(j));
    end
end

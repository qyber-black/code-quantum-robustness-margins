% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function D = open_speed(Ghat_list, dt, normfn, exact_hamiltonian)
%OPEN_SPEED Speed constant D = sum_k dt_k ||Ghat_k||_diamond from verified upper bounds.
%   Ghat_list         - cell array of structure generators
%   dt                - interval length, scalar or one per interval
%   normfn            - superoperator -> certified diamond-norm bound (default
%                       value_certified of qrobustness.lindblad.diamond_norm)
%   exact_hamiltonian - use qrobustness.lindblad.hamiltonian_dnorm for
%                       Hamiltonian superoperators -1i*[B, .] (default false)
%
%   Peer of python/src/qrobustness/openstates.py:open_speed.

    if nargin < 3 || isempty(normfn)
        normfn = @(G) getfield(qrobustness.lindblad.diamond_norm(G), 'value_certified'); %#ok<GFLD>
    end
    if nargin >= 4 && exact_hamiltonian
        inner = normfn;
        normfn = @(G) exact_or_inner(G, inner);
    end
    dts = dt(:)' .* ones(1, numel(Ghat_list));
    D = 0;
    for k = 1:numel(Ghat_list)
        D = D + dts(k) * normfn(Ghat_list{k});
    end
end

function v = exact_or_inner(G, inner)
    if isempty(qrobustness.lindblad.hamiltonian_part(G))
        v = inner(G);
    else
        v = getfield(qrobustness.lindblad.hamiltonian_dnorm(G), 'value_certified'); %#ok<GFLD>
    end
end

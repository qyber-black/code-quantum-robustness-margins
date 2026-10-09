% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function result = open_margin(fidelity_fn, L, FT_pro, varargin)
%OPEN_MARGIN Margin for a scalar open-system rate.
%   fidelity_fn - handle rate -> F^pro
%   L           - Lipschitz constant, e.g. from rate_lipschitz
%   FT_pro      - process-fidelity threshold
%   result      - as qrobustness.iterative_margin
%
%   Name-value options are passed to qrobustness.iterative_margin; 'omega'
%   defaults to [0, Inf], so no negative rate is certified.
%
%   Peer of python/src/qrobustness/lindblad.py:open_margin.

    % Supply the default only when the caller has not asked for a domain.
    if ~any(strcmpi(varargin(1:2:end), 'omega'))
        varargin = [{'omega', [0, Inf]}, varargin];
    end
    result = qrobustness.iterative_margin(fidelity_fn, L, FT_pro, varargin{:});
end

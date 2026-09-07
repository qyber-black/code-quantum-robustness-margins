% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function result = open_margin(fidelity_fn, L, FT_pro, varargin)
%OPEN_MARGIN Certified margin for a scalar open-system parameter.
%   Thin wrapper over qrobustness.iterative_margin with the open-system
%   Lipschitz constant and, by default, the one-sided domain [0, Inf)
%   appropriate for a decoherence rate; the lower boundary is reported
%   through the usual omega/boundary machinery. All error control
%   (margin_tol brackets) passes through.
%
%   The default domain is the whole point of this wrapper. Without it the
%   iteration would step to negative rates and certify them, which is not
%   a physical statement about decoherence.
%
%   Peer of python/src/qrobustness/lindblad.py:open_margin.

    % Only supply the default when the caller has not asked for a domain.
    if ~any(strcmpi(varargin(1:2:end), 'omega'))
        varargin = [{'omega', [0, Inf]}, varargin];
    end
    result = qrobustness.iterative_margin(fidelity_fn, L, FT_pro, varargin{:});
end

function F = fidelity_bound_at(rates, delta, uncertainty)
%FIDELITY_BOUND_AT Lower bound F_lb at perturbation delta.
%   rates       - from qrobustness.kosut.uncertainty_rates
%   delta       - perturbation
%   uncertainty - 'constant' (default) or 'trajectory'
%
%   Peer of python/src/qrobustness/kosut.py:fidelity_bound_at.

    if nargin < 3; uncertainty = 'constant'; end
    F = qrobustness.kosut.fidelity_bound( ...
        qrobustness.kosut.time_bandwidth(rates, delta, uncertainty));
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

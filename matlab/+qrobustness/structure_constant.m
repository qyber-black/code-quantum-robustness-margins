function C = structure_constant(kind, Hhat, dt, tau, controls)
%STRUCTURE_CONSTANT Structure constant C_{\hat H} for a drift or control perturbation.
%   kind     - 'drift' (C = tau dt ||\hat H||_F) or 'control'
%              (C = dt ||f||_1 ||\hat H||_F)
%   Hhat     - structure matrix \hat H; its traceless part is used
%   dt, tau  - interval length and number of intervals
%   controls - controls f^(k) of the perturbed term (required for 'control')
%
%   Peer of python/src/qrobustness/core.py:structure_constant.

    nf = norm(qrobustness.traceless(Hhat), 'fro');
    switch lower(kind)
        case 'drift'
            % C = t_f * ||H0||_F = tau*dt * ||H0||_F
            C = tau * dt * nf;
        case 'control'
            if nargin < 5 || isempty(controls)
                error('qrobustness:structure:Controls', ...
                    'controls required for kind=''control''.');
            end
            % C = Delta * ||f_m||_1 * ||H_m||_F
            C = dt * norm(controls(:), 1) * nf;
        otherwise
            error('qrobustness:structure:Kind', ...
                'kind must be ''drift'' or ''control''.');
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

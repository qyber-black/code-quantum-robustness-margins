function C = structure_constant(kind, Hhat, dt, tau, controls)
%STRUCTURE_CONSTANT C_Hhat for drift or control uncertainty.
%   kind     : 'drift' or 'control'
%   Hhat     : structure matrix
%   dt, tau  : pulse length and number of intervals
%   controls : row/column vector f_m^{(k)} (required for 'control')

    nf = norm(Hhat, 'fro');
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

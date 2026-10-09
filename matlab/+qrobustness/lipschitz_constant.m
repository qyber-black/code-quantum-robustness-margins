function L = lipschitz_constant(FT, N, C_H)
%LIPSCHITZ_CONSTANT Lipschitz constant L = B_T C_{\hat H}, B_T = sqrt((1 - F_T^2)/N).
%   FT  - fidelity threshold F_T, 0 < F_T < 1
%   N   - Hilbert-space dimension
%   C_H - structure constant C_{\hat H}
%
%   Peer of python/src/qrobustness/core.py:lipschitz_constant.
    if ~(FT > 0 && FT < 1)
        error('qrobustness:lipschitz:FT', 'FT must satisfy 0 < FT < 1.');
    end
    B_T = sqrt((1 - FT^2) / N);
    L = B_T * C_H;
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

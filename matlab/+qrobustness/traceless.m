function Hc = traceless(Hhat)
%TRACELESS Traceless part H - (Tr H / N) I of a Hermitian structure matrix.
%   Hhat - square Hermitian matrix; errors otherwise
%
%   Peer of python/src/qrobustness/core.py:traceless.

    [n, m] = size(Hhat);
    if n ~= m
        error('qrobustness:traceless:Square', ...
            'structure matrix must be square.');
    end
    % Elementwise |H - H'| <= atol + rtol*|H'|, as numpy.allclose in core.traceless.
    hermitian_rtol = 1e-10;
    hermitian_atol = 1e-12;
    if any(any(abs(Hhat - Hhat') > hermitian_atol + hermitian_rtol * abs(Hhat')))
        error('qrobustness:traceless:Hermitian', ...
            'structure matrix must be Hermitian.');
    end
    Hc = Hhat - (trace(Hhat) / n) * eye(n);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

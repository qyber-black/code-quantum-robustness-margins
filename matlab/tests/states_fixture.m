% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function f = states_fixture()
%STATES_FIXTURE Parity fixture for the state and open-system state margins, complex inputs decoded.
%   Written by scripts/gen_states_parity.py.

    this_dir = fileparts(mfilename('fullpath'));
    f = jsondecode(fileread(fullfile(this_dir, 'fixtures', 'states_parity.json')));
    in = f.inputs;
    c = @(s) s.re + 1i * s.im;
    cl = @(a) arrayfun(@(s) c(s), a, 'UniformOutput', false);
    f.H = cl(in.H); f.A = cl(in.A); f.B = cl(in.B);
    f.H = f.H(:)'; f.A = f.A(:)'; f.B = f.B(:)';
    f.psi0 = c(in.psi0); f.psi0 = f.psi0(:);
    f.dt = in.dt;
end

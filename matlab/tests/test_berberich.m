% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function test_berberich()
%TEST_BERBERICH Parity values for +qrobustness/+berberich.
%   Peer of python/tests/test_berberich.py (the two-interval hand formula).

    SZ = [1 0; 0 -1];
    dt = 0.5;
    FT = 0.99;
    H = {0.2 * SZ, 0.4 * SZ};
    dH = {SZ, 2 * SZ};
    rates = struct('w_avg', 0.3);

    indep = qrobustness.berberich.margin(H, dH, dt, FT, 'uncertainty', 'independent');
    sys = qrobustness.berberich.margin(H, dH, dt, FT, ...
        'uncertainty', 'systematic', 'rates', rates);

    assert(abs(indep.m - 0.08878922250864604) < 1e-12, 'independent margin');
    assert(abs(sys.m - 0.21645013028302632) < 1e-12, 'systematic margin');
    assert(indep.magnus_ok && sys.magnus_ok, 'Magnus condition on this example');

    assert_error(@() qrobustness.berberich.margin(H, dH, dt, FT, 'nominal_error', 1.5), ...
        'qrobustness:berberich:nominal_error', 'nominal_error out of range');
    % A largely negative w_avg makes the systematic step overshoot the Magnus bound.
    bad = struct('w_avg', -1e6);
    assert_error(@() qrobustness.berberich.margin({SZ}, {SZ}, 1.0, FT, ...
        'uncertainty', 'systematic', 'rates', bad), ...
        'qrobustness:berberich:Magnus', 'Magnus condition failed');
end

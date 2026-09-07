% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function test_lengthspace()
%TEST_LENGTHSPACE Peer of python/tests/test_lengthspace.py.
%   Properties are checked here rather than values copied from Python;
%   the value-level agreement between the engines is what test_consistency
%   and make test-parity are for.

    ls = @(f) str2func(['qrobustness.lengthspace.' f]);

    sx = [0 1; 1 0];
    sz = [1 0; 0 -1];
    dt = 0.25;
    tau = 4;
    H1 = repmat({sx}, 1, tau);
    H2 = repmat({sz}, 1, tau);

    % Grams of orthogonal structures are diagonal, with ||A||_F^2 on it.
    G = qrobustness.lengthspace.interval_grams({H1, H2});
    assert(numel(G) == tau);
    for k = 1:tau
        assert(abs(G{k}(1, 1) - 2) < 1e-12);
        assert(abs(G{k}(2, 2) - 2) < 1e-12);
        assert(abs(G{k}(1, 2)) < 1e-12);      % Tr(sx' sz) = 0
        assert(norm(G{k} - G{k}', 'fro') < 1e-14);
    end

    % Normalisation divides by the Hilbert dimension.
    Gn = qrobustness.lengthspace.interval_grams({H1, H2}, false, true);
    assert(abs(Gn{1}(1, 1) - 1) < 1e-12);

    % The traceless part of sx and sz is themselves.
    Gt = qrobustness.lengthspace.interval_grams({H1, H2}, true, false);
    assert(norm(Gt{1} - G{1}, 'fro') < 1e-12);

    % Angle budget and its zero case.
    assert(abs(qrobustness.lengthspace.angle_budget(1.0, 0.999) ...
               - acos(0.999)) < 1e-12);
    assert(abs(qrobustness.lengthspace.angle_budget(0.999, 0.999)) < 1e-12);

    % margin_from's zero-speed convention.
    assert(qrobustness.lengthspace.margin_from(1.0, 0) == Inf);
    assert(abs(qrobustness.lengthspace.margin_from(2.0, 4.0) - 0.5) < 1e-14);

    g = qrobustness.lengthspace.path_gauge(G, dt);

    % Positive homogeneity, the property the certificate actually uses.
    x = [0.3; -0.7];
    c1 = qrobustness.lengthspace.path_gauge_C(g, x);
    c2 = qrobustness.lengthspace.path_gauge_C(g, 2.5 * x);
    assert(abs(c2 - 2.5 * c1) < 1e-12);
    assert(qrobustness.lengthspace.path_gauge_C(g, [0; 0]) < 1e-14);

    % Closed form here: C(x) = dt*tau*sqrt(2)*||x||_2.
    expect = dt * tau * sqrt(2) * norm(x, 2);
    assert(abs(c1 - expect) < 1e-12);

    % The box worst case dominates every interior point of the box.
    m = [0.3; 0.7];
    cb = qrobustness.lengthspace.path_gauge_C_box(g, m);
    for t = 0:0.25:1
        for s = 0:0.25:1
            xv = [(2 * t - 1) * m(1); (2 * s - 1) * m(2)];
            assert(qrobustness.lengthspace.path_gauge_C(g, xv) <= cb + 1e-12);
        end
    end

    % alpha_cs bounds the gauge on the unit sphere, as a certified bound
    % must; sampling may not attain it.
    a = qrobustness.lengthspace.path_gauge_alpha_cs(g);
    for th = linspace(0, 2 * pi, 33)
        d = [cos(th); sin(th)];
        assert(qrobustness.lengthspace.path_gauge_C(g, d) <= a + 1e-10);
    end

    % Radius and inradius invert the gauge against the budget.
    budget = 1e-3;
    r = qrobustness.lengthspace.path_gauge_radius(g, x, budget);
    assert(abs(qrobustness.lengthspace.path_gauge_C(g, r * x) - budget) < 1e-12);
    assert(abs(qrobustness.lengthspace.path_gauge_inradius(g, budget) ...
               - budget / a) < 1e-14);

    % refine replicates each interval q times.
    [Hr, Hhr] = qrobustness.lengthspace.refine(H1, H2, 3);
    assert(numel(Hr) == 3 * tau && numel(Hhr) == 3 * tau);
    assert(norm(Hr{1} - sx, 'fro') < 1e-14);
    assert(norm(Hhr{1} - sz, 'fro') < 1e-14);
end

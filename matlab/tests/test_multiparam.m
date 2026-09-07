% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function test_multiparam()
%TEST_MULTIPARAM Multi-parameter layer: polytope, gauges, directional margin.

    mpk = @(f) f;  % readability only
    sx = [0 1; 1 0];
    sz = [1 0; 0 -1];
    dt = 0.25;
    tau = 4;
    N = 2;
    FT = 0.999;

    % --- cross-polytope ------------------------------------------------
    L = [2; 4];
    F = 0.9999;
    P = qrobustness.multiparam.safe_polytope([0; 0], L, F, FT);
    surplus = F - FT;
    assert(abs(P.surplus - surplus) < 1e-15);
    assert(norm(P.axis_radii - surplus ./ L) < 1e-15);
    assert(abs(P.inradius_linf - surplus / sum(L)) < 1e-15);
    assert(abs(P.inradius_l2 - surplus / norm(L, 2)) < 1e-15);

    % The l-infinity inradius is the most conservative of the three, and
    % the axis radii the least: sum >= 2-norm >= max component.
    assert(P.inradius_linf <= P.inradius_l2 + 1e-15);
    assert(P.inradius_l2 <= max(P.axis_radii) + 1e-15);

    % Membership and the boundary point.
    assert(qrobustness.multiparam.polytope_contains(P, [0; 0]));
    for d = [eye(2), -eye(2), [1; 1] / sqrt(2)]
        b = qrobustness.multiparam.polytope_boundary_point(P, d);
        assert(qrobustness.multiparam.polytope_contains(P, b * (1 - 1e-9)));
        assert(~qrobustness.multiparam.polytope_contains(P, b * (1 + 1e-6)));
    end

    % --- direction designs ---------------------------------------------
    A = qrobustness.multiparam.axis_directions(3);
    assert(isequal(size(A), [6 3]));
    for i = 1:6
        assert(abs(norm(A(i, :), 2) - 1) < 1e-14);
    end
    D = qrobustness.multiparam.diagonal_directions(3);
    assert(isequal(size(D), [8 3]));
    for i = 1:8
        assert(abs(norm(D(i, :), 2) - 1) < 1e-14);
    end
    % ORDER, not just membership. Python builds these with
    % itertools.product((-1, 1), repeat=p), which varies the last
    % coordinate fastest. Checking only that each row is a unit vector is
    % order-blind and passes against a permutation, which is exactly how a
    % column-order divergence reached a written CSV unnoticed.
    expected = [-1 -1 -1; -1 -1 1; -1 1 -1; -1 1 1
                 1 -1 -1;  1 -1 1;  1 1 -1;  1 1 1] / sqrt(3);
    assert(norm(D - expected, 'fro') < 1e-14);

    % --- the two gauges -------------------------------------------------
    H1 = repmat({sx}, 1, tau);
    H2 = repmat({sz}, 1, tau);
    gj = qrobustness.multiparam.joint_gauge({H1, H2}, dt);
    ga = qrobustness.multiparam.angular_gauge({H1, H2}, dt);

    x = [0.6; -0.8];
    Cj = qrobustness.lengthspace.path_gauge_C(gj, x);
    Ca = qrobustness.lengthspace.path_gauge_C(ga, x);

    % Full-gauge dominance: C_FS(x) <= C_joint(x)/sqrt(N). Both structures
    % here are traceless, so the traceless projection changes nothing and
    % the normalisation is exactly the 1/sqrt(N).
    assert(Ca <= Cj / sqrt(N) + 1e-12);
    assert(abs(Ca - Cj / sqrt(N)) < 1e-12);

    % The directional Lipschitz constant never exceeds the separable one.
    Ljoint = qrobustness.multiparam.joint_gauge_L_dir(gj, x, FT, N);
    Csep = qrobustness.structure_constant('drift', sx, dt, tau);
    Lsep = dot([qrobustness.lipschitz_constant(FT, N, Csep); ...
                qrobustness.lipschitz_constant(FT, N, Csep)], abs(x));
    assert(Ljoint <= Lsep + 1e-12);

    % Angle budget is positive below the nominal and zero at it.
    assert(qrobustness.multiparam.angular_gauge_budget(0.9999, FT) > 0);
    assert(abs(qrobustness.multiparam.angular_gauge_budget(FT, FT)) < 1e-14);

    % --- directional margin, both stepping rules ------------------------
    % A synthetic fidelity falling monotonically along the ray, so the
    % true crossing is known in closed form.
    s_star = 0.02;
    fid = @(mu) 1 - (1 - FT) * (norm(mu, 2) / s_star)^2;
    d = [1; 0];

    r1 = qrobustness.multiparam.directional_margin(fid, [50; 50], FT, d, ...
        'eta', 1e-10);
    assert(r1.M > 0 && r1.M <= s_star + 1e-9);

    r2 = qrobustness.multiparam.directional_margin(fid, [50; 50], FT, d, ...
        'eta', 1e-10, 'angular_gauge', ga);
    assert(r2.M > 0 && r2.M <= s_star + 1e-9);

    % Both are certified lower bounds on the same crossing.
    assert(r1.M <= s_star + 1e-9);
    assert(r2.M <= s_star + 1e-9);
end

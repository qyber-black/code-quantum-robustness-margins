% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function test_timevarying()
%TEST_TIMEVARYING Trajectory certificates: uniform radius and the FS angle.

    sx = [0 1; 1 0];
    sz = [1 0; 0 -1];
    dt = 0.25;
    tau = 4;
    N = 2;
    FT = 0.999;
    F0 = 0.99999;

    % --- uniform (first Lipschitz) radius -------------------------------
    L = [3; 5];
    r0 = qrobustness.timevarying.uniform_margin(L, F0, FT);
    assert(abs(r0 - (F0 - FT) / sum(L)) < 1e-15);
    % F must exceed FT: there is no surplus to spend.
    assert_error(@() qrobustness.timevarying.uniform_margin(L, FT, FT), ...
                 'qrobustness:timevarying:surplus', 'no fidelity surplus');

    % --- the Fubini-Study certificate -----------------------------------
    Hh = repmat({sx}, 1, tau);
    Hh2 = repmat({sz}, 1, tau);
    m = qrobustness.timevarying.fs_margin(Hh, dt, F0, FT, r0);

    % Exact speed for a traceless structure: dt*tau*||H||_F/sqrt(N).
    expect = dt * tau * norm(sx, 'fro') / sqrt(N);
    assert(abs(m.speed - expect) < 1e-13);
    assert(abs(m.r_fs - (acos(FT) - acos(F0)) / m.speed) < 1e-13);
    assert(abs(m.r - max(r0, m.r_fs)) < 1e-15);

    % Dominance: the geometric certificate is never weaker than the
    % first-order one for the same structure (paper thm:dominance).
    C = qrobustness.structure_constant('drift', sx, dt, tau);
    Lstruct = qrobustness.lipschitz_constant(FT, N, C);
    r0_same = qrobustness.timevarying.uniform_margin(Lstruct, F0, FT);
    m2 = qrobustness.timevarying.fs_margin(Hh, dt, F0, FT, r0_same);
    assert(m2.r_fs >= r0_same - 1e-15);

    % A pure identity component is global phase and must cost nothing:
    % adding a multiple of I leaves the speed unchanged.
    Hid = cell(1, tau);
    for k = 1:tau
        Hid{k} = sx + 3.7 * eye(2);
    end
    mid = qrobustness.timevarying.fs_margin(Hid, dt, F0, FT);
    assert(abs(mid.speed - m.speed) < 1e-12);
    assert(abs(mid.r_fs - m.r_fs) < 1e-10);

    % The half-spread constant is the weaker one, so it is never smaller
    % than the exact Choi speed for these structures.
    assert(m.speed_halfspread >= m.speed - 1e-12);

    % Angles are consistent with the fidelities they came from.
    assert(abs(cos(m.theta_T) - FT) < 1e-14);
    assert(abs(cos(m.theta_0) - F0) < 1e-14);
    assert(m.theta_T > m.theta_0);

    % --- fs_margin_joint: the box budget over several structures -------
    % One structure reduces to the scalar certificate: the worst box
    % length at m equals m times the single-structure speed, so the
    % largest certified m is exactly r_fs.
    [ell1, budget1] = qrobustness.timevarying.fs_margin_joint({Hh}, dt, F0, FT);
    assert(abs(budget1 - (acos(FT) - acos(F0))) < 1e-13);
    assert(abs(ell1(1.0) - m.speed) < 1e-12);
    assert(abs(budget1 / ell1(1.0) - m.r_fs) < 1e-10);

    % Positive homogeneity of the box length.
    assert(abs(ell1(2.5) - 2.5 * ell1(1.0)) < 1e-12);
    assert(ell1(0) < 1e-14);

    % Two structures: the joint length is at most the sum of the separate
    % lengths (triangle inequality) and at least either alone.
    [ell2, ~] = qrobustness.timevarying.fs_margin_joint({Hh, Hh2}, dt, F0, FT);
    mm = [0.3; 0.7];
    [ellA, ~] = qrobustness.timevarying.fs_margin_joint({Hh}, dt, F0, FT);
    [ellB, ~] = qrobustness.timevarying.fs_margin_joint({Hh2}, dt, F0, FT);
    assert(ell2(mm) <= ellA(mm(1)) + ellB(mm(2)) + 1e-12);
    assert(ell2(mm) >= ellA(mm(1)) - 1e-12);
    assert(ell2(mm) >= ellB(mm(2)) - 1e-12);

    % Mixed structures: speed is additive over intervals.
    Hmix = {sx, sz, sx, sz};
    mm = qrobustness.timevarying.fs_margin(Hmix, dt, F0, FT);
    assert(abs(mm.speed - dt * (2 * norm(sx, 'fro') + 2 * norm(sz, 'fro')) ...
               / sqrt(N)) < 1e-13);
end

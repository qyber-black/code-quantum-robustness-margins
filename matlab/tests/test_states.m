% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function test_states()
%TEST_STATES State-fidelity margins: attained speed bound, small-angle accuracy and parity with the Python fixture.
%   The fixture is written by scripts/gen_states_parity.py.

    % Speed attained: equal superposition of the extreme eigenvectors of a structure that commutes with the drift.
    B = diag([-0.4, 0.3, 1.1]);
    psi0 = [1; 0; 1] / sqrt(2);
    C = qrobustness.states.state_speed(repmat({B}, 1, 4), 0.3);
    assert(abs(C - 4 * 0.3 * 0.75) < 1e-14);
    for mu = linspace(0, 0.9 * (pi / 2) / C, 7)
        psi = qrobustness.states.propagate_state(repmat({mu * B}, 1, 4), 0.3, psi0);
        assert(abs(qrobustness.states.fs_angle(psi0, psi) - C * mu) < 1e-12);
    end
    for theta = [1e-12, 1e-8, 1e-3, 1]
        phi = [cos(theta); sin(theta)];
        assert(abs(qrobustness.states.fs_angle([1; 0], phi) / theta - 1) < 1e-12);
    end
    assert_error(@() qrobustness.states.trajectory_radius(0, 1), 'qrobustness:states:budget', 'no budget');
    tr = qrobustness.states.trajectory_radius(0.4, 2);
    assert(abs(tr.r - 0.2) < 1e-15);

    % Parity with the Python reference.
    f = states_fixture();
    s = f.states;
    state = @(sv, mu) qrobustness.states.propagate_state(cellfun(@(h, a, b) h + sv * a + mu * b, f.H, f.A, f.B, ...
                                                          'UniformOutput', false), f.dt, f.psi0);
    rel = @(a, b) abs(a - b) <= 1e-9 * max(abs(b), 1e-12) + 1e-12;
    assert(rel(qrobustness.states.state_speed(f.B, f.dt), s.C));
    assert(rel(qrobustness.states.state_speed_joint({f.B, f.A}, f.dt, [0.7, -0.4]), s.C_joint));
    assert(rel(qrobustness.states.fs_angle(state(0, 0), state(0.3, 0.2)), s.fs_angle));
    target = state(0, 0.05);
    fid = @(mu) qrobustness.states.state_fidelity(target, state(0, mu));
    m = qrobustness.states.state_angular_margin(fid, s.C, s.FT, 'k_max', 20000);
    assert(rel(m.M_minus, s.state_margin(1)) && rel(m.M_plus, s.state_margin(2)));
    [psi_p, E_p, gap_p] = qrobustness.states.nondegenerate_eigenvector(f.H{1} + 0.2 * f.A{1}, 1);
    assert(rel(E_p, s.prep_E) && rel(gap_p, s.prep_gap));
    assert(rel(qrobustness.states.preparation_speed(f.A{1}, psi_p, gap_p), s.prep_sigma));
    assert(rel(qrobustness.states.preparation_speed(f.A{1}, psi_p, gap_p, 'spread'), s.prep_spread));
    assert(s.prep_sigma <= s.prep_spread);
    assert(rel(qrobustness.states.gapped_preparation_radius(0.1, gap_p, qrobustness.states.half_spread(f.A{1}), s.C), s.prep_radius));
    assert_error(@() qrobustness.states.nondegenerate_eigenvector(diag([0, 0, 1]), 1), 'qrobustness:states:degenerate', 'degenerate');
end

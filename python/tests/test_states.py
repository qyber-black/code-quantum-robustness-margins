# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""State-fidelity certificates (``qrobustness.states``): the speed
bound is attained, holds for static and time-varying perturbations, and the
certified margins never contain a point that violates the threshold."""

import numpy as np
import pytest

from qrobustness import states as st


def _herm(rng, d, scale):
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    return scale * (a + a.conj().T) / 2


def _system(rng, d=5, K=6, dt=0.4):
    H0 = [_herm(rng, d, 1.0) for _ in range(K)]
    A = [_herm(rng, d, 0.7) for _ in range(K)]
    B = [_herm(rng, d, 0.7) for _ in range(K)]
    psi0 = rng.normal(size=d) + 1j * rng.normal(size=d)
    return H0, A, B, psi0 / np.linalg.norm(psi0), dt


def _state(system, s, mu):
    H0, A, B, psi0, dt = system
    return st.propagate_state(
        [h + s * a + mu * b for h, a, b in zip(H0, A, B, strict=True)], dt, psi0
    )


def test_speed_bound_is_attained():
    """Equal superposition of the extreme eigenvectors of a structure that
    commutes with the drift: the angle is exactly C |mu| (a constant too
    small by any factor fails here)."""
    B = np.diag([-0.4, 0.3, 1.1])
    psi0 = np.array([1, 0, 1]) / np.sqrt(2)
    dt, K = 0.3, 4
    C = st.state_speed([B] * K, dt)
    assert C == pytest.approx(K * dt * 0.75)
    for mu in np.linspace(0.0, 0.9 * (np.pi / 2) / C, 7):
        psi = st.propagate_state([mu * B] * K, dt, psi0)
        assert st.fs_angle(psi0, psi) == pytest.approx(C * mu, abs=1e-12)


def test_speed_bounds_static_and_trajectory_perturbations():
    rng = np.random.default_rng(1)
    for _ in range(10):
        system = _system(rng)
        H0, A, B, psi0, dt = system
        C = st.state_speed(B, dt)
        nominal = _state(system, 0.0, 0.0)
        for mu in rng.uniform(-1.5, 1.5, size=5):
            assert (
                st.fs_angle(nominal, _state(system, 0.0, mu))
                <= C * abs(mu) * (1 + 1e-12) + 1e-14
            )
        m = 0.8
        for _ in range(5):
            traj = rng.uniform(-m, m, size=len(B))
            psi = st.propagate_state(
                [h + x * b for h, b, x in zip(H0, B, traj, strict=True)], dt, psi0
            )
            angle = st.fs_angle(nominal, psi)
            assert angle <= min(m * C, np.pi / 2) * (1 + 1e-12) + 1e-14


def test_joint_gauge_bounds_the_angle_and_its_relaxation():
    rng = np.random.default_rng(2)
    H0, A, B, psi0, dt = _system(rng)
    B2 = [_herm(rng, len(psi0), 0.5) for _ in B]
    Cs = [st.state_speed(B, dt), st.state_speed(B2, dt)]
    nominal = st.propagate_state(H0, dt, psi0)
    for x in rng.uniform(-1, 1, size=(8, 2)):
        gauge = st.state_speed_joint([B, B2], dt, x)
        assert gauge <= abs(x[0]) * Cs[0] + abs(x[1]) * Cs[1] + 1e-12
        psi = st.propagate_state(
            [h + x[0] * b + x[1] * c for h, b, c in zip(H0, B, B2, strict=True)],
            dt,
            psi0,
        )
        assert st.fs_angle(nominal, psi) <= gauge * (1 + 1e-12) + 1e-14
    assert st.state_speed_joint([B, B2], dt, [1.0, 0.0]) == pytest.approx(Cs[0])


def test_state_margins_certify_and_the_angular_radius_dominates():
    rng = np.random.default_rng(3)
    for _ in range(4):
        system = _system(rng)
        C = st.state_speed(system[2], system[4])
        target = _state(system, 0.0, 0.05)

        def F(mu, system=system, target=target):
            return st.state_fidelity(target, _state(system, 0.0, mu))

        FT = 0.6 * F(0.0)
        L = st.state_lipschitz_constant(FT, C)
        for value in np.linspace(FT, 1.0, 50):  # dominance, step by step
            assert (np.arccos(FT) - np.arccos(value)) / C >= (value - FT) / L - 1e-15
        for angular in (True, False):
            result = st.state_angular_margin(F, C, FT, angular=angular, k_max=20000)
            grid = np.linspace(-result.M_minus, result.M_plus, 801)
            assert min(F(mu) for mu in grid) >= FT - 1e-12


def _prep(rng, d=4):
    """Gapped preparation family H_prep(mu) = P0 + mu P1 and its ground state."""
    P0, P1 = _herm(rng, d, 1.0), _herm(rng, d, 0.4)

    def ground(mu):
        return st.nondegenerate_eigenvector(P0 + mu * P1, 0)

    return P1, ground


def test_preparation_speed_bounds_the_moving_ground_state():
    rng = np.random.default_rng(7)
    for _ in range(40):
        P1, ground = _prep(rng)
        mu, h = rng.uniform(-0.3, 0.3), 1e-6
        psi, _, gap = ground(mu)
        dpsi = (
            ground(mu + h)[0] * np.vdot(ground(mu + h)[0], psi)
            - ground(mu - h)[0] * np.vdot(ground(mu - h)[0], psi)
        ) / (2 * h)
        c_prep = np.linalg.norm(dpsi - np.vdot(psi, dpsi) * psi)
        sigma = st.preparation_speed(P1, psi, gap)
        assert c_prep <= sigma * (1 + 1e-5) + 1e-9
        assert sigma <= st.preparation_speed(P1, psi, gap, bound="spread") + 1e-15


def test_prepared_state_angle_includes_the_preparation_term():
    """Lemma (a) with a parameter-dependent preparation: the angle is at most
    the integral of C_prep + C, and the evolution term alone can be zero."""
    rng = np.random.default_rng(8)
    H0, _, B, _, dt = _system(rng, d=4)
    P1, ground = _prep(rng)
    C = st.state_speed(B, dt)
    grid = np.linspace(0.0, 0.4, 81)
    c_prep = [st.preparation_speed(P1, *ground(m)[::2]) for m in grid]
    for k in (20, 40, 80):
        bound = np.trapezoid(np.array(c_prep[: k + 1]) + C, grid[: k + 1])

        def psi(m):
            return st.propagate_state(
                [h + m * b for h, b in zip(H0, B, strict=True)], dt, ground(m)[0]
            )

        assert st.fs_angle(psi(0.0), psi(grid[k])) <= bound * (1 + 1e-3)
    # zero Hamiltonian structure: the state still moves, by the preparation alone
    moved = st.fs_angle(ground(0.0)[0], ground(0.4)[0])
    assert moved > 0
    assert moved <= np.trapezoid(c_prep, grid) * (1 + 1e-3)


def test_gapped_preparation_radius_keeps_the_prepared_state_safe():
    """The closed-form radius solves r (s / (gap - 2 s r) + C) = budget, and a
    preparation certified with it stays above the threshold."""
    for s, C, g, beta in [
        (0.3, 0.0, 1.0, 0.1),
        (0.3, 0.7, 0.5, 0.2),
        (0.0, 0.7, 1, 0.2),
    ]:
        r = st.gapped_preparation_radius(beta, g, s, C)
        assert r * (s / (g - 2 * s * r) + C) == pytest.approx(beta, rel=1e-12)
        assert 2 * s * r < g
    assert st.gapped_preparation_radius(0.1, 1.0, 0.0, 0.0) == np.inf
    rng = np.random.default_rng(9)
    for _ in range(20):
        P0, P1 = _herm(rng, 4, 1.0), _herm(rng, 4, 0.4)
        chi, _, gap = st.nondegenerate_eigenvector(P0, 0)
        fT = 0.95
        r = st.gapped_preparation_radius(np.arccos(fT), gap, st.half_spread(P1), 0.0)
        for mu in np.linspace(-r, r, 41):
            psi = st.nondegenerate_eigenvector(P0 + mu * P1, 0)[0]
            assert st.state_fidelity(chi, psi) >= fT - 1e-12


def test_degenerate_eigenvalue_is_refused():
    with pytest.raises(ValueError):
        st.nondegenerate_eigenvector(np.diag([0.0, 0.0, 1.0]), 0)
    with pytest.raises(ValueError):
        st.preparation_speed(np.eye(2), np.array([1.0, 0.0]), 0.0)


def test_angles_are_accurate_for_small_angles():
    psi = np.array([1.0, 0.0])
    for theta in (1e-12, 1e-8, 1e-3, 1.0):
        phi = np.array([np.cos(theta), np.sin(theta)])
        assert st.fs_angle(psi, phi) == pytest.approx(theta, rel=1e-12)


def test_angular_margin_uses_the_atan2_map_near_unity():
    psi = np.array([1.0, 0.0])
    phi = np.array([1.0, 1e-8])
    phi = phi / np.linalg.norm(phi)

    def fidelity(_mu: float) -> float:
        return float(abs(np.vdot(psi, phi)))

    result = st.state_angular_margin(fidelity, C=1.0, FT=0.999, angular=True)
    assert st.fs_angle(psi, phi) > 0
    assert result.M_plus == pytest.approx(result.M_minus)

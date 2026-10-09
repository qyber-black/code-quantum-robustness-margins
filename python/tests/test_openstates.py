# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Open-system state-fidelity certificates (``qrobustness.openstates``): the
trace-norm speed bounds static and time-varying rate perturbations, the
closed limit reduces to ``qrobustness.states``, and the certified margins contain no
violating point."""

import numpy as np
import pytest

from qrobustness import lindblad as lb
from qrobustness import openstates as os_
from qrobustness import states as st

SZ = np.diag([1.0, -1.0, 0.0]).astype(complex)
LOWER = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=complex)


def _herm(rng, d, scale):
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    return scale * (a + a.conj().T) / 2


def _system(rng, K=4, dt=0.5):
    """Qutrit: random coherent drive, dephasing rate gz0 + mu (the nuisance)
    and amplitude damping; signal s on a random Hamiltonian structure."""
    H = [_herm(rng, 3, 1.0) for _ in range(K)]
    A = _herm(rng, 3, 0.6)
    psi = rng.normal(size=3) + 1j * rng.normal(size=3)
    psi /= np.linalg.norm(psi)
    return H, A, np.outer(psi, psi.conj()), dt


def _rho(system, s, mu, gz0=0.3, gm=0.1):
    H, A, rho0, dt = system
    G = [lb.generator(h + s * A, [SZ, LOWER], [gz0 + mu, gm]) for h in H]
    return os_.evolve_density(G, dt, rho0)


def test_closed_limit_matches_the_state_speed():
    """||-i[B, .]||_diamond = lambda_max - lambda_min, checked from both sides
    without assuming it: the SDP optimum is an independent value, and the
    equal superposition of the extreme eigenvectors is an explicit input
    whose output trace norm is a lower bound on the diamond norm."""
    rng = np.random.default_rng(1)
    for d in (2, 3):
        for _ in range(3):
            B = _herm(rng, d, 0.5)
            G = lb.hamiltonian_superop(B)
            w, V = np.linalg.eigh(B)
            spread = w[-1] - w[0]
            psi = (V[:, 0] + V[:, -1]) / np.sqrt(2)
            out = (G @ np.outer(psi, psi.conj()).reshape(-1, order="F")).reshape(
                d, d, order="F"
            )
            witness = np.sum(np.abs(np.linalg.eigvalsh((out + out.conj().T) / 2)))
            assert witness == pytest.approx(spread, rel=1e-12)
            sdp = lb.diamond_norm(G)
            assert sdp.raw == pytest.approx(spread, rel=1e-6)
            exact = lb.hamiltonian_dnorm(G)
            assert exact.value_certified >= spread
            assert exact.value_certified <= spread * (1 + 1e-10)
            assert exact.value_certified <= sdp.value_certified
    # D/2 from the exact Hamiltonian norms is the state speed
    B = [_herm(rng, 3, 0.5) for _ in range(3)]
    supers = [lb.hamiltonian_superop(b) for b in B]
    C = st.state_speed(B, 0.4)
    D = os_.open_speed(supers, 0.4, exact_hamiltonian=True)
    assert D / 2 == pytest.approx(C, rel=1e-10)
    assert D / 2 >= C
    # the solver-free default is a verified upper bound of it, not tighter
    assert os_.open_speed(supers, 0.4) / 2 >= C * (1 - 1e-9)


def test_exact_hamiltonian_leaves_other_structures_to_the_norm():
    rng = np.random.default_rng(5)
    G_h = lb.hamiltonian_superop(_herm(rng, 3, 0.5))
    G_d = lb.dissipator(SZ)
    G_mix = G_h + 0.2 * G_d
    assert lb.hamiltonian_part(G_d) is None
    assert lb.hamiltonian_part(G_mix) is None
    with pytest.raises(ValueError):
        lb.hamiltonian_dnorm(G_d)
    calls = []

    def norm(G):
        calls.append(G)
        return 1.0

    os_.open_speed([G_h, G_d, G_mix], 1.0, norm=norm, exact_hamiltonian=True)
    assert len(calls) == 2  # the dissipator and the mixed generator


def test_trace_norm_speed_bounds_static_and_time_varying_rates():
    rng = np.random.default_rng(2)
    D = os_.open_speed([lb.dissipator(SZ)], 1.0) * 4 * 0.5  # K = 4 segments of 0.5
    for _ in range(6):
        system = _system(rng)
        H, A, rho0, dt = system
        nominal = _rho(system, 0.0, 0.0)
        for mu in rng.uniform(0.0, 1.0, size=4):
            assert 2 * os_.trace_distance(nominal, _rho(system, 0.0, mu)) <= D * mu * (
                1 + 1e-9
            )
        m = 0.4
        for _ in range(4):
            traj = rng.uniform(0.0, m, size=len(H))
            G = [
                lb.generator(h, [SZ, LOWER], [0.3 + x, 0.1])
                for h, x in zip(H, traj, strict=True)
            ]
            rho = os_.evolve_density(G, dt, rho0)
            assert 2 * os_.trace_distance(nominal, rho) <= D * m * (1 + 1e-9)


def test_margins_certify_the_segment():
    rng = np.random.default_rng(4)
    D = os_.open_speed([lb.dissipator(SZ)], 0.5 * 4)
    for _ in range(3):
        system = _system(rng)
        chi = np.linalg.eigh(_rho(system, 0.0, 0.0))[1][:, -1]

        def fid(mu, system=system, chi=chi):
            return float(np.real(chi.conj() @ _rho(system, 0.0, mu) @ chi))

        omega = (-0.3, np.inf)  # the dephasing rate 0.3 + mu stays non-negative
        r = os_.open_state_fidelity_margin(
            fid, D, 0.8 * fid(0.0), omega=omega, k_max=20000
        )
        grid = np.linspace(-r.M_minus, r.M_plus, 201)
        assert min(fid(mu) for mu in grid) >= 0.8 * fid(0.0) - 1e-12

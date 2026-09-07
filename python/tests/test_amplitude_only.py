# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The free certificates depend on control amplitude, never on shape.

Proposition "the free certificates see amplitude, not shape": every gauge
behind a zero-evaluation certificate is a sum over intervals of a function
of the per-interval amplitude magnitudes, so permuting the intervals or
flipping their signs cannot change it. The nominal fidelity is not
invariant, which is the point: static robustification works precisely by
arranging what these certificates cannot see, so it improves the iterated
margin while leaving them alone or, when it spends amplitude, worsening
them.

The paper rests its explanation of the uncertainty-class reversal on this,
so it is asserted rather than described. The scrambling checks keep the
test from passing vacuously: a certificate that had come to depend on
shape would break the equalities, and one that had stopped depending on the
controller at all would be caught by the nominal fidelity collapsing while
the margins did not move.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qrobustness import (
    dH_structure,
    gate_fidelity,
    load_controllers,
    load_problem,
    multiparam as mp,
    propagator,
)
from qrobustness.timevarying import fs_margin

ROOT = Path(__file__).resolve().parents[2]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"

FT = 0.999
#: Reordering a floating sum is not free. The invariance is exact in the
#: reals; this is the room the summation order costs.
ROUNDOFF = 1e-13
#: The scrambled gates must actually be ruined, or the invariance is not
#: being tested against anything.
SCRAMBLED_MAX_FIDELITY = 0.5
N_CONTROLLERS = 4


@pytest.fixture(scope="module")
def ensemble():
    problem = load_problem(CTRL / "problem9.mat")
    return problem, load_controllers(CTRL / "controllers.csv", 1e-4)[:N_CONTROLLERS]


def _gauges(problem, c, u1, u2):
    """The three free quantities, and the nominal fidelity for contrast."""
    dt = c["tf"] / c["tau"]
    dHs = [
        dH_structure(problem["H0"], problem["H1"], problem["H2"], u1, u2, t)
        for t in ("H0", "H1", "H2")
    ]
    speeds = np.array([fs_margin(dH, dt, c["fid"], FT).speed for dH in dHs])
    spec = [
        ("drift", problem["H0"]),
        ("control", problem["H1"], u1),
        ("control", problem["H2"], u2),
    ]
    _, L = mp.structure_constants(spec, dt, c["tau"], FT, problem["dim"])
    d = np.ones(3) / np.sqrt(3.0)
    radius = mp.joint_gauge(dHs, dt).boundary_radius(
        d, c["fid"] - FT, FT, problem["dim"]
    )
    H = [
        problem["H0"] + u1[k] * problem["H1"] + u2[k] * problem["H2"]
        for k in range(c["tau"])
    ]
    fid = gate_fidelity(propagator(H, dt), problem["Uf"])
    return speeds, np.asarray(L), radius, fid


@pytest.mark.parametrize("scramble", ["permute", "flip_signs"])
def test_scrambling_the_control_leaves_every_free_gauge_unchanged(ensemble, scramble):
    """Reorder or sign-flip the intervals: the gauges do not move, the gate does."""
    problem, controllers = ensemble
    rng = np.random.default_rng(0)
    for c in controllers:
        s0, L0, r0, fid0 = _gauges(problem, c, c["u1"], c["u2"])
        if scramble == "permute":
            idx = rng.permutation(c["tau"])
            u1, u2 = c["u1"][idx], c["u2"][idx]
        else:
            sgn = rng.choice([-1.0, 1.0], size=c["tau"])
            u1, u2 = c["u1"] * sgn, c["u2"] * sgn
        s1, L1, r1, fid1 = _gauges(problem, c, u1, u2)

        # Exact in exact arithmetic, since each term depends on |u| alone
        # and only the order of the sum over intervals changes. In binary64
        # that reordering is worth a couple of ulp -- the speeds and the
        # Lipschitz constants happen to come back bit-identical, the gauge
        # radius to within 2e-16 relative -- so the assertion is a tight
        # tolerance rather than equality. Anything shape-dependent would
        # miss by orders of magnitude, as the fidelity below shows.
        assert s1 == pytest.approx(s0, rel=ROUNDOFF)
        assert L1 == pytest.approx(L0, rel=ROUNDOFF)
        assert r1 == pytest.approx(r0, rel=ROUNDOFF)
        # ... while the gate the certificate was about is destroyed.
        assert fid0 > FT
        assert fid1 < SCRAMBLED_MAX_FIDELITY


def test_the_control_speed_is_the_pulse_area(ensemble):
    """s_j = dt ||u_j||_1 ||H_j||_F / sqrt(N) for a multiplicative structure,
    and dt * tau * ||H_0||_F / sqrt(N) for the additive drift, so the drift
    speed is one number for the whole ensemble."""
    problem, controllers = ensemble
    N = problem["dim"]
    norms = [
        float(np.linalg.norm(H - np.trace(H) / N * np.eye(N), "fro"))
        for H in (problem["H0"], problem["H1"], problem["H2"])
    ]
    drift = set()
    for c in controllers:
        dt = c["tf"] / c["tau"]
        s, _, _, _ = _gauges(problem, c, c["u1"], c["u2"])
        want = np.array(
            [
                dt * c["tau"] * norms[0],
                dt * float(np.abs(c["u1"]).sum()) * norms[1],
                dt * float(np.abs(c["u2"]).sum()) * norms[2],
            ]
        ) / np.sqrt(N)
        assert s == pytest.approx(want, rel=1e-13)
        drift.add(round(float(s[0]), 9))
    assert len(drift) == 1, f"drift speed should not vary with the control: {drift}"


def test_the_toggling_frame_integral_matches_quadrature():
    """The exact divided-difference form against a fine trapezoid.

    The integral is what a static robustification objective drives down and
    what the free certificates cannot see, so the paper measures it. Its
    closed form is exact for piecewise-constant controls; this checks that
    against brute force on a random instance, where the residual is the
    quadrature's own error and not the formula's.
    """
    from scipy.linalg import expm

    from qrobustness.timevarying import toggling_frame_integral

    rng = np.random.default_rng(0)
    n, k, dt, steps = 4, 5, 0.3, 2001

    def herm():
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return a + a.conj().T

    H = [herm() for _ in range(k)]
    dH = [herm() for _ in range(k)]

    want = np.zeros((n, n), dtype=complex)
    U = np.eye(n, dtype=complex)
    grid = np.linspace(0.0, dt, steps)
    for Hk, dHk in zip(H, dH, strict=True):
        d = dHk - np.trace(dHk) / n * np.eye(n)
        acc = np.zeros((n, n), dtype=complex)
        for i, t in enumerate(grid):
            E = expm(1j * Hk * t)
            acc += (0.5 if i in (0, steps - 1) else 1.0) * (E @ d @ E.conj().T)
        want += U.conj().T @ (acc * dt / (steps - 1)) @ U
        U = expm(-1j * Hk * dt) @ U

    got = toggling_frame_integral(H, dH, dt)
    assert np.abs(got - want).max() < 1e-6


def test_the_toggling_frame_integral_is_shape_sensitive(ensemble):
    """Unlike the gauges, it does see temporal order -- which is the point."""
    from qrobustness.timevarying import toggling_frame_integral

    problem, controllers = ensemble
    c = controllers[0]
    dt = c["tf"] / c["tau"]
    H = [
        problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
        for k in range(c["tau"])
    ]
    dH = dH_structure(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], "H1"
    )
    base = np.linalg.norm(toggling_frame_integral(H, dH, dt), "fro")
    idx = np.random.default_rng(1).permutation(c["tau"])
    shuffled = np.linalg.norm(
        toggling_frame_integral([H[i] for i in idx], [dH[i] for i in idx], dt), "fro"
    )
    assert not np.isclose(base, shuffled, rtol=1e-3)

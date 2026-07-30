# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for GRAPE fidelity gradient and controller synthesis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from qrobustness import (
    fidelity_and_gradient,
    gate_fidelity,
    load_controllers,
    load_problem,
    optimize_controller,
    pack_controls,
    perturbed_hamiltonians,
    propagator,
    unpack_controls,
)

ROOT = Path(__file__).resolve().parents[2]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"


def test_pack_unpack_roundtrip():
    tau = 5
    u1 = np.linspace(-1, 1, tau)
    u2 = np.linspace(2, -2, tau)
    x = pack_controls(u1, u2)
    assert x.shape == (2 * tau,)
    # Fortran interleave: u1[0], u2[0], u1[1], u2[1], ...
    assert abs(x[0] - u1[0]) < 1e-15
    assert abs(x[1] - u2[0]) < 1e-15
    a, b = unpack_controls(x, tau)
    np.testing.assert_allclose(a, u1)
    np.testing.assert_allclose(b, u2)


def test_fidelity_matches_propagator():
    problem = load_problem(CTRL / "problem9.mat")
    ctrls = load_controllers(CTRL / "controllers.csv", 1e-4)
    c = ctrls[0]
    dt = c["tf"] / c["tau"]
    F, g1, g2 = fidelity_and_gradient(
        problem["H0"],
        problem["H1"],
        problem["H2"],
        c["u1"],
        c["u2"],
        problem["Uf"],
        dt,
    )
    H_list = perturbed_hamiltonians(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], "H0", 0.0
    )
    Fref = gate_fidelity(propagator(H_list, dt), problem["Uf"])
    assert abs(F - Fref) < 1e-12
    assert g1.shape == c["u1"].shape
    assert g2.shape == c["u2"].shape


@pytest.mark.parametrize("method", ["exact", "quadrature"])
def test_gradient_finite_difference(method):
    problem = load_problem(CTRL / "problem9.mat")
    rng = np.random.default_rng(0)
    tau = 4
    tf = 2.0
    dt = tf / tau
    u1 = rng.normal(size=tau)
    u2 = rng.normal(size=tau)

    def fg(a, b):
        return fidelity_and_gradient(
            problem["H0"],
            problem["H1"],
            problem["H2"],
            a,
            b,
            problem["Uf"],
            dt,
            n_quad=48,
            method=method,
        )

    F, g1, g2 = fg(u1, u2)
    h = 1e-7
    # FD one component of u1 and one of u2
    k = 1
    u1p = u1.copy()
    u1p[k] += h
    Fp, _, _ = fg(u1p, u2)
    u1m = u1.copy()
    u1m[k] -= h
    Fm, _, _ = fg(u1m, u2)
    g1_fd = (Fp - Fm) / (2 * h)
    assert abs(g1[k] - g1_fd) / max(1e-8, abs(g1_fd)) < 5e-4

    u2p = u2.copy()
    u2p[k] += h
    Fp, _, _ = fg(u1, u2p)
    u2m = u2.copy()
    u2m[k] -= h
    Fm, _, _ = fg(u1, u2m)
    g2_fd = (Fp - Fm) / (2 * h)
    assert abs(g2[k] - g2_fd) / max(1e-8, abs(g2_fd)) < 5e-4


def test_optimize_improves_fidelity():
    problem = load_problem(CTRL / "problem9.mat")
    rng = np.random.default_rng(42)
    tau = 32
    u1 = rng.normal(size=tau)
    u2 = rng.normal(size=tau)
    res = optimize_controller(
        problem["H0"],
        problem["H1"],
        problem["H2"],
        problem["Uf"],
        tf=15.0,
        tau=tau,
        u1_init=u1,
        u2_init=u2,
        maxiter=80,
        ftol=1e-10,
    )
    assert res.fid > res.fid_init
    assert res.error == 1.0 - res.fid


def test_csv_roundtrip_synth_smoke(tmp_path):
    """Write one optimised controller and reload via load_controllers."""

    problem = load_problem(CTRL / "problem9.mat")
    res = optimize_controller(
        problem["H0"],
        problem["H1"],
        problem["H2"],
        problem["Uf"],
        tf=15.0,
        tau=32,
        seed=7,
        maxiter=40,
    )
    # Bypass filter by writing a tiny error if needed for roundtrip of pulses
    x = pack_controls(res.u1, res.u2)
    row = [9, 1, 15, 32, res.error] + list(x)
    csv_path = tmp_path / "controllers.csv"
    with csv_path.open("w") as f:
        f.write(",".join(repr(float(v)) if i >= 4 else str(v) for i, v in enumerate(row)))
        f.write("\n")
    # load with loose filter
    loaded = load_controllers(csv_path, max_error=1.0)
    assert len(loaded) == 1
    np.testing.assert_allclose(loaded[0]["u1"], res.u1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(loaded[0]["u2"], res.u2, rtol=0, atol=1e-12)

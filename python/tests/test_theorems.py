# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Numerical verification of every theorem, via the verify harness.

Fast seeded instances; the ensemble-wide sweep is
scripts/run_theorem_verification.py.  One test per theorem or lemma of
the accompanying paper, on up to three systems (single-qubit pi-pulse,
the shipped three-qubit ensemble, and, where cheap, a two-qubit CNOT).
"""

from pathlib import Path

import numpy as np
import pytest

from qrobustness import (
    iterative_margin,
    lipschitz_constant,
    load_controllers,
    load_problem,
    structure_constant,
)
from qrobustness.core import gate_fidelity, propagator
from qrobustness import multiparam as mp
from qrobustness import verify
from qrobustness.timevarying import fs_margin, uniform_margin

ROOT = Path(__file__).resolve().parents[2]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
FT = 0.999

SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SZ = np.diag([1.0, -1.0]).astype(complex)


@pytest.fixture(scope="module")
def qubit():
    """Single-qubit pi-pulse with amplitude and detuning structures."""
    tau, T = 8, 1.0
    H = 0.5 * np.pi * SX
    from scipy.linalg import expm

    return {
        "H_list": [H] * tau,
        "dt": T / tau,
        "Uf": expm(-1j * T * H),
        "amp": [0.5 * np.pi * SX] * tau,
        "det": [0.5 * SZ] * tau,
    }


@pytest.fixture(scope="module")
def chain3():
    problem = load_problem(CTRL / "problem9.mat")
    c = load_controllers(CTRL / "controllers.csv", 1e-4)[0]
    dt = c["tf"] / c["tau"]
    H_list = [
        problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
        for k in range(c["tau"])
    ]
    dHs = [
        [problem["H0"]] * c["tau"],
        [c["u1"][k] * problem["H1"] for k in range(c["tau"])],
        [c["u2"][k] * problem["H2"] for k in range(c["tau"])],
    ]
    specs = [
        ("drift", problem["H0"]),
        ("control", problem["H1"], c["u1"]),
        ("control", problem["H2"], c["u2"]),
    ]
    _, L = mp.structure_constants(specs, dt, c["tau"], FT, problem["dim"])
    return {
        "problem": problem,
        "c": c,
        "dt": dt,
        "H_list": H_list,
        "dHs": dHs,
        "L": np.asarray(L),
        "Uf": problem["Uf"],
        "fid": c["fid"],
    }


def test_lem_angle_triangle_inequality():
    """Lemma (angle metric): arccos of the trace-amplitude fidelity obeys
    the triangle inequality, sampled over random triples in three
    dimensions. Everything angular in the paper rests on this."""
    for N in (2, 4, 8):
        rep = verify.check_metric_triangle(N, n_triples=150, seed=N)
        assert rep.passed, rep


def test_lem_absorption_sufficiency():
    """Lemma (absorption): the angular correction for a nominal error is
    sufficient -- a gate meeting the corrected threshold meets the real
    one."""
    for N in (2, 8):
        rep = verify.check_absorption(FT, N, n=150, seed=N)
        assert rep.passed, rep


def test_alg1_constant_margin(qubit):
    """Algorithm 1's certificate on the single-qubit model: the fidelity
    stays above the threshold everywhere inside the reported margin, for
    both structures.

    The slack allowed is the eta band the iteration terminates on; anything
    worse than that is a genuine violation, not a tolerance question.
    """
    for tag in ("amp", "det"):
        dH = qubit[tag]
        C = structure_constant("control", dH[0], qubit["dt"], len(dH), np.ones(len(dH)))
        L = lipschitz_constant(FT, 2, C)

        def fid_fn(mu, dH=dH):
            return gate_fidelity(
                propagator(
                    [qubit["H_list"][k] + mu * dH[k] for k in range(len(dH))],
                    qubit["dt"],
                ),
                qubit["Uf"],
            )

        M = float(iterative_margin(fid_fn, L, FT, mu0=0.0, eta=1e-6).M)
        rep = verify.check_constant_margin(fid_fn, M, FT, n=101, tol=2e-6)
        # eta terminates on a fidelity band, so slack down to -eta is by
        # design; anything worse is a genuine violation.
        assert rep.passed, rep


def test_thm_polytope(chain3):
    """Theorem (safe polytope): every sampled point of the certified
    cross-polytope meets the threshold on the three-qubit chain."""
    tau = len(chain3["H_list"])

    def fn(mu):
        Hp = [
            chain3["H_list"][k] + sum(mu[j] * chain3["dHs"][j][k] for j in range(3))
            for k in range(tau)
        ]
        return gate_fidelity(propagator(Hp, chain3["dt"]), chain3["Uf"])

    rep = verify.check_polytope(fn, chain3["L"], chain3["fid"], FT, n=60, seed=1)
    assert rep.passed, rep


def test_lem_tv_lipschitz(chain3):
    """Lemma (trajectory Lipschitz): the fidelity gap between two
    trajectories is bounded by the constants times their separation,
    checked on sampled pairs and homotopies."""
    m = 0.5 * float((chain3["fid"] - FT) / chain3["L"].sum())
    rep = verify.check_lipschitz_pairs(
        chain3["H_list"],
        chain3["dHs"],
        chain3["dt"],
        chain3["Uf"],
        chain3["L"],
        FT,
        m,
        n=40,
        seed=2,
    )
    assert rep.passed, rep


def test_lem_tv_slope(chain3):
    """The slope form of the trajectory Lipschitz lemma.

    The module docstring promises a fast seeded instance of every check here;
    this one ran only in the ensemble sweep, so a regression in it would have
    surfaced only during a full verification run.
    """
    m = 0.5 * float((chain3["fid"] - FT) / chain3["L"].sum())
    rep = verify.check_tv_slope(
        chain3["H_list"],
        chain3["dHs"],
        chain3["dt"],
        chain3["Uf"],
        chain3["L"],
        FT,
        m,
        n=24,
        n_homotopy=2,
        seed=3,
    )
    assert rep.passed, rep


def test_thm_fs_angle_bound(qubit, chain3):
    """Theorem (Choi-FS trajectory bound): the accrued Choi angle never
    exceeds speed times budget, on both models and under refinement."""
    for dH_key in ("amp", "det"):
        dH = qubit[dH_key]
        fs = fs_margin(dH, qubit["dt"], 1.0, FT)
        rep = verify.check_fs_angle(
            qubit["H_list"], dH, qubit["dt"], fs.r_fs, fs.speed, n=60, seed=3
        )
        assert rep.passed, rep
    dH = chain3["dHs"][1]
    fs = fs_margin(dH, chain3["dt"], chain3["fid"], FT)
    rep = verify.check_fs_angle(
        chain3["H_list"], dH, chain3["dt"], fs.r_fs, fs.speed, n=20, refine=4, seed=4
    )
    assert rep.passed, rep


def test_thm_tv_and_fs_margins_hold(chain3):
    """Both uniform trajectory certificates survive the adversary."""
    dH = chain3["dHs"][1]
    r0 = uniform_margin(chain3["L"][1], chain3["fid"], FT)
    fs = fs_margin(dH, chain3["dt"], chain3["fid"], FT, r0=r0)
    rep = verify.check_trajectory_certificate(
        chain3["H_list"],
        dH,
        chain3["dt"],
        chain3["Uf"],
        FT,
        fs.r,
        refinements=(1, 4),
        n_starts=2,
        maxiter=100,
        n_random=25,
        seed=5,
    )
    assert rep.passed, rep


def test_fidelity_cross_check_tightness(chain3):
    """The two independent fidelity routes agree with the ensemble's
    recorded value and with each other to near roundoff."""
    F, tol = verify.fidelity_cross_check(chain3["H_list"], chain3["dt"], chain3["Uf"])
    assert F == pytest.approx(chain3["fid"], abs=1e-9)
    assert tol < 1e-10  # both routes agree to near roundoff


def test_thm_fs_choi_speed_is_exact(chain3):
    """The Choi-state FS speed equals ||Gbar||_F/sqrt(N) exactly: compare
    the angle accrued over a short evolution against the formula."""
    from scipy.linalg import expm

    dH = chain3["dHs"][1]
    N = chain3["problem"]["dim"]
    H = np.asarray(chain3["H_list"][0])
    G = np.asarray(dH[0])
    Gbar = G - np.trace(G) / N * np.eye(N)
    v_exact = np.linalg.norm(Gbar, "fro") / np.sqrt(N)
    # Angle between U_S(s) and U(s) for small s at unit delta: theta ~ v s.
    s = 1e-5
    US = expm(-1j * s * H)
    U = expm(-1j * s * (H + G))
    ang = np.arccos(min(1.0, abs(np.trace(US.conj().T @ U)) / N))
    assert ang / s == pytest.approx(v_exact, rel=1e-4)

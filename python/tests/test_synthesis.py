# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GRAPE synthesis: exact gradients, convergence, reproducibility."""

import numpy as np
import pytest

from qrobustness.synthesis import (
    fidelity_and_control_gradient,
    grape,
    grape_ensemble,
)

Z = np.diag([1.0, -1.0])
X = np.array([[0.0, 1.0], [1.0, 0.0]])
I2 = np.eye(2)
ZZ = np.kron(Z, Z)
X1 = np.kron(X, I2)
X2 = np.kron(I2, X)
H0 = np.pi * ZZ + 0.2 * np.pi * (np.kron(Z, I2) - np.kron(I2, Z))
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def test_control_gradient_matches_finite_differences():
    """The GRAPE gradient matches central differences in every control
    amplitude; a wrong gradient would still converge, just worse."""
    rng = np.random.default_rng(3)
    tau, dt = 6, 0.4
    u = rng.standard_normal((2, tau))
    F, g = fidelity_and_control_gradient(H0, [X1, X2], u, dt, CNOT)
    h = 1e-6
    for j in range(2):
        for k in range(tau):
            up, um = u.copy(), u.copy()
            up[j, k] += h
            um[j, k] -= h
            Fp, _ = fidelity_and_control_gradient(H0, [X1, X2], up, dt, CNOT)
            Fm, _ = fidelity_and_control_gradient(H0, [X1, X2], um, dt, CNOT)
            assert (Fp - Fm) / (2 * h) == pytest.approx(g[j, k], abs=5e-9)


def test_grape_converges_on_cnot():
    """Synthesis reaches a CNOT on the two-qubit model to 1e-8 error, the
    quality the case studies assume of their ensembles."""
    r = grape(H0, [X1, X2], CNOT, tf=4.0, tau=20, seed=0)
    assert r.error < 1e-8


def test_grape_is_seed_reproducible():
    """The same seed gives bit-identical controls: every ensemble in the
    repository is reproduced from its seeds alone."""
    r1 = grape(H0, [X1, X2], CNOT, tf=4.0, tau=20, seed=5, maxiter=60)
    r2 = grape(H0, [X1, X2], CNOT, tf=4.0, tau=20, seed=5, maxiter=60)
    assert np.array_equal(r1.u, r2.u)
    assert r1.fidelity == r2.fidelity


def test_grape_ensemble_filters_by_error():
    """The ensemble keeps only attempts under max_error and records the seed
    each survivor came from, in attempt order."""
    ens = grape_ensemble(
        H0, [X1, X2], CNOT, tf=4.0, tau=20, n_attempts=3, max_error=1e-6, seed0=10
    )
    assert len(ens) >= 1
    assert all(r.error <= 1e-6 for r in ens)
    assert all(
        ens[i].seed == 10 + j
        for i, j in zip(range(len(ens)), [r.seed - 10 for r in ens], strict=True)
    )

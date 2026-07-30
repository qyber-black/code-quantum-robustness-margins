# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exact closed-form segment derivative vs finite differences and quadrature.

For piecewise-constant controls dU/dmu is exact in the eigenbasis of the
interval Hamiltonian, so the Gauss-Legendre path is only an approximation of
what qrobustness.core._dU_dmu_exact computes in closed form.  These tests pin
the closed form down directly (finite differences, commuting case, degenerate
spectra) and then assert that the two paths agree on the real case-study data.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from qrobustness import DU_METHODS, dH_structure, differential_sensitivity, load_controllers, load_problem
from qrobustness.core import _dU_dmu_exact, _segment_eig, _segment_propagator
from qrobustness.optimize import fidelity_and_gradient

ROOT = Path(__file__).resolve().parents[2]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"

DT = 0.4688


def _herm(rng, n):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return 0.5 * (A + A.conj().T)


def _cases():
    rng = np.random.default_rng(20260730)
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    degenerate = Q @ np.diag([1.0, 1.0, 1.0, 2.0]).astype(complex) @ Q.conj().T
    return {
        "generic": _herm(rng, 8),
        "degenerate": 0.5 * (degenerate + degenerate.conj().T),
        "zero": np.zeros((4, 4), dtype=complex),
        "scalar": 3.0 * np.eye(5, dtype=complex),
    }


@pytest.mark.parametrize("name", sorted(_cases()))
def test_exact_matches_central_difference(name):
    """Central difference in mu, at the roundoff floor of eps=1e-6."""
    H = _cases()[name]
    rng = np.random.default_rng(7)
    dH = _herm(rng, H.shape[0])
    lam, V = _segment_eig(H)
    dU = _dU_dmu_exact(lam, V, dH, DT)

    eps = 1e-6
    fd = (expm(-1j * DT * (H + eps * dH)) - expm(-1j * DT * (H - eps * dH))) / (2 * eps)
    assert np.linalg.norm(dU - fd) / np.linalg.norm(dU) < 1e-8


@pytest.mark.parametrize("name", sorted(_cases()))
def test_exact_finite_and_propagator_consistent(name):
    """No NaN from the X == 0 mask, and V,lam reproduce expm."""
    H = _cases()[name]
    rng = np.random.default_rng(11)
    dH = _herm(rng, H.shape[0])
    lam, V = _segment_eig(H)

    assert np.all(np.isfinite(_dU_dmu_exact(lam, V, dH, DT)))
    assert np.linalg.norm(V.conj().T @ V - np.eye(H.shape[0])) < 1e-13
    assert np.linalg.norm(_segment_propagator(lam, V, DT) - expm(-1j * DT * H)) < 1e-12


def test_exact_commuting_case_is_exact():
    """dH = H commutes, so dU/dmu = -1j*dt*H*expm(-1j*dt*H) analytically."""
    rng = np.random.default_rng(3)
    H = _herm(rng, 6)
    lam, V = _segment_eig(H)
    dU = _dU_dmu_exact(lam, V, H, DT)
    ref = -1j * DT * H @ expm(-1j * DT * H)
    assert np.linalg.norm(dU - ref) / np.linalg.norm(ref) < 1e-13


def test_unknown_method_rejected():
    rng = np.random.default_rng(5)
    H = _herm(rng, 4)
    assert DU_METHODS == ("exact", "quadrature")
    with pytest.raises(ValueError, match="Unknown method"):
        differential_sensitivity([H], [H], DT, np.eye(4), method="simpson")
    with pytest.raises(ValueError, match="Unknown method"):
        fidelity_and_gradient(H, H, H, [0.1], [0.2], np.eye(4), DT, method="simpson")


@pytest.mark.parametrize("structure", ["H0", "H1", "H2"])
def test_zeta_exact_matches_quadrature_on_case_study(structure):
    """The two paths agree on real data; measured worst case is ~3e-9."""
    problem = load_problem(CTRL / "problem9.mat")
    controllers = load_controllers(CTRL / "controllers.csv")[:12]
    H0, H1, H2, Uf = problem["H0"], problem["H1"], problem["H2"], problem["Uf"]

    for c in controllers:
        dt = c["tf"] / c["tau"]
        H_list = [H0 + c["u1"][k] * H1 + c["u2"][k] * H2 for k in range(c["tau"])]
        dH_list = dH_structure(H0, H1, H2, c["u1"], c["u2"], structure)
        z_exact = differential_sensitivity(H_list, dH_list, dt, Uf, method="exact")
        z_quad = differential_sensitivity(H_list, dH_list, dt, Uf, 48, method="quadrature")
        assert z_exact == pytest.approx(z_quad, rel=1e-7, abs=1e-12)


def test_gradient_exact_matches_quadrature_on_case_study():
    problem = load_problem(CTRL / "problem9.mat")
    c = load_controllers(CTRL / "controllers.csv")[0]
    H0, H1, H2, Uf = problem["H0"], problem["H1"], problem["H2"], problem["Uf"]
    dt = c["tf"] / c["tau"]

    Fe, g1e, g2e = fidelity_and_gradient(H0, H1, H2, c["u1"], c["u2"], Uf, dt, method="exact")
    Fq, g1q, g2q = fidelity_and_gradient(
        H0, H1, H2, c["u1"], c["u2"], Uf, dt, 48, method="quadrature"
    )
    assert Fe == pytest.approx(Fq, rel=1e-12, abs=1e-14)
    assert g1e == pytest.approx(g1q, rel=1e-7, abs=1e-12)
    assert g2e == pytest.approx(g2q, rel=1e-7, abs=1e-12)


def test_n_quad_is_inert_under_exact():
    """A positional n_quad must not select quadrature, nor change the exact path."""
    problem = load_problem(CTRL / "problem9.mat")
    c = load_controllers(CTRL / "controllers.csv")[0]
    H0, H1, H2, Uf = problem["H0"], problem["H1"], problem["H2"], problem["Uf"]
    dt = c["tf"] / c["tau"]
    H_list = [H0 + c["u1"][k] * H1 + c["u2"][k] * H2 for k in range(c["tau"])]
    dH_list = dH_structure(H0, H1, H2, c["u1"], c["u2"], "H1")

    z4 = differential_sensitivity(H_list, dH_list, dt, Uf, 4)
    z32 = differential_sensitivity(H_list, dH_list, dt, Uf, 32)
    assert z4 == z32

    # 4-node quadrature is inaccurate here, so the equality above is a real
    # assertion that the positional argument did not change the exact path.
    z4_quad = differential_sensitivity(H_list, dH_list, dt, Uf, 4, method="quadrature")
    assert abs(z4_quad - z32) > 1e-12 * abs(z32)

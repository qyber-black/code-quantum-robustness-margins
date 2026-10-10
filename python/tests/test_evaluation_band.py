# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The evaluation band, zero-gauge directions and per-ray evaluation counts
of the continuation (iterative_margin, directional_margin)."""

import numpy as np
import pytest

from qrobustness import iterative_margin
from qrobustness import multiparam as mp
from qrobustness.core import gate_fidelity, propagator

BIAS = 1e-4


def _biased(mu):
    """A fidelity evaluated within BIAS of the truth 1 - |mu|, on the high side."""
    return 1.0 - abs(mu) + BIAS


@pytest.mark.parametrize("margin_tol", [None, 1e-8])
def test_band_keeps_the_prefix_below_the_true_crossing(margin_tol):
    """With eval_tol equal to the evaluation error, no point whose true
    fidelity is below F_T is certified: the prefix stays below 0.5."""
    r = iterative_margin(_biased, 1.0, 0.5, eval_tol=BIAS, margin_tol=margin_tol)
    assert r.M_plus <= 0.5
    assert r.M_minus <= 0.5


def test_upper_edge_of_the_band_is_not_promoted():
    """F = F_T + eval_tol lies in the closed band, so the continuation does not certify it."""
    FT = 0.5
    tol = 0.1

    def fn(mu):
        if abs(mu) <= 1.0:
            return 0.9
        return FT + tol

    r = iterative_margin(fn, 1.0, FT, eval_tol=tol, margin_tol=None, omega=(-5.0, 5.0))
    assert r.M_plus <= 1.0
    assert r.M_minus <= 1.0
    # No safe point lies beyond 1: continuation stops there instead of
    # repeating the same step until k_max.
    assert r.status_plus == "stalled"


def test_without_band_the_biased_evaluation_overshoots():
    """The same run without the band certifies past the true crossing, which
    is what the band is for."""
    r = iterative_margin(_biased, 1.0, 0.5)
    assert r.M_plus > 0.5


def test_zero_band_is_unchanged():
    def f(mu):
        return float(np.cos(mu) ** 2)

    a = iterative_margin(f, 2.0, 0.9, margin_tol=1e-8)
    b = iterative_margin(f, 2.0, 0.9, margin_tol=1e-8, eval_tol=0.0)
    assert (a.M_minus, a.M_plus, a.M_upper) == (b.M_minus, b.M_plus, b.M_upper)


def test_per_ray_evaluation_counts_add_up():
    def f(mu):
        return float(np.cos(mu) ** 2)

    r = iterative_margin(f, 2.0, 0.9, margin_tol=1e-8, return_diagnostics=True)
    assert r.n_evals_minus > 0 and r.n_evals_plus > 0
    assert r.n_evals_minus + r.n_evals_plus + 1 == r.n_evals


def test_zero_angular_gauge_certifies_the_ray_whatever_L_dir():
    """Two identical structures along (1, -1)/sqrt(2) cancel on every
    interval: the angular gauge is zero although the separable Lipschitz
    constant is not, and the whole admissible ray is certified."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.diag([1.0, -1.0]).astype(complex)
    K, dt = 4, 0.25
    H = [np.pi / 2 * X] * K
    struct = [[Z] * K, [Z] * K]
    Uf = propagator(H, dt)

    def fid(x):
        return gate_fidelity(propagator([h + x[0] * Z + x[1] * Z for h in H], dt), Uf)

    d = np.array([1.0, -1.0]) / np.sqrt(2)
    A = mp.angular_gauge(struct, dt)
    assert A.C(d) == pytest.approx(0.0, abs=1e-15)
    r = mp.directional_margin(
        fid, np.array([1.0, 1.0]), 0.99, d, angular_gauge=A, omega=(-2.0, 3.0)
    )
    assert r.reason_plus == "zero_gauge"
    assert (r.M_minus, r.M_plus, r.M) == (2.0, 3.0, 2.0)


@pytest.mark.parametrize("margin_tol", [None, 1e-8])
def test_radius_is_taken_at_the_lower_fidelity(margin_tol):
    """True fidelity FT - delta + |mu - a| dips below FT at a; the evaluator
    reads eps high. A radius at the computed value jumps over the dip to a
    point still read as safe; at F - eps it stops at the first crossing."""
    FT, a, eps = 0.5, 0.3, 1e-3
    delta = eps / 4

    def f_hat(mu):
        return min(1.0, FT - delta + abs(mu - a)) + eps

    first = a - delta
    r = iterative_margin(f_hat, 1.0, FT, eval_tol=eps, margin_tol=margin_tol)
    assert r.M_plus <= first * (1 + 1e-12)

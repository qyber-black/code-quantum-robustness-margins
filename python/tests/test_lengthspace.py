# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests of the shared gauge--budget core (qrobustness.lengthspace)."""

import numpy as np
import pytest

from qrobustness.lengthspace import (
    PathGauge,
    angle_budget,
    interval_grams,
    margin_from,
    refine,
    traceless,
)


def _rand_herm(rng, n):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (A + A.conj().T) / 2


@pytest.fixture
def structures():
    rng = np.random.default_rng(7)
    # p = 2 structures, tau = 5 intervals, dimension 4
    return [[_rand_herm(rng, 4) for _ in range(5)] for _ in range(2)]


def test_traceless_is_traceless_and_projection(structures):
    H = structures[0][0]
    Hb = traceless(H)
    assert abs(np.trace(Hb)) < 1e-12
    # Idempotent projection.
    np.testing.assert_allclose(traceless(Hb), Hb, atol=1e-14)


def test_interval_grams_symmetric_psd(structures):
    for grams in (
        interval_grams(structures),
        interval_grams(structures, make_traceless=True, normalise=True),
    ):
        assert len(grams) == 5
        for G in grams:
            np.testing.assert_allclose(G, G.T, atol=0)
            assert np.min(np.linalg.eigvalsh(G)) > -1e-10


def test_gauge_matches_direct_norm(structures):
    """C(x) equals dt * sum_k ||sum_j x_j H_j^(k)||_F for raw grams."""
    dt = 0.3
    gauge = PathGauge(grams=interval_grams(structures), dt=dt)
    x = np.array([0.7, -1.2])
    direct = dt * sum(
        np.linalg.norm(x[0] * structures[0][k] + x[1] * structures[1][k], "fro")
        for k in range(5)
    )
    assert gauge.C(x) == pytest.approx(direct, rel=1e-12)


def test_positive_homogeneity(structures):
    gauge = PathGauge(grams=interval_grams(structures), dt=0.3)
    x = np.array([0.4, 0.9])
    assert gauge.C(2.5 * x) == pytest.approx(2.5 * gauge.C(x), rel=1e-12)


def test_box_dominates_gauge_on_vertices(structures):
    """C_box(m) >= C(sigma * m) for every sign vertex, with equality
    at the maximising vertex."""
    gauge = PathGauge(
        grams=interval_grams(structures, make_traceless=True, normalise=True), dt=0.3
    )
    m = np.array([0.01, 0.025])
    box = gauge.C_box(m)
    vertex_vals = [
        gauge.C(np.array(s) * m) for s in ((1, 1), (1, -1), (-1, 1), (-1, -1))
    ]
    assert box >= max(vertex_vals) - 1e-15
    # Per-interval maximisation can only exceed any single vertex.
    assert box >= max(vertex_vals)


def test_alpha_cs_bounds_sphere_samples(structures):
    gauge = PathGauge(grams=interval_grams(structures), dt=0.3)
    rng = np.random.default_rng(11)
    alpha = gauge.alpha_cs()
    for _ in range(200):
        d = rng.normal(size=2)
        d /= np.linalg.norm(d)
        assert gauge.C(d) <= alpha * (1 + 1e-12)


def test_margin_from_guard():
    assert margin_from(0.5, 0.0) == float("inf")
    assert margin_from(0.5, 2.0) == 0.25


def test_angle_budget_clamps_and_signs():
    assert angle_budget(1.0, 0.999) == pytest.approx(np.arccos(0.999))
    # F0 slightly above 1 from rounding is clamped.
    assert angle_budget(1.0 + 1e-12, 0.999) == angle_budget(1.0, 0.999)
    assert angle_budget(0.999, 0.999) == 0.0


def test_refine_structure(structures):
    H, Hh = refine(structures[0], structures[1], 4)
    assert len(H) == 20 and len(Hh) == 20
    np.testing.assert_array_equal(H[0], H[3])
    np.testing.assert_array_equal(H[4], structures[0][1])


def test_wrappers_delegate_exactly(structures):
    """JointGauge / AngularGauge / fs_margin_joint reproduce the shared
    core exactly (published-number protection)."""
    from qrobustness.multiparam import angular_gauge, joint_gauge
    from qrobustness.timevarying import fs_margin_joint

    dt = 0.3
    jg = joint_gauge(structures, dt)
    ag = angular_gauge(structures, dt)
    cen = PathGauge(grams=interval_grams(structures, make_traceless=True), dt=dt)
    ang = PathGauge(
        grams=interval_grams(structures, make_traceless=True, normalise=True), dt=dt
    )
    x = np.array([0.7, -1.2])
    assert jg.C(x) == cen.C(x)
    assert ag.C(x) == ang.C(x)
    assert jg.alpha2_certified() == cen.alpha_cs()
    ell, budget = fs_margin_joint(structures, dt, 0.9999, 0.999)
    m = np.array([0.01, 0.025])
    assert ell(m) == ang.C_box(m)
    assert budget == angle_budget(0.9999, 0.999)


def test_angular_gauge_is_joint_gauge_over_sqrt_N(structures):
    """``C_FS_stat(x) = C_joint(x)/sqrt(N)`` exactly, not merely ``<=``.

    Both gauges are built from the traceless structures, the angular one
    additionally normalised by the Hilbert dimension, so the dominance step
    of the paper's Theorem~5 is an identity rather than an inequality.  The
    fixture structures are deliberately not traceless, so this would fail if
    either gauge reverted to raw Frobenius grams.
    """
    from qrobustness.multiparam import angular_gauge, joint_gauge

    dt = 0.3
    N = np.asarray(structures[0][0]).shape[0]
    jg = joint_gauge(structures, dt)
    ag = angular_gauge(structures, dt)
    for x in (np.array([0.7, -1.2]), np.array([1.0, 0.0]), np.array([-0.3, 0.9])):
        assert ag.C(x) == pytest.approx(jg.C(x) / np.sqrt(N), rel=1e-12, abs=0.0)

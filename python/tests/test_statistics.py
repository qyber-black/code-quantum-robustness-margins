# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rank statistics for the case study.

Table I is reported *descriptively* (Pearson r and Spearman rho), so the paper
makes no inferential claim.  The toolbox additionally emits a cross-check --
Kendall tau_b with Holm-corrected two-sided p-values for the three
margin-versus-sensitivity comparisons -- to confirm the descriptive reading
does not depend on the choice of rank statistic.  The MATLAB peer must agree
with the Python reference exactly, hence the shared closed-form asymptotic
p-value rather than each engine's own library routine.
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/lipschitz-margin-python"


def _driver():
    spec = importlib.util.spec_from_file_location(
        "_drv", ROOT / "scripts/run_lipschitz_margin_case_study.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_drv"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_holm_is_step_down_monotone_and_capped():
    holm = _driver().holm
    assert holm([0.01, 0.02, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    # Monotone in the sorted order, and never above 1.
    adj = holm([0.5, 0.6, 0.9])
    assert adj == sorted(adj)
    assert max(adj) <= 1.0
    assert holm([0.4, 0.5, 0.9]) == pytest.approx([1.0, 1.0, 1.0])
    # A single test is unadjusted.
    assert holm([0.031]) == pytest.approx([0.031])


def test_holm_is_never_smaller_than_the_raw_p():
    holm = _driver().holm
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = list(rng.uniform(size=3))
        assert all(a >= b - 1e-15 for a, b in zip(holm(p), p))


def test_kendall_beats_a_sign_flip_that_pearson_would_miss():
    """tau_b is a rank statistic: monotone but nonlinear structure survives."""
    x = np.linspace(0.1, 3.0, 40)
    y = np.exp(-5 * x)  # perfectly monotone decreasing, strongly nonlinear
    tau, p = kendalltau(x, y, variant="b")
    assert tau == pytest.approx(-1.0)
    assert p < 1e-10


@pytest.mark.skipif(
    not (RESULTS / "focal_tests_0.999.csv").is_file(),
    reason="run `make lipschitz-margin-python` first",
)
def test_published_cross_check_matches_a_fresh_recompute():
    """The published cross-check must be reproducible from the margins table."""
    from scipy.stats import spearmanr

    rows = list(csv.DictReader((RESULTS / "margins_table_0.999.csv").open()))
    published = list(csv.DictReader((RESULTS / "focal_tests_0.999.csv").open()))
    assert len(published) == 3
    raw_rho, raw_tau = [], []
    for j, rec in enumerate(published):
        M = np.array([float(r[f"M_H{j}"]) for r in rows])
        Z = np.abs(np.array([float(r[f"zeta_H{j}"]) for r in rows]))
        rho = spearmanr(M, Z)
        tau, tau_p = kendalltau(M, Z, variant="b")
        assert int(rec["n"]) == len(rows) == 61
        assert float(rec["spearman_rho"]) == pytest.approx(rho.statistic, abs=5e-7)
        assert float(rec["p_two_sided"]) == pytest.approx(rho.pvalue, rel=1e-5)
        assert float(rec["kendall_tau_b"]) == pytest.approx(tau, abs=5e-7)
        assert float(rec["kendall_p_two_sided"]) == pytest.approx(tau_p, rel=1e-5)
        raw_rho.append(rho.pvalue)
        raw_tau.append(tau_p)
    for rec, a in zip(published, _driver().holm(raw_rho)):
        assert float(rec["p_holm"]) == pytest.approx(a, rel=1e-5)
    for rec, a in zip(published, _driver().holm(raw_tau)):
        assert float(rec["kendall_p_holm"]) == pytest.approx(a, rel=1e-5)


@pytest.mark.skipif(
    not (RESULTS / "focal_tests_0.999.csv").is_file(),
    reason="run `make lipschitz-margin-python` first",
)
def test_the_two_rank_statistics_agree_on_the_reading():
    """The point of the cross-check: the descriptive reading of Table I must
    not depend on whether rho or tau_b is used."""
    published = list(csv.DictReader((RESULTS / "focal_tests_0.999.csv").open()))
    for rec in published:
        rho, tau = float(rec["spearman_rho"]), float(rec["kendall_tau_b"])
        assert np.sign(rho) == np.sign(tau)          # same direction
        assert abs(tau) <= abs(rho) + 1e-9           # tau_b is the smaller scale
        # Same verdict at the 5% level after Holm correction.
        assert (float(rec["p_holm"]) < 0.05) == (float(rec["kendall_p_holm"]) < 0.05)


@pytest.mark.skipif(
    not (RESULTS / "correlations_0.999.tex").is_file(),
    reason="run `make lipschitz-margin-python` first",
)
def test_table_lower_triangle_is_spearman():
    """Table I's lower triangle is Spearman rho, matching the caption."""
    import re

    rows_tex = []
    for line in (RESULTS / "correlations_0.999.tex").read_text().splitlines():
        nums = re.findall(r"\$([+-]?\d+\.\d+)\$", line)
        if len(nums) >= 7:
            rows_tex.append([float(x) for x in nums[:7]])
    C = np.asarray(rows_tex)
    assert C.shape == (7, 7)

    rows = list(csv.DictReader((RESULTS / "margins_table_0.999.csv").open()))
    for j in range(3):
        M = np.array([float(r[f"M_H{j}"]) for r in rows])
        Z = np.abs(np.array([float(r[f"zeta_H{j}"]) for r in rows]))
        ra = np.argsort(np.argsort(M)).astype(float)
        rb = np.argsort(np.argsort(Z)).astype(float)
        rho = np.corrcoef(ra, rb)[0, 1]
        tau = kendalltau(M, Z, variant="b").statistic
        # Row 4+j is |zeta_j|, column 1+j is M_j: a lower-triangle cell.
        cell = C[4 + j, 1 + j]
        assert cell == pytest.approx(round(rho, 2), abs=1e-9)
        if abs(round(tau, 2) - round(rho, 2)) > 1e-9:
            assert cell != pytest.approx(round(tau, 2), abs=1e-9)

# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Smoke tests for paper-style plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from qrobustness.plotting import (  # noqa: E402
    plot_fidelity_error_sweeps,
    plot_margins_vs_index,
    plot_margins_vs_sensitivity,
)


def test_summary_plots_write(tmp_path: Path):
    """The two summary figures write a non-empty file and keep linear axes:
    the data is already log10-transformed, so a log scale would square it."""
    rng = np.random.default_rng(0)
    n = 12
    err = np.logspace(-7, -4, n)
    M0 = 0.005 + 0.001 * rng.random(n)
    M1 = 0.008 + 0.002 * rng.random(n)
    M2 = 0.009 + 0.003 * rng.random(n)
    z0 = np.logspace(-4, -2, n)
    z1 = np.logspace(-6, -4, n)
    z2 = np.logspace(-6, -4, n)

    p1 = tmp_path / "fid_err.png"
    p2 = tmp_path / "sens.png"
    fig1 = plot_margins_vs_index(err, M0, M1, M2, out_path=p1, dpi=72)
    fig2 = plot_margins_vs_sensitivity(z0, z1, z2, M0, M1, M2, out_path=p2, dpi=72)
    assert p1.is_file() and p1.stat().st_size > 0
    assert p2.is_file() and p2.stat().st_size > 0
    # linear axes (data already log10-transformed)
    assert fig1.axes[0].get_yscale() == "linear"
    assert fig2.axes[0].get_xscale() == "linear"
    assert fig2.axes[1].get_xscale() == "linear"


def test_log10_axis_rejects_an_unknown_axis():
    """An axis name that is neither x nor y is an error in both engines.

    Both engines raise: treating an unrecognised name as one of the two
    would format the wrong axis and say nothing.
    """
    import matplotlib.pyplot as plt

    from qrobustness.plotting import log10_axis

    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="must be"):
            log10_axis(ax, "z", (1e-3, 1.0))
    finally:
        plt.close(fig)


def test_sweep_plot_write(tmp_path: Path):
    """The sweep figure writes, keeps a linear y axis, and holds the label
    sizes and tick count the manuscript layout depends on."""
    xs = [np.linspace(-0.01, 0.01, 50) for _ in range(3)]
    ys = [1e-6 + 0.05 * x**2 for x in xs]
    out = tmp_path / "H0_all.png"
    fig = plot_fidelity_error_sweeps(xs, ys, 0.999, out_path=out, dpi=72)
    assert out.is_file()
    assert fig.axes[0].get_yscale() == "linear"
    assert fig.axes[0].xaxis.label.get_fontsize() == 18
    assert fig.axes[0].yaxis.label.get_fontsize() == 18
    assert len(fig.axes[0].get_xticks()) == 5

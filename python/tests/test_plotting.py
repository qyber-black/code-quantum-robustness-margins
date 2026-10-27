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
    assert [line.get_label() for line in fig2.axes[0].lines] == [
        r"$H_1$ Perturbation", r"$H_2$ Perturbation"
    ]
    assert [line.get_label() for line in fig2.axes[1].lines] == [r"$H_0$ Perturbation"]
    np.testing.assert_array_equal(fig2.axes[0].lines[0].get_ydata(), M1)
    np.testing.assert_array_equal(fig2.axes[0].lines[1].get_ydata(), M2)
    np.testing.assert_array_equal(fig2.axes[1].lines[0].get_ydata(), M0)


def test_sweep_plot_write(tmp_path: Path):
    xs = [np.linspace(-0.01, 0.01, 50) for _ in range(3)]
    ys = [1e-6 + 0.05 * x**2 for x in xs]
    out = tmp_path / "H0_all.png"
    fig = plot_fidelity_error_sweeps(xs, ys, 0.999, out_path=out, dpi=72)
    assert out.is_file()
    assert fig.axes[0].get_yscale() == "linear"
    assert fig.axes[0].xaxis.label.get_fontsize() == 18
    assert fig.axes[0].yaxis.label.get_fontsize() == 18
    assert len(fig.axes[0].get_xticks()) == 5


def test_controller_identity_and_plot_quantities():
    """The MATLAB/Octave peer checks the same unsorted controller fixture."""
    err = [1e-4, 1e-6, 1e-5]
    m0 = [0.008, 0.004, 0.006]
    m1 = [0.011, 0.007, 0.009]
    m2 = [0.012, 0.010, 0.015]
    fig = plot_margins_vs_index(err, m0, m1, m2)
    top, bottom = fig.axes
    for line, expected in zip(
        top.lines, ([0.004, 0.006, 0.008], [0.007, 0.009, 0.011], [0.010, 0.015, 0.012])
    ):
        np.testing.assert_allclose(line.get_ydata(), np.log10(expected), atol=1e-14)
    np.testing.assert_allclose(bottom.lines[0].get_ydata(), [-6, -5, -4], atol=1e-14)
    assert top.get_xlim() == bottom.get_xlim()
    assert np.diff(top.get_ylim())[0] < 1
    assert [line.get_color() for line in top.lines] == [
        (0, 0.447, 0.698),
        (0, 0.620, 0.451),
        (0.835, 0.369, 0),
    ]
    assert top.lines[1].get_markerfacecolor() == "none"
    assert [line.get_marker() for line in top.lines] == ["s", ">", "<"]
    assert all(line.get_markersize() == 3.5 for line in top.lines)

    x = [-0.01, 0, 0.01]
    fig = plot_fidelity_error_sweeps([x] * 8, [[1e-3, 1e-6, 1e-3]] * 8, 0.999)
    curves = fig.axes[0].lines[:-1]
    assert len(set(tuple(line.get_color()) for line in curves)) == 7
    assert curves[0].get_color() == curves[7].get_color()
    threshold = fig.axes[0].lines[-1]
    assert threshold.get_color() == "black" and threshold.get_linestyle() == "--"
    np.testing.assert_allclose(threshold.get_ydata(), np.log10(1 - 0.999), atol=1e-14)

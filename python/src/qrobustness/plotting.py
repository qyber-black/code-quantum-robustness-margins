# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Paper-style plotting helpers mirroring MATLAB +qrobustness plot_* functions.

Decade axes use log10-transformed values on linear axes (not matplotlib log scales).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

# Match MATLAB manuscript marker colors
COLOR_ERR = (0.066, 0.443, 0.745)
COLOR_H0 = (0.0, 0.0, 1.0)
COLOR_H1 = (0.0, 1.0, 0.0)
COLOR_H2 = (1.0, 0.0, 0.0)

# ColorOrder from the manuscript H*_all.fig exports
MATLAB_COLOR_ORDER = [
    (0.0660, 0.4430, 0.7450),
    (0.8660, 0.3290, 0.0000),
    (0.9290, 0.6940, 0.1250),
    (0.5210, 0.0860, 0.8190),
    (0.2310, 0.6660, 0.1960),
    (0.1840, 0.7450, 0.9370),
    (0.8190, 0.0150, 0.5450),
]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedLocator, FuncFormatter
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for qrobustness.plotting; "
            "install with: pip install 'qrobustness[plot]'"
        ) from exc
    return plt, FixedLocator, FuncFormatter


def apply_plot_style(fig) -> None:
    """Force light theme suitable for manuscript figures."""
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
        ax.grid(True, color=(0.15, 0.15, 0.15), alpha=0.15)
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor((0.15, 0.15, 0.15))


def log10_axis(ax, which: str, raw_lim: Sequence[float], *, minor: bool = True) -> None:
    """Configure a linear axis whose data are already log10-transformed."""
    _, FixedLocator, FuncFormatter = _require_matplotlib()
    which = which.lower()
    lo = int(np.floor(np.log10(raw_lim[0])))
    hi = int(np.ceil(np.log10(raw_lim[1])))
    majors = np.arange(lo, hi + 1)

    def _fmt(v, _pos):
        e = int(round(v))
        return rf"$10^{{{e}}}$"

    if which == "y":
        ax.set_yscale("linear")
        ax.set_yticks(majors)
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
        ax.set_ylim(np.log10(raw_lim[0]), np.log10(raw_lim[1]))
        if minor:
            minors = []
            for e in range(lo, hi):
                minors.extend(e + np.log10(np.arange(2, 10)))
            ax.yaxis.set_minor_locator(FixedLocator(minors))
            ax.grid(True, which="minor", linestyle=":", alpha=0.25)
    elif which == "x":
        ax.set_xscale("linear")
        ax.set_xticks(majors)
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
        ax.set_xlim(np.log10(raw_lim[0]), np.log10(raw_lim[1]))
        if minor:
            minors = []
            for e in range(lo, hi):
                minors.extend(e + np.log10(np.arange(2, 10)))
            ax.xaxis.set_minor_locator(FixedLocator(minors))
            ax.grid(True, which="minor", linestyle=":", alpha=0.25)
    else:
        raise ValueError("which must be 'x' or 'y'")


def plot_margins_vs_index(
    err: Sequence[float],
    M0: Sequence[float],
    M1: Sequence[float],
    M2: Sequence[float],
    *,
    out_path: Optional[PathLike] = None,
    dpi: int = 300,
):
    """Manuscript-style margins vs controller index (sorted by epsilon_0)."""
    plt, _, _ = _require_matplotlib()
    err = np.asarray(err, dtype=float).ravel()
    ord_ = np.argsort(err)
    # Clamp for log10: near-perfect fidelity can yield eps<=0 from roundoff.
    floor_pos = np.finfo(float).tiny
    err_s = np.maximum(err[ord_], floor_pos)
    m0 = np.maximum(np.asarray(M0, dtype=float).ravel()[ord_], floor_pos)
    m1 = np.maximum(np.asarray(M1, dtype=float).ravel()[ord_], floor_pos)
    m2 = np.maximum(np.asarray(M2, dtype=float).ravel()[ord_], floor_pos)
    idx = np.arange(1, err_s.size + 1)

    fig, ax = plt.subplots(figsize=(528 / 96, 482 / 96), dpi=96)
    ax.plot(
        idx,
        np.log10(err_s),
        "-",
        color=COLOR_ERR,
        linewidth=1.2,
        label="nominal fidelity error",
    )
    ax.plot(
        idx,
        np.log10(m0),
        "s",
        color=COLOR_H0,
        markerfacecolor=COLOR_H0,
        markersize=6,
        linestyle="none",
        label=r"$H_0$ robustness margins",
    )
    ax.plot(
        idx,
        np.log10(m1),
        ">",
        color=COLOR_H1,
        markerfacecolor=COLOR_H1,
        markersize=6,
        linestyle="none",
        label=r"$H_1$ robustness margins",
    )
    ax.plot(
        idx,
        np.log10(m2),
        "<",
        color=COLOR_H2,
        markerfacecolor=COLOR_H2,
        markersize=6,
        linestyle="none",
        label=r"$H_2$ robustness margins",
    )
    log10_axis(ax, "y", [1e-7, 1e-1])
    ax.set_xlim(1, err_s.size)
    ax.set_xlabel("controller index")
    ax.set_ylabel("")
    ax.legend(loc="lower right")
    ax.tick_params(labelsize=14)
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    ):
        item.set_fontsize(14)
    apply_plot_style(fig)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    return fig


def plot_margins_vs_sensitivity(
    abs_z0: Sequence[float],
    abs_z1: Sequence[float],
    abs_z2: Sequence[float],
    M0: Sequence[float],
    M1: Sequence[float],
    M2: Sequence[float],
    *,
    out_path: Optional[PathLike] = None,
    dpi: int = 300,
):
    """Two-panel M vs |zeta|; x is log10(|zeta|) on a linear axis."""
    plt, _, _ = _require_matplotlib()
    z0 = np.maximum(np.asarray(abs_z0, dtype=float).ravel(), np.finfo(float).tiny)
    z1 = np.maximum(np.asarray(abs_z1, dtype=float).ravel(), np.finfo(float).tiny)
    z2 = np.maximum(np.asarray(abs_z2, dtype=float).ravel(), np.finfo(float).tiny)
    m0 = np.asarray(M0, dtype=float).ravel()
    m1 = np.asarray(M1, dtype=float).ravel()
    m2 = np.asarray(M2, dtype=float).ravel()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(520 / 96, 520 / 96), dpi=96)

    ax1.plot(
        np.log10(z0),
        m0,
        "s",
        color=COLOR_H0,
        markerfacecolor=COLOR_H0,
        markersize=6,
        linestyle="none",
        label=r"$H_0$ Perturbation",
    )
    ax1.set_xlabel(r"$|\zeta|$")
    ax1.set_ylabel("Robustness margin")
    ax1.legend(loc="upper left")
    log10_axis(ax1, "x", [float(z0.min()) * 0.8, float(z0.max()) * 1.2])
    ax1.tick_params(labelsize=12)
    # Match MATLAB-style scientific offset on the H0 panel
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))

    ax2.plot(
        np.log10(z1),
        m1,
        ">",
        color=COLOR_H1,
        markerfacecolor=COLOR_H1,
        markersize=6,
        linestyle="none",
        label=r"$H_1$ Perturbation",
    )
    ax2.plot(
        np.log10(z2),
        m2,
        "<",
        color=COLOR_H2,
        markerfacecolor=COLOR_H2,
        markersize=6,
        linestyle="none",
        label=r"$H_2$ Perturbation",
    )
    ax2.set_xlabel(r"$|\zeta|$")
    ax2.set_ylabel("Robustness margin")
    ax2.legend(loc="upper left")
    z12 = np.concatenate([z1, z2])
    log10_axis(ax2, "x", [float(z12.min()) * 0.8, float(z12.max()) * 1.2])
    ax2.tick_params(labelsize=12)

    apply_plot_style(fig)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    return fig


def plot_fidelity_error_sweeps(
    X_list: Sequence[Sequence[float]],
    Y_list: Sequence[Sequence[float]],
    FT: float,
    *,
    xlabel: str = r"Perturbation strength $\mu$",
    xlim: Optional[Tuple[float, float]] = None,
    font_size: int = 18,
    num_xticks: int = 5,
    out_path: Optional[PathLike] = None,
    dpi: int = 300,
):
    """Spaghetti plot of fidelity error vs delta (log10(error) on linear y)."""
    plt, _, _ = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=96)
    xmax = 0.0
    for i, (x, y) in enumerate(zip(X_list, Y_list)):
        x = np.asarray(x, dtype=float).ravel()
        y = np.maximum(np.asarray(y, dtype=float).ravel(), np.finfo(float).tiny)
        color = MATLAB_COLOR_ORDER[i % len(MATLAB_COLOR_ORDER)]
        ax.plot(x, np.log10(y), "-", color=color, linewidth=1.0)
        xmax = max(xmax, float(np.max(np.abs(x))))

    thr = 1.0 - FT
    ax.plot([-xmax, xmax], [np.log10(thr), np.log10(thr)], "-.", color="r", linewidth=2)
    if xlim is None:
        ax.set_xlim(-xmax, xmax)
    else:
        ax.set_xlim(xlim)
    xlo, xhi = ax.get_xlim()
    ax.set_xticks(np.linspace(xlo, xhi, num_xticks))
    log10_axis(ax, "y", [1e-7, 1.2e-3])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fidelity error")
    ax.tick_params(labelsize=font_size)
    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(font_size)
    apply_plot_style(fig)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    return fig


def save_fig(fig, path: PathLike, *, dpi: int = 300) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")

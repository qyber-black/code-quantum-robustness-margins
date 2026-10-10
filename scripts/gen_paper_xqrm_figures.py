#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the figures of the xQRM paper from the driver results.

Writes results/paper-xqrm/figures/*.pdf (copied into the paper repository
by "make sync-xQRM"):

  fig_margins       per controller: r_0, r_FS, M, M^K and M^K_tv (structure
                    H_1), with brackets [r_FS, m_adv] for the probed controllers
                    (multiparam_0.999_angular, kosut_comparison_0.999_angular[_tv],
                    fs_validity, tv_bracket)
  fig_directions    free angular and cross-polytope radii against M(d) and its
                    upper bracket along each probed direction, one controller
                    (multiparam_0.999_angular)
  fig_single_qubit  analytic fidelity curves of the pi-pulse with every
                    certificate (single_qubit_0.999)
  fig_ratios        M/M^K and r_FS/M^K_tv over the 3-qubit, CNOT and 4-qubit
                    ensembles (multiparam angular, kosut_comparison, fs_validity,
                    cnot_margins, scaling4q_margins)
  fig_open          open-system one-step and iterated margins against the
                    reference crossings (open_margins, open_amp)
  fig_slice         resolved safe region and nested certified regions in the
                    (mu_1, mu_2) plane (slice_ctrl<c>_0.999.npz, multiparam angular)
  fig_validity      adversarial minimum fidelity against budget for the
                    violating controller (budget_sweep_ctrl16_H1.csv)
  fig_mixed         mixed coherent-dissipative region in the (mu_1, gamma_z)
                    plane (mixed_ctrl<c>.npz)

Options: --allow-missing skips figures whose inputs are absent; --controller
selects the controller for fig_directions, fig_slice and fig_mixed.

Colours are an Okabe-Ito CVD-safe subset; series are also separated by
marker shape.

Usage: python3 scripts/gen_paper_xqrm_figures.py [--allow-missing] [--controller N]"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# No CreationDate, so the PDFs are byte-reproducible.
PDF_METADATA = {"CreationDate": None}
# ruff: noqa: E402 -- matplotlib.use('Agg') above must execute before
# pyplot is imported, so these cannot shift to the top of the file.
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from _paper import col, configure, have, read

from _drivers import DEFAULT_FT, WITNESS_CONTROLLER, WITNESS_STRUCTURE

ROOT = Path(__file__).resolve().parents[1]

BLUE, ORANGE, GREEN, PINK, VERMIL = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
)
GRAY = "#6e6e6e"
FT = DEFAULT_FT

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)


def fig_margins(res: Path, out: Path) -> None:
    mp = read(res / "multiparameter-margin-python/multiparam_0.999_angular.csv")
    kos = read(res / "time-bandwidth-bound-python/kosut_comparison_0.999_angular.csv")
    kos_tv = read(
        res / "time-bandwidth-bound-python/kosut_comparison_0.999_angular_tv.csv"
    )
    fsv = read(res / "time-bandwidth-bound-python/fs_validity_0.999.csv")
    tvb = read(res / "multiparameter-margin-python/tv_bracket_0.999.csv")

    err = col(mp, "err")
    order = np.argsort(err)
    r0 = col(mp, "r0_H1")[order]
    M = np.minimum(col(mp, "M_+e1"), col(mp, "M_-e1"))[order]
    MK = col(kos, "KM_H1")[order]
    MKtv = col(kos_tv, "KM_H1")[order]
    rfs = np.full(len(mp), np.nan)
    for r in fsv:
        if r["structure"] == "H1":
            rfs[int(r["controller"]) - 1] = float(r["r_fs"])
    rfs = rfs[order]
    ctrl_ids = col(mp, "controller").astype(int)[order]
    pos_of = {c: x for x, c in enumerate(ctrl_ids)}

    x = np.arange(len(r0))
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    # Adversarial brackets [r_FS, m_adv] on M_tv for the controllers that were probed.
    first = True
    for r in tvb:
        px = pos_of[int(r["controller"])]
        lo, hi = float(r["r_fs"]), float(r["m_adv"])
        ax.plot(
            [px, px],
            [lo, hi],
            color=GRAY,
            lw=2.4,
            alpha=0.45,
            solid_capstyle="butt",
            zorder=1,
            label=r"bracket on $M_{\mathrm{tv}}$" if first else None,
        )
        first = False
    ax.plot(x, M, "o", ms=2.4, color=BLUE, zorder=3, label=r"$M$ (constant, iterated)")
    ax.plot(x, MK, "s", ms=2.4, color=GREEN, zorder=3, label=r"$M^K$ (constant)")
    ax.plot(x, r0, "^", ms=2.4, color=ORANGE, zorder=3, label=r"$r_0$ (time-varying)")
    ax.plot(
        x,
        MKtv,
        "D",
        ms=2.2,
        color=PINK,
        zorder=3,
        label=r"$M^{K,\mathrm{tri}}_{\mathrm{tv}}$ (trajectory)",
    )
    ax.plot(
        x,
        rfs,
        "v",
        ms=2.4,
        color=VERMIL,
        zorder=3,
        label=r"$r_{\mathrm{FS}}$ (geometric)",
    )
    ax.set_yscale("log")
    lo = min(r0.min(), MKtv.min())
    ax.set_ylim(bottom=lo / 7.0)  # room for the legend below the data
    ax.set_xlabel(r"controller (ordered by nominal error $\varepsilon_0$)")
    ax.set_ylabel(r"margin (structure $H_1$)")
    ax.legend(
        loc="lower center",
        ncol=2,
        frameon=False,
        borderpad=0.2,
        handletextpad=0.4,
        labelspacing=0.25,
        columnspacing=0.8,
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_margins.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def fig_directions(res: Path, out: Path, controller: int = 1) -> None:
    mp = read(res / "multiparameter-margin-python/multiparam_0.999_angular.csv")
    row = next(r for r in mp if int(float(r["controller"])) == controller)
    surplus = float(row["fid"]) - FT
    L = np.array([float(row[f"L_H{j}"]) for j in range(3)])

    dir_names = [c[2:] for c in row if c.startswith("M_")]
    rang = np.array([float(row[f"rang_{n}"]) for n in dir_names])

    def unit(name):
        if name[1] == "e":  # +e0 etc.
            d = np.zeros(3)
            d[int(name[2])] = 1.0 if name[0] == "+" else -1.0
            return d
        signs = np.array([1.0 if ch == "p" else -1.0 for ch in name[4:]])
        return signs / np.sqrt(3.0)

    dirs = {n: unit(n) for n in dir_names}
    poly = np.array([surplus / (np.abs(dirs[n]) @ L) for n in dir_names])
    Md = np.array([float(row[f"M_{n}"]) for n in dir_names])
    M_upper = np.array([float(row[f"Mupper_{n}"]) for n in dir_names])

    order = np.argsort(Md)
    labels = [_direction_label(dir_names[i]) for i in order]

    x = np.arange(len(dir_names))
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    ax.vlines(
        x,
        Md[order],
        M_upper[order],
        color=GRAY,
        lw=2.0,
        alpha=0.45,
        label=r"bracket $[M(d), M_{\mathrm{upper}}(d)]$",
    )
    ax.plot(x, Md[order], "o", ms=3.0, color=BLUE, label=r"iterated $M(d)$")
    ax.plot(x, rang[order], "s", ms=2.6, color=GREEN, label="free angular radius")
    ax.plot(x, poly[order], "^", ms=3.0, color=ORANGE, label="cross-polytope radius")
    ax.set_yscale("log")
    ax.set_ylim(top=M_upper.max() * 3.5)  # room for the legend above the data
    ax.set_xticks(x)
    ax.set_xticklabels([f"${lab}$" for lab in labels], rotation=60, fontsize=6)
    ax.set_xlabel("direction $d$ (sorted by $M(d)$)")
    ax.set_ylabel(f"certified radius (controller {controller})")
    ax.legend(
        loc="upper left",
        frameon=False,
        borderpad=0.2,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_directions.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="skip artefacts whose inputs are absent (development only)",
    )
    ap.add_argument("--controller", type=int, default=1)
    args = ap.parse_args()
    res = ROOT / "results"
    configure(res, args.allow_missing)
    out = ROOT / "results" / "paper-xqrm" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_margins(res, out)
    fig_directions(res, out, args.controller)
    fig_single_qubit(res, out)
    fig_ratios(res, out)
    fig_open(res, out)
    fig_slice(res, out, args.controller)
    fig_validity(res, out)
    fig_mixed(res, out, args.controller)
    print("wrote the xQRM figures")


def fig_single_qubit(res: Path, out: Path) -> None:
    """Analytic fidelity curves of the pi-pulse together with every certificate."""
    sq = read(res / "single-qubit-python/single_qubit_0.999.csv")
    OMEGA = np.pi

    def F_amp(d):
        return np.abs(np.cos(0.5 * np.pi * d))

    def F_det(d):
        og = np.sqrt(OMEGA**2 + d**2)
        return np.abs(np.sin(0.5 * og)) * OMEGA / og

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.3), sharey=True)
    for ax, row, F, name in zip(
        axes, sq, (F_amp, F_det), ("amplitude", "detuning"), strict=True
    ):
        dmax = 1.35 * float(row["delta_star"])
        d = np.linspace(0, dmax, 400)
        ax.plot(
            d,
            F(d),
            color="#333333",
            lw=1.2,
            label=r"$F(\delta)$" if name == "amplitude" else None,
        )
        ax.axhline(0.999, color=GRAY, lw=0.8, ls=":")  # F_T; no legend entry
        marks = [
            ("r0", ORANGE, r"$r_0$"),
            ("KM_tv", PINK, r"$M^{K,\mathrm{tri}}_{\mathrm{tv}}$"),
            ("r_fs", VERMIL, r"$r_{\mathrm{FS}}$"),
            ("KM", GREEN, r"$M^K$"),
            ("M", BLUE, r"$M$"),
        ]
        for key, color, lab in marks:
            ax.axvline(
                float(row[key]),
                color=color,
                lw=1.1,
                label=lab if name == "amplitude" else None,
            )
        ax.plot(
            [float(row["delta_star"])],
            [0.999],
            "o",
            ms=4,
            color="#333333",
            zorder=5,
            label=r"$\delta^\ast$" if name == "amplitude" else None,
        )
        ax.set_xlabel(rf"$\delta$ ({name})")
        ax.set_xlim(0, dmax)
    axes[0].set_ylim(0.9970, 1.0002)
    axes[0].set_ylabel(r"fidelity $F$")
    axes[0].legend(
        loc="lower left",
        frameon=False,
        borderpad=0.2,
        handletextpad=0.4,
        labelspacing=0.25,
        fontsize=6.5,
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_single_qubit.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def _direction_label(name: str) -> str:
    """Math label for a probed direction.

    Sign patterns encoded as ``p``/``m`` in the CSV (``diagmpp``) become
    ``d(-,+,+)``; coordinate directions become ``+e_j`` / ``-e_j``.
    """
    if name.startswith("diag"):
        return "d({})".format(",".join("+" if c == "p" else "-" for c in name[4:]))
    return name.replace("+e", "+e_").replace("-e", "-e_")


def fig_ratios(res: Path, out: Path) -> None:
    """Certificate-to-bound ratios over the three ensembles."""
    data = []  # (ensemble label, M/MK values, rfs/KMtv values)
    mp_rows = read(res / "multiparameter-margin-python/multiparam_0.999_angular.csv")
    kos = read(res / "time-bandwidth-bound-python/kosut_comparison_0.999_angular.csv")
    kos_tv = read(
        res / "time-bandwidth-bound-python/kosut_comparison_0.999_angular_tv.csv"
    )
    fsv = read(res / "time-bandwidth-bound-python/fs_validity_0.999.csv")
    rfs3 = {}
    for r in fsv:
        rfs3.setdefault(r["structure"], {})[int(r["controller"])] = float(r["r_fs"])
    mk, rk, mkd, rkd = [], [], [], []
    for j, tag in enumerate(("H0", "H1", "H2")):
        M = np.minimum(col(mp_rows, f"M_+e{j}"), col(mp_rows, f"M_-e{j}"))
        KM = col(kos, f"KM_{tag}")
        ratio = M / KM
        mk.extend(ratio)
        mkd.extend([tag == "H0"] * len(ratio))
        for r in kos_tv:
            c = int(float(r["controller"]))
            if c in rfs3.get(tag, {}):
                rk.append(rfs3[tag][c] / float(r[f"KM_{tag}"]))
                rkd.append(tag == "H0")
    data.append(("3-qubit", np.array(mk), np.array(rk), np.array(mkd), np.array(rkd)))
    for csvf, label, tags in (
        ("cnot-python/cnot_margins_0.999.csv", "CNOT", ("H0", "X1", "X2")),
        (
            "scaling-python/scaling4q_margins_0.999.csv",
            "4-qubit",
            ("H0", "X1", "X2", "X3", "X4"),
        ),
    ):
        p = res / csvf
        if not have(p, out / "fig_ratios.pdf"):
            continue
        rows = read(p)
        mk = np.concatenate([col(rows, f"M_{t}") / col(rows, f"KM_{t}") for t in tags])
        rk = np.concatenate(
            [col(rows, f"rfs_{t}") / col(rows, f"KMtv_{t}") for t in tags]
        )
        # The drift structure is the first tag; it forms a cluster of its own.
        drift = np.concatenate([np.full(len(rows), t == tags[0]) for t in tags])
        data.append((label, mk, rk, drift, drift))

    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    rng = np.random.default_rng(0)
    medians = []
    for i, (_label, mk, rk, mkd, rkd) in enumerate(data):
        for k, (v, drift, color, lab) in enumerate(
            (
                (mk, mkd, BLUE, r"$M/M^K$ (constant)"),
                (
                    rk,
                    rkd,
                    VERMIL,
                    r"$r_{\mathrm{FS}}/M^{K,\mathrm{tri}}_{\mathrm{tv}}$ (trajectory)",
                ),
            )
        ):
            center = i + (k - 0.5) * 0.36
            xs = center + rng.uniform(-0.09, 0.09, v.size)
            # The drift structure is drawn as an open square.
            ax.plot(
                xs[~drift],
                v[~drift],
                "o",
                ms=1.8,
                alpha=0.5,
                color=color,
                label=lab if i == 0 else None,
                zorder=3,
            )
            ax.plot(
                xs[drift],
                v[drift],
                "s",
                ms=2.4,
                mfc="none",
                mew=0.6,
                alpha=0.85,
                color=color,
                zorder=3,
            )
            medians.append((center, float(np.median(v)), color))
        if i == 0:
            # A marker convention spanning both colours, so the key stays neutral.
            ax.plot(
                [],
                [],
                "s",
                ms=2.4,
                mfc="none",
                mew=0.6,
                color=GRAY,
                label="drift structure",
            )
    for center, med, color in medians:
        ax.plot(
            [center - 0.16, center + 0.16],
            [med, med],
            color=color,
            lw=2.4,
            zorder=6,
            solid_capstyle="butt",
        )
    ax.axhline(1.0, color=GRAY, lw=0.8, ls=":")
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data])
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.9)  # legend headroom
    ax.set_ylabel("structured / universal margin")
    ax.legend(
        loc="upper right",
        frameon=False,
        borderpad=0.2,
        handletextpad=0.4,
        labelspacing=0.25,
        fontsize=6.5,
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_ratios.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def fig_open(res: Path, out: Path) -> None:
    """Open-system one-step and iterated margins against reference crossings."""
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    deph = read(res / "lindblad-margin-python/open_margins_0.999.csv")
    # Iterated margins: small filled markers; one-step radii use larger
    # open markers at higher zorder, so that a one-step point coinciding
    # with an iterated one reads as a ring (halo) around that dot.
    ax.plot(
        col(deph, "M_gamma"),
        col(deph, "gamma_star"),
        "o",
        ms=2.4,
        color=BLUE,
        label=r"iterated, dephasing",
        zorder=3,
    )
    ax.plot(
        col(deph, "r0_gamma"),
        col(deph, "gamma_star"),
        "o",
        ms=4.4,
        mfc="none",
        mew=0.8,
        color=BLUE,
        zorder=4,
        label=r"one-step, dephasing",
    )
    amp_csv = res / "lindblad-margin-python/open_amp_0.999.csv"
    if have(amp_csv, out / "fig_open.pdf"):
        amp = read(amp_csv)
        ax.plot(
            col(amp, "M_amp"),
            col(amp, "amp_star"),
            "s",
            ms=2.4,
            color=VERMIL,
            label="iterated, amp. damping",
            zorder=3,
        )
        ax.plot(
            col(amp, "r0_amp"),
            col(amp, "amp_star"),
            "s",
            ms=4.4,
            mfc="none",
            mew=0.8,
            color=VERMIL,
            zorder=4,
            label=r"one-step, amp. damping",
        )
        ax.plot(
            col(amp, "M_diag"),
            col(amp, "g_star_diag"),
            "^",
            ms=2.4,
            color=GREEN,
            label="iterated, joint diagonal",
            zorder=3,
        )
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=GRAY,
        lw=0.8,
        ls="--",
        label="exact ($y = x$)",
        zorder=1,
    )
    # Guide line y = 2x: the first-order factor 2 for amplitude damping.
    ax.plot(
        [lo, hi],
        [2.0 * lo, 2.0 * hi],
        color=GRAY,
        lw=0.8,
        ls=(0, (1, 1.5)),
        label=r"factor $2$ ($y = 2x$)",
        zorder=1,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Label every tick that carries data; the default log formatter leaves
    # the amplitude-damping cluster near 8e-5 unlabelled.
    ticks = [3e-5, 4e-5, 6e-5, 8e-5]
    ax.set_xticks(ticks)
    ax.set_xticklabels([rf"${t * 1e5:g}\times10^{{-5}}$" for t in ticks])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(r"certified margin")
    ax.set_ylabel(r"reference threshold crossing")
    ax.legend(
        loc="lower right",
        frameon=False,
        borderpad=0.2,
        handletextpad=0.4,
        labelspacing=0.25,
        fontsize=6.5,
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_open.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def fig_slice(res: Path, out: Path, controller: int = 1) -> None:
    """Numerically resolved safe region against the three nested certified
    free regions in the (mu_1, mu_2) plane."""
    npz_path = res / f"multiparameter-margin-python/slice_ctrl{controller}_0.999.npz"
    if not have(npz_path, out / "fig_slice.pdf"):
        return
    d = np.load(npz_path)
    mu1, mu2, F = d["mu1"], d["mu2"], d["F"]
    L, fid, FT_ = float(d["L"][1]), float(d["fid"]), float(d["FT"])
    L2 = float(d["L"][2])
    surplus = fid - FT_

    mp_rows = read(res / "multiparameter-margin-python/multiparam_0.999_angular.csv")
    row = next(r for r in mp_rows if int(float(r["controller"])) == controller)

    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    # Numerically resolved boundary of the safe region (F = FT contour).
    ax.contourf(
        mu1, mu2, (F.T >= FT_).astype(float), levels=[0.5, 1.5], colors=["#e8f0e8"]
    )
    ax.contour(mu1, mu2, F.T, levels=[FT_], colors="#333333", linewidths=1.2)
    # Nested free regions: the static angular gauge (outermost), joint
    # Frobenius gauge, cross-polytope (innermost).
    if "theta" in d:
        th, rj, ra = d["theta"], d["r_joint"], d["r_angular"]
        ax.fill(ra * np.cos(th), ra * np.sin(th), color=GREEN, alpha=0.20, lw=0)
        ax.plot(
            ra * np.cos(th),
            ra * np.sin(th),
            color=GREEN,
            lw=1.0,
            label="free angular gauge",
        )
        ax.plot(
            rj * np.cos(th),
            rj * np.sin(th),
            color=BLUE,
            lw=1.0,
            ls="--",
            label="free joint gauge",
        )
    # Certified free cross-polytope slice given by L1|m1| + L2|m2| <= surplus.
    a, b = surplus / L, surplus / L2
    ax.fill([a, 0, -a, 0], [0, b, 0, -b], color=ORANGE, alpha=0.35, lw=0)
    ax.plot(
        [a, 0, -a, 0, a], [0, b, 0, -b, 0], color=ORANGE, lw=1.0, label="free polytope"
    )
    # Directional margins in-plane (axes +-e1, +-e2).
    pts = [
        (float(row["M_+e1"]), 0),
        (-float(row["M_-e1"]), 0),
        (0, float(row["M_+e2"])),
        (0, -float(row["M_-e2"])),
    ]
    ax.plot(
        [p[0] for p in pts],
        [p[1] for p in pts],
        "o",
        ms=4,
        color=BLUE,
        label=r"directional $M(d)$",
        zorder=5,
    )
    ax.axhline(0, color=GRAY, lw=0.4)
    ax.axvline(0, color=GRAY, lw=0.4)
    ax.set_xlabel(r"$\mu_1$ (control structure $H_1$)")
    ax.set_ylabel(r"$\mu_2$ (control structure $H_2$)")
    ax.set_aspect("equal")
    ax.legend(
        loc="upper right",
        frameon=False,
        borderpad=0.2,
        handletextpad=0.4,
        labelspacing=0.25,
        fontsize=6.5,
    )
    # Inset: zoom on the nested free regions (they sit an order of
    # magnitude inside the resolved boundary).
    if "theta" in d:
        r_zoom = 1.25 * float(np.max(d["r_angular"]))
        axi = ax.inset_axes([0.015, 0.015, 0.36, 0.36])
        axi.fill(ra * np.cos(th), ra * np.sin(th), color=GREEN, alpha=0.20, lw=0)
        axi.plot(ra * np.cos(th), ra * np.sin(th), color=GREEN, lw=1.0)
        axi.plot(rj * np.cos(th), rj * np.sin(th), color=BLUE, lw=1.0, ls="--")
        axi.fill([a, 0, -a, 0], [0, b, 0, -b], color=ORANGE, alpha=0.35, lw=0)
        axi.plot([a, 0, -a, 0, a], [0, b, 0, -b, 0], color=ORANGE, lw=1.0)
        axi.set_xlim(-r_zoom, r_zoom)
        axi.set_ylim(-r_zoom, r_zoom)
        axi.set_aspect("equal")
        axi.set_xticks([])
        axi.set_yticks([])
        for s in axi.spines.values():
            s.set_linewidth(0.6)
        ax.indicate_inset_zoom(axi, edgecolor=GRAY, lw=0.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_slice.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def fig_validity(res: Path, out: Path) -> None:
    """Adversarial min-fidelity against budget for the violating controller."""
    path = (
        res
        / "time-bandwidth-bound-python"
        / f"budget_sweep_ctrl{WITNESS_CONTROLLER}_{WITNESS_STRUCTURE}.csv"
    )
    if not have(path, out / "fig_validity.pdf"):
        return
    lines = path.read_text().splitlines()
    certs = {}
    for tok in lines[-1].lstrip("# ").split():
        if "=" in tok:
            k, v = tok.split("=")
            certs[k] = float(v)
    rows = [
        dict(zip(lines[0].split(","), line.split(","), strict=True))
        for line in lines[1:]
        if not line.startswith("#")
    ]
    m = np.array([float(r["m"]) for r in rows])

    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    for q, color, mk in ((1, BLUE, "o"), (4, GREEN, "s"), (16, PINK, "D")):
        F = np.array([float(r[f"Fmin_x{q}"]) for r in rows])
        ax.plot(
            m,
            F,
            mk + "-",
            ms=2.6,
            lw=0.9,
            color=color,
            label=rf"adversary, $\times{q}$ grid",
        )
    ax.axhline(0.999, color="#333333", lw=0.8, ls=":")
    ax.annotate(r"$F_T$", (m[0], 0.99905), fontsize=7)
    for key, color, lab, y, side in (
        ("r0", ORANGE, r"$r_0$", 0.99908, "right"),
        ("KMtv", "#777777", r"$M^{K,\mathrm{tri}}_{\mathrm{tv}}$", 0.99868, "left"),
        ("rfs", VERMIL, r"$r_{\mathrm{FS}}$", 0.99902, "right"),
        # Above the curves: at the bottom the refined adversaries cross it.
        ("KM", "#333333", r"$M^K$", 0.99998, "left"),
    ):
        ax.axvline(certs[key], color=color, lw=1.0, ls="--")
        if side == "left":
            ax.annotate(
                lab,
                (certs[key] * 0.96, y),
                fontsize=6.5,
                color=color,
                ha="right",
                va="top",
            )
        else:
            ax.annotate(lab, (certs[key] * 1.04, y), fontsize=7, color=color)
    ax.set_ylim(0.99865, 1.00005)
    ax.set_xscale("log")
    ax.set_xticks([5e-4, 1e-3, 2e-3, 5e-3])
    ax.set_xticklabels(
        [
            r"$5\times10^{-4}$",
            r"$10^{-3}$",
            r"$2\times10^{-3}$",
            r"$5\times10^{-3}$",
        ]
    )
    ax.tick_params(axis="x", which="minor", labelbottom=False)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(r"sup-norm budget $m$")
    ax.set_ylabel(r"minimum fidelity found")
    ax.legend(
        loc="lower left",
        frameon=False,
        borderpad=0.2,
        handletextpad=0.4,
        labelspacing=0.25,
        fontsize=6.5,
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_validity.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def fig_mixed(res: Path, out: Path, controller: int = 1) -> None:
    """Mixed coherent-dissipative region in the (mu_1, gamma_z) plane."""
    path = res / f"lindblad-margin-python/mixed_ctrl{controller}.npz"
    if not have(path, out / "fig_mixed.pdf"):
        return
    d = np.load(path)
    mu, g, F = d["mu"], d["gamma"], d["F"]
    ft_pro = float(d["FT"]) ** 2
    L_mu, L_g, s = float(d["L_mu"]), float(d["L_gamma"]), float(d["surplus"])

    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    ax.contourf(
        mu, g, (F.T >= ft_pro).astype(float), levels=[0.5, 1.5], colors=["#e8f0e8"]
    )
    ax.contour(mu, g, F.T, levels=[ft_pro], colors="#333333", linewidths=1.2)
    # Free mixed simplex L_mu |mu| + L_g gamma <= surplus, with gamma >= 0.
    a, b = s / L_mu, s / L_g
    ax.fill([a, 0, -a], [0, b, 0], color=ORANGE, alpha=0.35, lw=0)
    ax.plot(
        [a, 0, -a, a], [0, b, 0, 0], color=ORANGE, lw=1.0, label="free mixed simplex"
    )
    # Iterated directional margins.
    dmu = np.asarray(d["ray_dmu"], dtype=float)
    dg = np.asarray(d["ray_dgamma"], dtype=float)
    M = np.asarray(d["ray_M"], dtype=float)
    ax.plot(M * dmu, M * dg, "o", ms=4, color=BLUE, zorder=5, label="iterated margins")
    ax.axvline(0, color=GRAY, lw=0.4)
    ax.set_xlabel(r"$\mu_1$ (control structure $H_1$)")
    ax.set_ylabel(r"dephasing rate $\gamma_z$")
    ax.set_ylim(bottom=0)
    ax.legend(
        loc="upper right",
        frameon=False,
        borderpad=0.2,
        handletextpad=0.4,
        labelspacing=0.25,
        fontsize=6.5,
    )
    # Inset: the free simplex is invisible at full scale (thin along the
    # coherent direction) -- zoom near the origin to display it.
    axin = ax.inset_axes([0.06, 0.45, 0.34, 0.5])
    axin.contourf(
        mu, g, (F.T >= ft_pro).astype(float), levels=[0.5, 1.5], colors=["#e8f0e8"]
    )
    axin.fill([a, 0, -a], [0, b, 0], color=ORANGE, alpha=0.5, lw=0)
    axin.plot([a, 0, -a, a], [0, b, 0, 0], color=ORANGE, lw=1.0)
    axin.plot(M * dmu, M * dg, "o", ms=3, color=BLUE, zorder=5)
    axin.set_xlim(-2.5 * a, 2.5 * a)
    axin.set_ylim(0, 1.3 * b)
    axin.tick_params(labelsize=5)
    axin.set_title("origin zoom", fontsize=6)
    ax.indicate_inset_zoom(axin, edgecolor=GRAY, lw=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_mixed.pdf", metadata=PDF_METADATA)
    plt.close(fig)


if __name__ == "__main__":
    main()

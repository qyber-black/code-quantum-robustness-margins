#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
r"""Full paper case study in Python -> results/lipschitz-margin-python/.

Computes margins and \zeta, writes margins CSV + correlations tex, and paper-style plots
(including H*_all sweeps when --sweep is set; Makefile always passes --sweep).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from qrobustness import (
    dH_structure,
    differential_sensitivity,
    fidelity_vs_delta,
    iterative_margin,
    lipschitz_constant,
    load_controllers,
    load_problem,
    make_fidelity_fn,
    perturbed_hamiltonians,
    structure_constant,
)
from qrobustness.plotting import (
    plot_fidelity_error_sweeps,
    plot_margins_vs_index,
    plot_margins_vs_sensitivity,
)

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
OUT_DIR = ROOT / "results/lipschitz-margin-python"
BUILD = ROOT / "build"

FT = 0.999
ETA = 1e-6
STRUCTURES = ("H0", "H1", "H2")
XLIMS = {
    "H0": (-8e-3, 8e-3),
    "H1": (-2e-2, 2e-2),
    "H2": (-2e-2, 2e-2),
}


def write_correlation_tex(rows: list[dict], path: Path) -> None:
    vars_ = ["err", "M_H0", "M_H1", "M_H2", "zeta_H0", "zeta_H1", "zeta_H2"]
    labels = [
        r"$\varepsilon_0$",
        "$M_0$",
        "$M_1$",
        "$M_2$",
        r"$\zeta_0$",
        r"$\zeta_1$",
        r"$\zeta_2$",
    ]
    X = np.column_stack([np.array([r[v] for r in rows], dtype=float) for v in vars_])
    # Pearson / Spearman
    P = np.corrcoef(X, rowvar=False)

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return np.corrcoef(ra, rb)[0, 1]

    S = np.eye(7)
    for i in range(7):
        for j in range(i):
            S[i, j] = S[j, i] = spearman(X[:, i], X[:, j])

    lines = [
        "% Auto-generated: upper Pearson, lower Spearman; signed equation zeta",
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        " & " + " & ".join(labels) + r" \\",
        r"\midrule",
    ]
    for i in range(7):
        cells = [labels[i]]
        for j in range(7):
            if i == j:
                v = 1.0
            elif j > i:
                v = P[i, j]
            else:
                v = S[i, j]
            cells.append(f"${v:.2f}$")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="Compute fidelity-vs-delta sweeps and H*_all.png",
    )
    ap.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help="Publish directory (default: results/lipschitz-margin-python)",
    )
    ap.add_argument(
        "--controller-dir",
        type=Path,
        default=CTRL,
        help="Directory with problem9.mat + controllers.csv "
        "(default: paper set; e.g. results/synth-python)",
    )
    ap.add_argument(
        "--max-error",
        type=float,
        default=1e-4,
        help="Nominal error filter for load_controllers",
    )
    ap.add_argument(
        "--FT",
        type=float,
        default=FT,
        help="Fidelity threshold for margins (default 0.999)",
    )
    args = ap.parse_args()
    ft = args.FT

    ctrl_dir = args.controller_dir
    problem = load_problem(ctrl_dir / "problem9.mat")
    controllers = load_controllers(ctrl_dir / "controllers.csv", args.max_error)
    rows: list[dict] = []
    sweeps = {tag: {"X": [], "Y": []} for tag in STRUCTURES}

    for i, c in enumerate(controllers):
        dt = c["tf"] / c["tau"]
        row = {
            "controller": i + 1,
            "fid": c["fid"],
            "err": c["error"],
        }
        print(f"Controller {i+1}/{len(controllers)} fid={c['fid']:.6g}", flush=True)
        for tag in STRUCTURES:
            if tag == "H0":
                C = structure_constant("drift", problem["H0"], dt, c["tau"])
            elif tag == "H1":
                C = structure_constant("control", problem["H1"], dt, c["tau"], c["u1"])
            else:
                C = structure_constant("control", problem["H2"], dt, c["tau"], c["u2"])
            L = lipschitz_constant(ft, problem["dim"], C)
            fid_fn = make_fidelity_fn(
                problem["H0"],
                problem["H1"],
                problem["H2"],
                c["u1"],
                c["u2"],
                problem["Uf"],
                dt,
                tag,
            )
            margin = iterative_margin(fid_fn, L, ft, mu0=0.0, eta=ETA)
            H_list = perturbed_hamiltonians(
                problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag, 0.0
            )
            dH = dH_structure(problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag)
            zeta = differential_sensitivity(H_list, dH, dt, problem["Uf"], n_quad=32)
            row[f"M_{tag}"] = float(margin.M)
            row[f"Mm_{tag}"] = float(margin.M_minus)
            row[f"Mp_{tag}"] = float(margin.M_plus)
            row[f"zeta_{tag}"] = float(zeta)

            if args.sweep:
                span = 1.05 * max(margin.M_minus, margin.M_plus)
                if span <= 0:
                    span = 1e-3
                x = np.linspace(-span, span, 401)
                x, F = fidelity_vs_delta(fid_fn, x)
                sweeps[tag]["X"].append(x)
                sweeps[tag]["Y"].append(1.0 - F)
        rows.append(row)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)

    csv_name = f"margins_table_{ft:g}.csv"
    fields = list(rows[0].keys())
    for dest in (out / csv_name, BUILD / f"margins_table_{ft:g}_python.csv"):
        with dest.open("w", newline="") as f:
            # lineterminator: match the MATLAB peer, which writes LF.
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {dest}")

    tex_name = f"correlations_{ft:g}.tex"
    write_correlation_tex(rows, out / tex_name)
    print(f"Wrote {out / tex_name}")

    if args.no_plots:
        return

    plot_margins_vs_index(
        [r["err"] for r in rows],
        [r["M_H0"] for r in rows],
        [r["M_H1"] for r in rows],
        [r["M_H2"] for r in rows],
        out_path=out / "robustness_margins_fid_err.png",
    )
    plot_margins_vs_sensitivity(
        np.abs([r["zeta_H0"] for r in rows]),
        np.abs([r["zeta_H1"] for r in rows]),
        np.abs([r["zeta_H2"] for r in rows]),
        [r["M_H0"] for r in rows],
        [r["M_H1"] for r in rows],
        [r["M_H2"] for r in rows],
        out_path=out / "robustness_margins_sensitivity.png",
    )
    if args.sweep:
        for tag in STRUCTURES:
            plot_fidelity_error_sweeps(
                sweeps[tag]["X"],
                sweeps[tag]["Y"],
                ft,
                xlabel=rf"Perturbation strength $\delta_{tag[-1]}$",
                xlim=XLIMS[tag],
                out_path=out / f"{tag}_all.png",
            )
    print(f"Published paper deliverables to {out}")


if __name__ == "__main__":
    main()

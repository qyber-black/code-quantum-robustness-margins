#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare the Lipschitz margin with the Kosut-Lidar-Rabitz bound.

Supplementary analysis (not part of the paper's main results): for every
controller and perturbation structure, computes the certified margin M of
Algorithm 1 alongside the margin implied by Theorem 1 of arXiv:2507.01215,
specialised to this closed-system coherent perturbation model
(see qrobustness/kosut.py and README.md).

Peer of matlab/examples/run_time_bandwidth_bound_comparison.m; both write the same CSV
columns (CSV_HEADERS below == qrobustness.compat.kosut_csv_headers) so
scripts/compare_time_bandwidth_bound.py can cross-check them.

Writes results/time-bandwidth-bound-python/kosut_comparison_<FT>.csv and, unless
--no-plots, a scatter of both margins per structure.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from qrobustness import (
    dH_structure,
    iterative_margin,
    kosut_margin,
    lipschitz_constant,
    load_controllers,
    load_problem,
    make_fidelity_fn,
    perturbed_hamiltonians,
    structure_constant,
    uncertainty_rates,
)
from qrobustness.kosut import T_OMEGA_MAX, fidelity_bound_at, time_bandwidth

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
OUT_DIR = ROOT / "results/time-bandwidth-bound-python"

FT = 0.999
ETA = 1e-6
STRUCTURES = ("H0", "H1", "H2")
PER_STRUCTURE = ("M", "KM", "ratio", "KTOb", "Kflb", "wunc", "wavg", "wdev")
#: Must match qrobustness.compat.kosut_csv_headers (MATLAB peer).
CSV_HEADERS = ["controller", "fid", "err"] + [
    f"{f}_{tag}" for tag in STRUCTURES for f in PER_STRUCTURE
]


def plot_comparison(rows: list[dict], ft: float, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from qrobustness.plotting import COLOR_H0, COLOR_H1, COLOR_H2, apply_plot_style

    colors = {"H0": COLOR_H0, "H1": COLOR_H1, "H2": COLOR_H2}
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for tag in STRUCTURES:
        ours = np.array([r[f"M_{tag}"] for r in rows], dtype=float)
        theirs = np.array([r[f"KM_{tag}"] for r in rows], dtype=float)
        ax.scatter(ours, theirs, s=14, color=colors[tag], label=f"${tag[0]}_{tag[1]}$")
    lo = min(
        min(r[f"M_{t}"] for r in rows for t in STRUCTURES),
        min(r[f"KM_{t}"] for r in rows for t in STRUCTURES),
    )
    hi = max(
        max(r[f"M_{t}"] for r in rows for t in STRUCTURES),
        max(r[f"KM_{t}"] for r in rows for t in STRUCTURES),
    )
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="equality")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Lipschitz margin $\mathcal{M}$ (Algorithm 1)")
    ax.set_ylabel(r"Kosut et al. implied margin $\mathcal{M}^{\mathrm{K}}$")
    ax.set_title(rf"$\mathcal{{F}}_T = {ft:g}$")
    ax.legend(loc="best", fontsize=8)
    apply_plot_style(fig)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="Output directory")
    ap.add_argument("--controller-dir", type=Path, default=CTRL)
    ap.add_argument("--max-error", type=float, default=1e-4)
    ap.add_argument("--FT", type=float, default=FT, help="Fidelity threshold (default 0.999)")
    ap.add_argument(
        "--literal-theorem",
        action="store_true",
        help="Evaluate their Theorem 1 literally (F_nom = 1) instead of "
        "absorbing the nominal error eps_0 into the threshold",
    )
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    ft = args.FT

    problem = load_problem(args.controller_dir / "problem9.mat")
    controllers = load_controllers(args.controller_dir / "controllers.csv", args.max_error)

    rows: list[dict] = []
    for i, c in enumerate(controllers):
        dt = c["tf"] / c["tau"]
        eps0 = 0.0 if args.literal_theorem else c["error"]
        row = {"controller": i + 1, "fid": c["fid"], "err": c["error"]}
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
            M = float(iterative_margin(fid_fn, L, ft, mu0=0.0, eta=ETA).M)

            H_list = perturbed_hamiltonians(
                problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag, 0.0
            )
            dH = dH_structure(problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag)
            rates = uncertainty_rates(H_list, dH, dt)
            KM = kosut_margin(rates, ft, nominal_error=eps0)

            row[f"M_{tag}"] = M
            row[f"KM_{tag}"] = KM
            row[f"ratio_{tag}"] = M / KM if KM > 0 else float("inf")
            # The reference bound evaluated at the certified Lipschitz margin.
            row[f"KTOb_{tag}"] = time_bandwidth(rates, M)
            row[f"Kflb_{tag}"] = fidelity_bound_at(rates, M)
            # Per-unit-delta uncertainty measures (their Eq. 28).
            row[f"wunc_{tag}"] = rates.w_unc
            row[f"wavg_{tag}"] = rates.w_avg
            row[f"wdev_{tag}"] = rates.w_dev
        rows.append(row)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"kosut_comparison_{ft:g}.csv"
    assert list(rows[0].keys()) == CSV_HEADERS, "column order must match the MATLAB peer"
    with csv_path.open("w", newline="") as f:
        # lineterminator: match the MATLAB peer, which writes LF.
        w = csv.DictWriter(f, fieldnames=CSV_HEADERS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    print(f"\nSummary (FT={ft:g}, "
          f"{'literal F_nom=1' if args.literal_theorem else 'eps_0 absorbed'}, "
          f"{len(rows)} controllers)")
    print(f"{'struct':>6} {'median M':>12} {'median M^K':>12} {'median M/M^K':>14} "
          f"{'max T*Omega_bnd@M':>18}")
    for tag in STRUCTURES:
        M = np.array([r[f"M_{tag}"] for r in rows])
        KM = np.array([r[f"KM_{tag}"] for r in rows])
        ratio = np.array([r[f"ratio_{tag}"] for r in rows])
        tob = np.array([r[f"KTOb_{tag}"] for r in rows])
        print(f"{tag:>6} {np.median(M):12.4e} {np.median(KM):12.4e} "
              f"{np.median(ratio):14.2f} {tob.max():18.4e}")
    print(f"(their bound is vacuous for T*Omega_bnd >= {T_OMEGA_MAX:.4f} rad)")

    if not args.no_plots:
        png = out / f"kosut_vs_lipschitz_{ft:g}.png"
        plot_comparison(rows, ft, png)
        print(f"Wrote {png}")


if __name__ == "__main__":
    main()

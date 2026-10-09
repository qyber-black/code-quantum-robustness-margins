#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare the iterated margin M with the Kosut-Lidar-Rabitz margin M^K.

For every controller of the main ensemble and structure H0, H1, H2, computes
M and the margin M^K implied by Theorem 1 of arXiv:2507.01215 specialised to
this closed-system coherent model (qrobustness.kosut). Peer of
matlab/examples/run_time_bandwidth_bound_comparison.m with the same columns
(CSV_HEADERS == qrobustness.compat.kosut_csv_headers), cross-checked by
scripts/compare_time_bandwidth_bound.py.

Options: --FT, --out, --controller-dir, --max-error, --absorption (angular
or additive absorption of the nominal error; additive is not conservative),
--uncertainty (constant, or trajectory for M^{K,tri}_tv), --literal-theorem
(F_nom = 1, no absorption), --no-plots.

Writes results/time-bandwidth-bound-python/
kosut_comparison_<FT>[_angular][_tv].csv:
    controller, fid, err: instance and nominal fidelity / error.
    M_<s>, KM_<s>, ratio_<s>: M, M^K and M/M^K for s in H0, H1, H2.
    KTOb_<s>, Kflb_<s>: time-bandwidth product and fidelity lower bound of
        the reference at M.
    wunc_<s>, wavg_<s>, wdev_<s>: per-unit-delta uncertainty measures.
and, unless --no-plots, kosut_vs_lipschitz_<FT><suffix>.png (M^K against M).
"""

from __future__ import annotations

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

from _drivers import DEFAULT_ETA, DEFAULT_MAX_ERROR, base_parser

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
OUT_DIR = ROOT / "results/time-bandwidth-bound-python"

ETA = DEFAULT_ETA
#: Relative bracket tolerance, as in the other drivers (so M here matches M
#: from the multiparameter driver).
MARGIN_TOL = 1e-8
STRUCTURES = ("H0", "H1", "H2")

#: Scatter figure size and dpi (96, as the library's figures).
FIG_SIZE = (5.2, 4.0)
FIG_DPI = 96

PER_STRUCTURE = ("M", "KM", "ratio", "KTOb", "Kflb", "wunc", "wavg", "wdev")
#: Must equal qrobustness.compat.kosut_csv_headers (MATLAB peer).
CSV_HEADERS = ["controller", "fid", "err"] + [
    f"{f}_{tag}" for tag in STRUCTURES for f in PER_STRUCTURE
]


def plot_comparison(rows: list[dict], ft: float, out_path: Path) -> None:
    """Scatter M^K against M per structure, with the equality line."""
    # Local import: the backend is set before pyplot is imported, and
    # --no-plots needs no matplotlib.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from qrobustness.plotting import (
        COLOR_H0,
        COLOR_H1,
        COLOR_H2,
        PNG_METADATA,
        apply_plot_style,
    )

    colors = {"H0": COLOR_H0, "H1": COLOR_H1, "H2": COLOR_H2}
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
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
    # Fixed metadata so the PNG bytes do not depend on the matplotlib version.
    fig.savefig(out_path, dpi=FIG_DPI, metadata=PNG_METADATA)
    plt.close(fig)


def main() -> None:
    """Tabulate both margins for every controller and structure."""
    ap = base_parser(OUT_DIR, description=__doc__)
    ap.add_argument("--controller-dir", type=Path, default=CTRL)
    ap.add_argument("--max-error", type=float, default=DEFAULT_MAX_ERROR)
    ap.add_argument(
        "--absorption",
        choices=("angular", "additive"),
        default="angular",
        help="How the nominal error eps_0 is absorbed into the threshold "
        "(angular is the sufficient correction; additive reproduces "
        "QRM tables and is not a sufficient condition). The "
        "angular CSV carries an _angular suffix so both can coexist.",
    )
    ap.add_argument(
        "--uncertainty",
        choices=("constant", "trajectory"),
        default="constant",
        help="Uncertainty class for M^K: constant delta (as published) or "
        "the certified worst-case over sup-norm-bounded trajectories "
        "(adds a _tv suffix to the CSV; the KM columns then hold M^K_tv).",
    )
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
    controllers = load_controllers(
        args.controller_dir / "controllers.csv", args.max_error
    )

    rows: list[dict] = []
    for i, c in enumerate(controllers):
        dt = c["tf"] / c["tau"]
        eps0 = 0.0 if args.literal_theorem else c["error"]
        row = {"controller": i + 1, "fid": c["fid"], "err": c["error"]}
        print(f"Controller {i + 1}/{len(controllers)} fid={c['fid']:.6g}", flush=True)
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
            M = float(
                iterative_margin(
                    fid_fn, L, ft, mu0=0.0, eta=ETA, margin_tol=MARGIN_TOL
                ).M
            )

            H_list = perturbed_hamiltonians(
                problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag, 0.0
            )
            dH = dH_structure(
                problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag
            )
            rates = uncertainty_rates(H_list, dH, dt)
            KM = kosut_margin(
                rates,
                ft,
                nominal_error=eps0,
                absorption=args.absorption,
                uncertainty=args.uncertainty,
            )

            row[f"M_{tag}"] = M
            row[f"KM_{tag}"] = KM
            row[f"ratio_{tag}"] = M / KM if KM > 0 else float("inf")
            # Reference bound evaluated at M.
            row[f"KTOb_{tag}"] = time_bandwidth(rates, M, args.uncertainty)
            row[f"Kflb_{tag}"] = fidelity_bound_at(rates, M, args.uncertainty)
            # Per-unit-delta uncertainty measures (arXiv:2507.01215, Eq. 28).
            row[f"wunc_{tag}"] = rates.w_unc
            row[f"wavg_{tag}"] = rates.w_avg
            row[f"wdev_{tag}"] = rates.w_dev
        rows.append(row)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_angular" if args.absorption == "angular" else ""
    if args.uncertainty == "trajectory":
        suffix += "_tv"
    csv_path = out / f"kosut_comparison_{ft:g}{suffix}.csv"
    if list(rows[0].keys()) != CSV_HEADERS:
        raise SystemExit(
            "ERROR: column order must match the MATLAB peer; got "
            f"{list(rows[0].keys())!r}"
        )
    with csv_path.open("w", newline="") as f:
        # LF line endings, as the MATLAB peer writes.

        w = csv.DictWriter(f, fieldnames=CSV_HEADERS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    print(
        f"\nSummary (FT={ft:g}, "
        f"{'literal F_nom=1' if args.literal_theorem else f'eps_0 absorbed ({args.absorption})'}, "
        f"{len(rows)} controllers)"
    )
    print(
        f"{'struct':>6} {'median M':>12} {'median M^K':>12} {'median M/M^K':>14} "
        f"{'max T*Omega_bnd@M':>18}"
    )
    for tag in STRUCTURES:
        M = np.array([r[f"M_{tag}"] for r in rows])
        KM = np.array([r[f"KM_{tag}"] for r in rows])
        ratio = np.array([r[f"ratio_{tag}"] for r in rows])
        tob = np.array([r[f"KTOb_{tag}"] for r in rows])
        print(
            f"{tag:>6} {np.median(M):12.4e} {np.median(KM):12.4e} "
            f"{np.median(ratio):14.2f} {tob.max():18.4e}"
        )
    print(f"(their bound is vacuous for T*Omega_bnd >= {T_OMEGA_MAX:.4f} rad)")

    if not args.no_plots:
        png = out / f"kosut_vs_lipschitz_{ft:g}{suffix}.png"
        plot_comparison(rows, ft, png)
        print(f"Wrote {png}")


if __name__ == "__main__":
    main()

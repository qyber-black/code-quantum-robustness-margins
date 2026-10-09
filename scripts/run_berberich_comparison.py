#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Margins M^B implied by the Berberich et al. bound (arXiv:2509.08481, Thm 2.1).

Computes, per controller of the main ensemble and per structure H0, H1, H2,
the margins implied by the Berberich et al. algorithm-level bound in both
uncertainty classes, for comparison with the structured certificates and the
Kosut bound M^K (the xQRM paper, Numerical evaluation, time variation).

Options: --FT, --out, --controller-dir, --max-error; --attack N attacks
MB_ind with the standard adversary for the first N controllers as a spot
check (exit status 1 on a violation); --n-starts, --maxiter, --seed set that
search.

Writes results/time-bandwidth-bound-python/berberich_comparison_<FT>.csv:
    controller, structure, fid, err: instance and nominal fidelity / error.
    MB_ind, MB_sys: M^B for sup-norm trajectories and constant perturbations.
    gamma_ind, gamma_sys: the corresponding gamma rates.
    magnus_ok: 1 if m Delta w_max < pi, required for MB_sys to be valid.

    Fmin_attack: least fidelity the adversary found at MB_ind (NaN if not run).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from qrobustness import (
    dH_structure,
    load_controllers,
    load_problem,
    perturbed_hamiltonians,
    uncertainty_rates,
)
from qrobustness import berberich
from qrobustness.lengthspace import refine as refine_lists
from qrobustness.timevarying import adversarial_fidelity

from _drivers import DEFAULT_MAX_ERROR, REFINEMENTS, VIOLATION_TOL, base_parser

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
OUT_DIR = ROOT / "results/time-bandwidth-bound-python"

STRUCTURES = ("H0", "H1", "H2")

#: Seed strides that keep every (controller, structure, refinement) attack on
#: its own random stream: 300 per controller covers 3 structures x 100, so
#: 100 per structure covers the refinement index.
SEED_STRIDE_CTRL = 300
SEED_STRIDE_STRUCTURE = 100


def main() -> None:
    """Tabulate the Berberich margins, with an optional attack on the first N."""
    ap = base_parser(OUT_DIR, description=__doc__)
    ap.add_argument("--controller-dir", type=Path, default=CTRL)
    ap.add_argument("--max-error", type=float, default=DEFAULT_MAX_ERROR)
    ap.add_argument(
        "--attack",
        type=int,
        default=0,
        help="adversarially attack MB_ind for the first N "
        "controllers (x1/x4/x16 grids, mixed starts)",
    )
    ap.add_argument("--n-starts", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260801)
    args = ap.parse_args()
    ft = args.FT

    problem = load_problem(args.controller_dir / "problem9.mat")
    controllers = load_controllers(
        args.controller_dir / "controllers.csv", args.max_error
    )

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"berberich_comparison_{ft:g}.csv"
    n_viol = 0
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "controller",
                "structure",
                "fid",
                "err",
                "MB_ind",
                "MB_sys",
                "gamma_ind",
                "gamma_sys",
                "magnus_ok",
                "Fmin_attack",
            ],
            lineterminator="\n",
        )
        w.writeheader()
        for i, c in enumerate(controllers):
            dt = c["tf"] / c["tau"]
            for tag in STRUCTURES:
                H_list = perturbed_hamiltonians(
                    problem["H0"],
                    problem["H1"],
                    problem["H2"],
                    c["u1"],
                    c["u2"],
                    tag,
                    0.0,
                )
                dH = dH_structure(
                    problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag
                )
                rates = uncertainty_rates(H_list, dH, dt)
                mb_i = berberich.margin(
                    H_list,
                    dH,
                    dt,
                    ft,
                    nominal_error=c["error"],
                    uncertainty="independent",
                )
                mb_s = berberich.margin(
                    H_list,
                    dH,
                    dt,
                    ft,
                    nominal_error=c["error"],
                    uncertainty="systematic",
                    rates=rates,
                )
                fmin = float("nan")
                if i < args.attack and mb_i.m > 0:
                    fmin = np.inf
                    for g, q in enumerate(REFINEMENTS):
                        Hg, Hhatg = refine_lists(H_list, dH, q)
                        Fq, _, _nfev = adversarial_fidelity(
                            Hg,
                            Hhatg,
                            dt / q,
                            problem["Uf"],
                            mb_i.m,
                            n_starts=args.n_starts,
                            maxiter=args.maxiter,
                            seed=(
                                args.seed
                                + SEED_STRIDE_CTRL * i
                                + SEED_STRIDE_STRUCTURE * STRUCTURES.index(tag)
                                + g
                            ),
                            starts="mixed",
                        )
                        fmin = min(fmin, Fq)
                    if fmin < ft - VIOLATION_TOL:
                        n_viol += 1
                w.writerow(
                    {
                        "controller": i + 1,
                        "structure": tag,
                        "fid": c["fid"],
                        "err": c["error"],
                        "MB_ind": mb_i.m,
                        "MB_sys": mb_s.m,
                        "gamma_ind": mb_i.gamma,
                        "gamma_sys": mb_s.gamma,
                        "magnus_ok": int(mb_s.magnus_ok),
                        "Fmin_attack": fmin,
                    }
                )
                f.flush()
            print(f"ctrl {i + 1:2d}/{len(controllers)} done", flush=True)
    print(f"Wrote {csv_path}; attack violations: {n_viol}")
    sys.exit(1 if n_viol else 0)


if __name__ == "__main__":
    main()

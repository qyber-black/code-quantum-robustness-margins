#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Implied margins of the Berberich et al. algorithm-level bound
(arXiv:2509.08481 Thm 2.1) over the shipped ensemble, both uncertainty
classes, for the like-for-like comparison with the structured
certificates and the Kosut universal bound.

Writes results/time-bandwidth-bound-python/berberich_comparison_<FT>.csv
with columns: controller, structure, fid, err, MB_ind (sup-norm
trajectory class), MB_sys (constant class), gamma_ind, gamma_sys,
magnus_ok.  With --attack N, additionally attacks MB_ind for the first
N controllers (all structures) with the standard adversary as a safety
spot check (their theorem proves the certificate; this guards the
specialisation).
"""

from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
OUT_DIR = ROOT / "results/time-bandwidth-bound-python"

STRUCTURES = ("H0", "H1", "H2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--controller-dir", type=Path, default=CTRL)
    ap.add_argument("--max-error", type=float, default=1e-4)
    ap.add_argument("--FT", type=float, default=0.999)
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
                    for g, q in enumerate((1, 4, 16)):
                        Hg, Hhatg = refine_lists(H_list, dH, q)
                        Fq, _ = adversarial_fidelity(
                            Hg,
                            Hhatg,
                            dt / q,
                            problem["Uf"],
                            mb_i.m,
                            n_starts=args.n_starts,
                            maxiter=args.maxiter,
                            seed=args.seed + 300 * i + 100 * STRUCTURES.index(tag) + g,
                            starts="mixed",
                        )
                        fmin = min(fmin, Fq)
                    if fmin < ft - 1e-10:
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

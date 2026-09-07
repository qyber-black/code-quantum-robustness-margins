#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adversarial min-fidelity as a function of the sup-norm budget.

For one (controller, structure) -- default: controller 16, H1, the
case whose x4-refined trajectory violates the constant-class margin
M^K -- runs the multi-start gradient adversary at a sweep of budgets
on the control grid and on x4/x16 refinements.  The resulting curves,
with the certificates r_0, M^K_tv, r_FS and M^K marked, are the
validity figure of the paper.

Writes results/time-bandwidth-bound-python/budget_sweep_ctrl<i>_<tag>.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from qrobustness import (
    dH_structure,
    perturbed_hamiltonians,
    uncertainty_rates,
)
from qrobustness import kosut
from qrobustness import multiparam as mp
from qrobustness.lengthspace import refine as refine_lists
from qrobustness.timevarying import adversarial_fidelity, fs_margin, uniform_margin

from _drivers import REFINEMENTS, base_parser, load_ensemble, three_structure_specs

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/time-bandwidth-bound-python"

STRUCTURES = ("H0", "H1", "H2")

#: The sweep spans half the smallest certificate to somewhat past the
#: largest, so every marked certificate falls inside the plotted range.
BUDGET_LO_FACTOR = 0.5
BUDGET_HI_FACTOR = 1.4

#: Seed stride keeping each budget's multi-start attack on its own stream;
#: the refinement index occupies the units place.
SEED_STRIDE_BUDGET = 100


def main() -> None:
    """Sweep the budget for one (controller, structure) and record the curves."""
    ap = base_parser(OUT_DIR, description=__doc__)
    ap.add_argument("--controller", type=int, default=16)
    ap.add_argument("--structure", default="H1", choices=STRUCTURES)
    ap.add_argument("--n-budgets", type=int, default=14)
    ap.add_argument("--n-starts", type=int, default=6)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()
    ft = args.FT
    tag = args.structure

    problem, controllers = load_ensemble()
    c = controllers[args.controller - 1]
    dt = c["tf"] / c["tau"]
    H_list = perturbed_hamiltonians(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag, 0.0
    )
    dH = dH_structure(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag
    )

    rates = uncertainty_rates(H_list, dH, dt)
    KM = kosut.margin(rates, ft, nominal_error=c["error"])
    KMtv = kosut.margin(rates, ft, nominal_error=c["error"], uncertainty="trajectory")
    specs = three_structure_specs(problem, c)
    _, L = mp.structure_constants(specs, dt, c["tau"], ft, problem["dim"])
    j = STRUCTURES.index(tag)
    r0 = uniform_margin(L[j], c["fid"], ft)
    rfs = fs_margin(dH, dt, c["fid"], ft).r_fs

    budgets = np.geomspace(BUDGET_LO_FACTOR * r0, BUDGET_HI_FACTOR * KM, args.n_budgets)
    rows = []
    for bi, m in enumerate(budgets):
        row = {"m": float(m)}
        for g, q in enumerate(REFINEMENTS):
            Hr, dHr = refine_lists(H_list, dH, q)
            Fmin, _ = adversarial_fidelity(
                Hr,
                dHr,
                dt / q,
                problem["Uf"],
                float(m),
                n_starts=args.n_starts,
                maxiter=args.maxiter,
                seed=args.seed + SEED_STRIDE_BUDGET * bi + g,
            )
            row[f"Fmin_x{q}"] = Fmin
        rows.append(row)
        print(
            f"budget {bi + 1}/{len(budgets)} m={m:.3e}: "
            + " ".join(f"x{q}={row[f'Fmin_x{q}']:.6f}" for q in REFINEMENTS),
            flush=True,
        )

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"budget_sweep_ctrl{args.controller}_{tag}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        f.write(f"# r0={r0:.6e} rfs={rfs:.6e} KMtv={KMtv:.6e} KM={KM:.6e} FT={ft:g}\n")
    print(f"Wrote {path}")
    print(f"certificates: r0={r0:.3e} KMtv={KMtv:.3e} rfs={rfs:.3e} KM={KM:.3e}")


if __name__ == "__main__":
    main()

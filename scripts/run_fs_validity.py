#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compute and adversarially test the Fubini-Study trajectory margin r_FS.

For every controller and structure: r_FS = (arccos FT - arccos F0) /
(dt sum_k E_k) certifies every measurable trajectory ||delta||_inf <=
r_FS (see qrobustness.timevarying.fs_margin).  The adversary attacks at
budgets r_FS and 1.05 r_FS on the control grid and with every interval
split x4 and x16 (the certificate covers arbitrarily fast trajectories,
and sub-interval sign modulation is the known failure mode of weaker
readings, so the refined grids carry the burden of proof).

Writes results/time-bandwidth-bound-python/fs_validity_<FT>.csv; also
serves as the ensemble record of r_FS.
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from qrobustness import (
    dH_structure,
    load_controllers,
    load_problem,
    perturbed_hamiltonians,
)
from qrobustness.core import gate_fidelity, propagator
from qrobustness.lengthspace import refine as refine_lists
from qrobustness.timevarying import adversarial_fidelity, fs_margin

from _drivers import (
    DEFAULT_MAX_ERROR,
    REFINEMENTS,
    SEED_STRIDE_CTRL,
    SEED_STRIDE_REFINEMENT,
    SEED_STRIDE_STRUCTURE,
    VIOLATION_TOL,
    base_parser,
)

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
OUT_DIR = ROOT / "results/time-bandwidth-bound-python"

#: Budget multiples probed: at the certificate and 5% above it.
BUDGET_FACTORS = (1.0, 1.05)

CSV_HEADERS = (
    ["controller", "structure", "fid", "err", "r_fs", "speed"]
    + [f"Fmin_{key}_x{q}" for q in REFINEMENTS for key in ("m1", "m105")]
    + ["violated"]
)

STRUCTURES = ("H0", "H1", "H2")


def attack_one(job):
    """Attack one (controller, structure) pair; returns the CSV row dict."""
    (i, c, tag, problem, ft, n_starts, maxiter, seed, starts) = job
    dt = c["tf"] / c["tau"]
    H_list = perturbed_hamiltonians(
        problem["H0"],
        problem["H1"],
        problem["H2"],
        c["u1"],
        c["u2"],
        "H0",
        0.0,
    )
    F0 = gate_fidelity(propagator(H_list, dt), problem["Uf"])
    dH = dH_structure(
        problem["H0"],
        problem["H1"],
        problem["H2"],
        c["u1"],
        c["u2"],
        tag,
    )
    res = fs_margin(dH, dt, F0, ft)
    row = {
        "controller": i + 1,
        "structure": tag,
        "fid": F0,
        "err": c["error"],
        "r_fs": res.r_fs,
        "speed": res.speed,
    }
    violated = False
    n_attacks = 0
    for g, q in enumerate(REFINEMENTS):
        Hg, Hhatg = refine_lists(H_list, dH, q)
        for b, fac in enumerate(BUDGET_FACTORS):
            Fmin, _ = adversarial_fidelity(
                Hg,
                Hhatg,
                dt / q,
                problem["Uf"],
                fac * res.r_fs,
                n_starts=n_starts,
                maxiter=maxiter,
                seed=(
                    seed
                    + SEED_STRIDE_CTRL * i
                    + SEED_STRIDE_STRUCTURE * STRUCTURES.index(tag)
                    + SEED_STRIDE_REFINEMENT * g
                    + b
                ),
                starts=starts,
            )
            key = "m1" if fac == 1.0 else "m105"
            row[f"Fmin_{key}_x{q}"] = Fmin
            n_attacks += 1
            if fac == 1.0 and Fmin < ft - VIOLATION_TOL:
                # violation only beyond the numerical allowance
                violated = True
    row["violated"] = int(violated)
    return row, n_attacks


def main() -> None:
    """Attack r_FS on every (controller, structure) pair and record the outcome."""
    ap = base_parser(OUT_DIR, description=__doc__)
    ap.add_argument("--controller-dir", type=Path, default=CTRL)
    ap.add_argument("--max-error", type=float, default=DEFAULT_MAX_ERROR)
    ap.add_argument("--n-starts", type=int, default=6)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument(
        "--starts",
        choices=("legacy", "mixed"),
        default="legacy",
        help="Adversary initialisation: 'mixed' adds sign-modulated boundary starts",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker processes over (controller, structure) pairs",
    )
    ap.add_argument("--first", type=int, default=None)
    args = ap.parse_args()
    ft = args.FT

    problem = load_problem(args.controller_dir / "problem9.mat")
    controllers = load_controllers(
        args.controller_dir / "controllers.csv", args.max_error
    )
    if args.first is not None:
        controllers = controllers[: args.first]

    jobs = [
        (i, c, tag, problem, ft, args.n_starts, args.maxiter, args.seed, args.starts)
        for i, c in enumerate(controllers)
        for tag in STRUCTURES
    ]

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(attack_one, jobs, chunksize=1))
    else:
        results = [attack_one(j) for j in jobs]

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"fs_validity_{ft:g}.csv"
    n_attacks = 0
    n_violations = 0
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADERS, lineterminator="\n")
        w.writeheader()
        for row, na in results:
            n_attacks += na
            n_violations += row["violated"]
            w.writerow(row)
            print(
                f"ctrl {row['controller']:2d}/{len(controllers)} "
                f"{row['structure']}: r_fs={row['r_fs']:.3e} "
                f"Fmin@r_fs(x16)={row['Fmin_m1_x16']:.6f} "
                f"{'VIOLATED' if row['violated'] else 'ok'}",
                flush=True,
            )

    print(
        f"\n{n_attacks} attacks, {n_violations} violations "
        f"(threshold FT={ft:g}, Fubini-Study trajectory margin, "
        f"starts={args.starts})"
    )
    print(f"Wrote {csv_path}")
    sys.exit(1 if n_violations else 0)


if __name__ == "__main__":
    main()

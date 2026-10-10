#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compute the Fubini-Study trajectory margin r_FS and attack it.

For every controller of the main ensemble and structure H0, H1, H2, computes
r_FS = (arccos FT - theta_0)/s_j and the least fidelity the multi-start
adversary finds at budgets r_FS and 1.05 r_FS on the control grid and its x4
and x16 refinements (the xQRM paper, Scenario T, a geometric trajectory
certificate). Exit status 1 if any attack at r_FS violates FT by more than
VIOLATION_TOL.

Options: --FT, --out, --controller-dir, --max-error, --n-starts, --maxiter,
--seed, --starts (legacy or mixed adversary starts), --jobs (worker
processes), --first N (first N controllers only).

Writes results/time-bandwidth-bound-python/fs_validity_<FT>.csv:
    controller, structure, fid, err: instance and nominal fidelity / error.
    r_fs, speed: r_FS and the Choi speed s_j.
    Fmin_m1_x<q>, Fmin_m105_x<q>: adversarial minimum at r_FS and 1.05 r_FS,
        q in 1, 4, 16.
    violated: 1 if an attack at r_FS fell below FT - VIOLATION_TOL.
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

#: Start count of the production fs_validity run. The Makefile does not
#: pass --n-starts, and the default scheme is legacy (no sign modulation).
FS_STARTS = 6

#: Budget multiples probed: at the certificate and 5% above that value.
BUDGET_FACTORS = (1.0, 1.05)

CSV_HEADERS = (
    ["controller", "structure", "fid", "err", "r_fs", "speed"]
    + [f"Fmin_{key}_x{q}" for q in REFINEMENTS for key in ("m1", "m105")]
    + ["violated"]
)

STRUCTURES = ("H0", "H1", "H2")


def attack_one(job):
    """Attack one (controller, structure) pair; return the CSV row dict."""
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
            Fmin, _, _nfev = adversarial_fidelity(
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
                # violation only when beyond the numerical allowance
                violated = True
    row["violated"] = int(violated)
    return row, n_attacks


def main() -> None:
    """Attack r_FS on every (controller, structure) pair and record that outcome."""
    ap = base_parser(OUT_DIR, description=__doc__)
    ap.add_argument("--controller-dir", type=Path, default=CTRL)
    ap.add_argument("--max-error", type=float, default=DEFAULT_MAX_ERROR)
    ap.add_argument("--n-starts", type=int, default=FS_STARTS)
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

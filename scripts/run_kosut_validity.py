#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adversarial validity test of the corrected time-bandwidth margin M^K.

The margin M^K implied by Theorem 1 of arXiv:2507.01215 (with the nominal
error eps_0 absorbed into the threshold via the angular relation, see
qrobustness.kosut.effective_threshold) claims: every uncertainty
trajectory with ||delta||_inf <= M^K keeps the target-gate fidelity at or
above F_T.  This script attacks that claim directly: for every controller
and perturbation structure it minimises the fidelity over piecewise-
constant trajectories with sup-norm budget m = M^K and m = 1.05 M^K (the
latter to confirm the adversary has teeth near the boundary), on the
control grid and on sub-interval refinements of it (trajectories faster
than the control grid are inside the claim, so they must be attacked
too).

A fidelity below F_T at budget M^K is a counterexample to the corrected
bound as specialised here and must be investigated before any paper
claim; at 1.05 M^K a violation is unremarkable (the bound makes no claim
beyond its margin) but indicates how tight the certificate is.

Writes results/time-bandwidth-bound-python/validity_<FT>.csv (one row
per controller x structure).
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
    uncertainty_rates,
)
from qrobustness.kosut import margin as kosut_margin
from qrobustness.lengthspace import refine as refine_lists
from qrobustness.timevarying import adversarial_fidelity

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

STRUCTURES = ("H0", "H1", "H2")
BUDGET_FACTORS = (1.0, 1.05)
DEFAULT_REFINEMENTS = REFINEMENTS


def grid_label(q: int) -> str:
    """Column suffix for a refinement level; the control grid is unlabelled."""
    return "grid" if q == 1 else f"x{q}"


def csv_headers(refinements) -> list:
    """Column names for the chosen refinement levels."""
    return (
        ["controller", "structure", "fid", "err", "KM"]
        + [f"Fmin_{tag}_{grid_label(q)}" for q in refinements for tag in ("m1", "m105")]
        + [
            "violated",
            # Only populated on a violating row: the trajectory's own
            # interaction-picture measures and how far its coherent average
            # exceeds the constant-scaling value at m = M^K.
            "omega_avg_traj",
            "omega_unc_traj",
            "omega_avg_excess",
        ]
    )


def attack_one(job):
    """Attack one (controller, structure) pair; returns the CSV row dict.

    Runs in a worker process; everything needed is in the job tuple.
    """
    (
        i,
        c,
        tag,
        problem,
        ft,
        uncertainty,
        refinements,
        n_starts,
        maxiter,
        seed,
        starts,
    ) = job
    dt = c["tf"] / c["tau"]
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
        problem["H0"],
        problem["H1"],
        problem["H2"],
        c["u1"],
        c["u2"],
        tag,
    )
    rates = uncertainty_rates(H_list, dH, dt)
    KM = kosut_margin(
        rates,
        ft,
        nominal_error=c["error"],
        absorption="angular",
        uncertainty=uncertainty,
    )
    row = {
        "controller": i + 1,
        "structure": tag,
        "fid": c["fid"],
        "err": c["error"],
        "KM": KM,
    }
    violated = False
    n_attacks = 0
    if KM > 0.0:
        for g, q in enumerate(refinements):
            Hg, Hhatg = refine_lists(H_list, dH, q)
            dtg = dt / q
            for b, fac in enumerate(BUDGET_FACTORS):
                m = fac * KM
                Fmin, delta = adversarial_fidelity(
                    Hg,
                    Hhatg,
                    dtg,
                    problem["Uf"],
                    m,
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
                row[f"Fmin_{key}_{grid_label(q)}"] = Fmin
                n_attacks += 1
                # A violation counts only beyond the numerical allowance.
                if fac == 1.0 and Fmin < ft - VIOLATION_TOL:
                    violated = True
                    # The trajectory's OWN interaction-picture measures.
                    # The bound is evaluated at measures generated by
                    # scaling a constant delta; a sign-modulated
                    # trajectory defeats the coherent averaging, so its
                    # Omega_avg is larger than the constant-scaling
                    # value the margin was read off. Recording it is what
                    # lets the paper quote the size of that gap instead
                    # of asserting it. Reuses uncertainty_rates on the
                    # per-interval scaled structure: with
                    # H_unc^(k) = delta_k Hhat^(k), the measures of the
                    # scaled structure at unit delta ARE the
                    # trajectory's measures.
                    dH_traj = [dk * Hk for dk, Hk in zip(delta, Hhatg, strict=True)]
                    rt = uncertainty_rates(Hg, dH_traj, dtg)
                    row["omega_avg_traj"] = rt.w_avg
                    row["omega_unc_traj"] = rt.w_unc
                    # How far the trajectory's coherent average exceeds
                    # the value implied by constant scaling at m = M^K.
                    denom = m * rates.w_avg
                    row["omega_avg_excess"] = (
                        rt.w_avg / denom if denom > 0 else float("nan")
                    )
    else:  # vacuous margin: nothing to attack, nothing claimed
        for q in refinements:
            for key in ("m1", "m105"):
                row[f"Fmin_{key}_{grid_label(q)}"] = float("nan")
    row["violated"] = int(violated)
    return row, n_attacks


def main() -> None:
    """Attack M^K on every selected (controller, structure) pair."""
    ap = base_parser(OUT_DIR, description=__doc__)
    ap.add_argument("--controller-dir", type=Path, default=CTRL)
    ap.add_argument("--max-error", type=float, default=DEFAULT_MAX_ERROR)
    ap.add_argument("--n-starts", type=int, default=6)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument(
        "--refinements",
        type=int,
        nargs="+",
        default=list(DEFAULT_REFINEMENTS),
        help="Sub-intervals per control interval (1 = the control grid itself)",
    )
    ap.add_argument(
        "--starts",
        choices=("legacy", "mixed"),
        default="legacy",
        help="Adversary initialisation: 'mixed' adds "
        "sign-modulated boundary starts (stronger against "
        "coherent-averaging margins)",
    )
    ap.add_argument(
        "--allow-violations",
        action="store_true",
        help="record constant-margin counterexamples without "
        "failing; use only for the paper's documented probe",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Worker processes over (controller, structure) pairs",
    )
    ap.add_argument(
        "--first",
        type=int,
        default=None,
        help="Only the first N controllers (for a quick pass)",
    )
    ap.add_argument(
        "--controller",
        type=int,
        default=None,
        help="Run one 1-based controller index (for witness reproduction)",
    )
    ap.add_argument(
        "--structure",
        choices=STRUCTURES,
        default=None,
        help="Run one uncertainty structure (for witness reproduction)",
    )
    ap.add_argument(
        "--witness-out",
        type=Path,
        default=None,
        help="Write recorded constant-margin counterexamples to this CSV",
    )
    ap.add_argument(
        "--uncertainty",
        choices=("constant", "trajectory"),
        default="constant",
        help="Which M^K to attack: the constant-delta margin "
        "(claims constant perturbations only; sub-grid attacks "
        "probe the sup-norm over-claim) or the trajectory margin "
        "M^K_tv (claims all sup-norm-bounded trajectories; any "
        "violation would be a genuine counterexample). Writes "
        "validity_<FT>_tv.csv for the trajectory variant.",
    )
    args = ap.parse_args()
    ft = args.FT
    refinements = tuple(args.refinements)
    headers = csv_headers(refinements)

    problem = load_problem(args.controller_dir / "problem9.mat")
    controllers = load_controllers(
        args.controller_dir / "controllers.csv", args.max_error
    )
    if args.first is not None:
        controllers = controllers[: args.first]
    if args.controller is not None:
        if not 1 <= args.controller <= len(controllers):
            raise SystemExit(f"ERROR: controller must be in [1, {len(controllers)}]")
        controller_jobs = [(args.controller - 1, controllers[args.controller - 1])]
    else:
        controller_jobs = list(enumerate(controllers))

    jobs = [
        (
            i,
            c,
            tag,
            problem,
            ft,
            args.uncertainty,
            refinements,
            args.n_starts,
            args.maxiter,
            args.seed,
            args.starts,
        )
        for i, c in controller_jobs
        for tag in (STRUCTURES if args.structure is None else (args.structure,))
    ]

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(attack_one, jobs, chunksize=1))
    else:
        results = [attack_one(j) for j in jobs]

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "_tv" if args.uncertainty == "trajectory" else ""
    csv_path = args.out / f"validity_{ft:g}{suffix}.csv"
    n_attacks = 0
    n_violations = 0
    witness_rows = []
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, lineterminator="\n")
        w.writeheader()
        for row, na in results:
            n_attacks += na
            n_violations += row["violated"]
            if row["violated"]:
                witness_rows.append(row)
            w.writerow({h: row.get(h, "") for h in headers})
            fm = row.get(f"Fmin_m1_{grid_label(refinements[0])}", float("nan"))
            print(
                f"ctrl {row['controller']:2d}/{len(controllers)} "
                f"{row['structure']}: KM={row['KM']:.3e} "
                f"Fmin@KM({grid_label(refinements[0])})={fm:.6f} "
                f"{'VIOLATED' if row['violated'] else 'ok'}",
                flush=True,
            )

    print(
        f"\n{n_attacks} attacks, {n_violations} violations "
        f"(threshold FT={ft:g}, angular absorption, "
        f"{args.uncertainty} uncertainty class, "
        f"refinements {refinements}, starts={args.starts})"
    )
    print(f"Wrote {csv_path}")
    witness_path = args.witness_out
    if witness_path is None and args.uncertainty == "constant":
        witness_path = args.out / f"validity_witness_{ft:g}.csv"
    if witness_path is not None:
        witness_path.parent.mkdir(parents=True, exist_ok=True)
        with witness_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers, lineterminator="\n")
            w.writeheader()
            for row in witness_rows:
                w.writerow({h: row.get(h, "") for h in headers})
        print(f"Wrote {witness_path}")
    sys.exit(1 if n_violations and not args.allow_violations else 0)


if __name__ == "__main__":
    main()

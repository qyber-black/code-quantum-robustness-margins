#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ensemble-wide numerical verification of every certificate.

Sweeps the shipped three-qubit ensemble with the verify harness: for
every controller and structure, the constant margin M (dense grid),
the joint polytope, the trajectory Lipschitz lemma, the FS angle bound
and the uniform trajectory certificates at budget r_FS (passed as
FSMargin.r, which is max(r_0, r_FS) defensively and equals r_FS under
the dominance theorem)
(gradient adversary plus adversarially-shaped random trajectories,
control grid and x4 refined); plus the model-independent metric and
absorption lemmas.  Any check whose slack is negative beyond its
derived numerical allowance is reported as a violation.

Writes results/verification-python/verification_<FT>.csv (one row per
check instance) and exits nonzero if any check failed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from qrobustness import (
    iterative_margin,
    load_controllers,
    load_problem,
)
from qrobustness.core import gate_fidelity, propagator
from qrobustness import multiparam as mp
from qrobustness import verify
from qrobustness.timevarying import fs_margin, uniform_margin

from _drivers import DEFAULT_ETA, DEFAULT_FT, DEFAULT_MAX_ERROR, three_structure_specs

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
#: Bracket refinement, matched to the other drivers.
MARGIN_TOL = 1e-8
#: Safe-radius continuation step, matched to the other drivers.
ETA = DEFAULT_ETA

OUT_DIR = ROOT / "results/verification-python"

FT = DEFAULT_FT
STRUCTURES = ("H0", "H1", "H2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--FT", type=float, default=FT)
    ap.add_argument(
        "--controller-dir",
        type=Path,
        default=CTRL,
        help="Ensemble to verify (problem9.mat + controllers.csv). Pointing "
        "this at a freshly synthesised ensemble checks the certificates on "
        "controllers the toolbox has never seen, which the frozen ensemble "
        "cannot do.",
    )
    ap.add_argument(
        "--max-error",
        type=float,
        default=DEFAULT_MAX_ERROR,
        help="Keep controllers with nominal error at or below this. A "
        "synthesised ensemble needs a looser value than the shipped one.",
    )
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--grid-n", type=int, default=101)
    ap.add_argument("--n-starts", type=int, default=4)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--slope-homotopy", type=int, default=8)
    ap.add_argument(
        "--slope-span",
        type=float,
        default=20.0,
        help="probe span as a multiple of the one-step radius",
    )
    ap.add_argument(
        "--slope-n",
        type=int,
        default=24,
        help="random directions for the tv slope check "
        "(times --slope-homotopy points each)",
    )
    args = ap.parse_args()
    ft = args.FT

    problem = load_problem(args.controller_dir / "problem9.mat")
    controllers = load_controllers(
        args.controller_dir / "controllers.csv", args.max_error
    )
    if args.first:
        controllers = controllers[: args.first]
    # Every certificate here is conditional on the nominal fidelity being
    # above the threshold, so a controller that already fails it is out of
    # scope rather than a violation. The shipped ensemble is filtered well
    # below 1 - FT and never exercises this, but a synthesised one does:
    # without the filter the safe radius goes negative and the random
    # trajectory draw raises instead of reporting.
    n_loaded = len(controllers)
    controllers = [c for c in controllers if c["fid"] > ft]
    n_skipped = n_loaded - len(controllers)
    if n_skipped:
        print(
            f"skipping {n_skipped}/{n_loaded} controllers with nominal "
            f"fidelity at or below FT={ft:g}: no certificate applies",
            flush=True,
        )
    if not controllers:
        raise SystemExit(
            f"ERROR: no controller in {args.controller_dir} has error <= "
            f"{args.max_error:g} and nominal fidelity above FT={ft:g}; "
            "the run would verify nothing and pass."
        )

    reports = []

    def record(scope, rep):
        reports.append(
            {
                "scope": scope,
                "check": rep.name,
                "n": rep.n_checks,
                "min_slack": rep.min_slack,
                "tol": rep.tol,
                "passed": int(rep.passed),
                # The slope check reports how much room the bound leaves;
                # blank for checks that have no such quantity.
                "max_fraction_of_bound": rep.details.get("max_fraction_of_bound", ""),
            }
        )
        if not rep.passed:
            print(
                f"VIOLATION {scope} {rep.name}: slack={rep.min_slack:.3e} "
                f"tol={rep.tol:.1e} at {rep.argmin}",
                flush=True,
            )

    for N in (2, 4, 8):
        record(f"N={N}", verify.check_metric_triangle(N, 500, seed=N))
        record(f"N={N}", verify.check_absorption(ft, N, 500, seed=N))

    for ci, c in enumerate(controllers):
        dt = c["tf"] / c["tau"]
        tau = c["tau"]
        H_list = [
            problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
            for k in range(tau)
        ]
        dHs = [
            [problem["H0"]] * tau,
            [c["u1"][k] * problem["H1"] for k in range(tau)],
            [c["u2"][k] * problem["H2"] for k in range(tau)],
        ]
        specs = three_structure_specs(problem, c)
        _, L = mp.structure_constants(specs, dt, tau, ft, problem["dim"])
        L = np.asarray(L)
        F0 = c["fid"]
        scope0 = f"ctrl{ci + 1}"

        def joint_fn(mu, H_list=H_list, dHs=dHs, tau=tau, dt=dt):
            Hp = [
                H_list[k] + sum(mu[j] * dHs[j][k] for j in range(3)) for k in range(tau)
            ]
            return gate_fidelity(propagator(Hp, dt), problem["Uf"])

        record(scope0, verify.check_polytope(joint_fn, L, F0, ft, n=60, seed=ci))
        m_safe = 0.5 * float((F0 - ft) / L.sum())
        record(
            scope0,
            verify.check_lipschitz_pairs(
                H_list, dHs, dt, problem["Uf"], L, ft, m_safe, n=30, seed=ci
            ),
        )
        # Sweep well past the one-step radius: the realised slope is
        # largest near the threshold crossing, and the check's own
        # safe-set filter discards probes that fall below FT, so a
        # generous span explores up to the boundary and no further.
        r_one = args.slope_span * float((F0 - ft) / L.sum())
        record(
            scope0,
            verify.check_tv_slope(
                H_list,
                dHs,
                dt,
                problem["Uf"],
                L,
                ft,
                r_one,
                n=args.slope_n,
                n_homotopy=args.slope_homotopy,
                seed=ci,
            ),
        )

        for j, tag in enumerate(STRUCTURES):
            dH = dHs[j]
            scope = f"ctrl{ci + 1}:{tag}"

            def fid_fn(mu, dH=dH, H_list=H_list, tau=tau, dt=dt):
                return gate_fidelity(
                    propagator([H_list[k] + mu * dH[k] for k in range(tau)], dt),
                    problem["Uf"],
                )

            M = float(
                iterative_margin(
                    fid_fn, L[j], ft, mu0=0.0, eta=ETA, margin_tol=MARGIN_TOL
                ).M
            )
            record(
                scope,
                verify.check_constant_margin(fid_fn, M, ft, n=args.grid_n, tol=2e-6),
            )

            r0 = uniform_margin(L[j], F0, ft)
            fs = fs_margin(dH, dt, F0, ft, r0=r0)
            record(
                scope,
                verify.check_fs_angle(
                    H_list, dH, dt, fs.r_fs, fs.speed, n=25, refine=4, seed=ci
                ),
            )
            record(
                scope,
                verify.check_trajectory_certificate(
                    H_list,
                    dH,
                    dt,
                    problem["Uf"],
                    ft,
                    fs.r,
                    refinements=(1, 4),
                    n_starts=args.n_starts,
                    maxiter=args.maxiter,
                    n_random=30,
                    seed=1000 * ci + j,
                ),
            )
        n_bad = sum(1 for r in reports if not r["passed"])
        print(
            f"ctrl {ci + 1}/{len(controllers)} done "
            f"({len(reports)} checks, {n_bad} violations)",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"verification_{ft:g}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(reports[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(reports)
    n_bad = sum(1 for r in reports if not r["passed"])
    n_probe = sum(r["n"] for r in reports)
    print(
        f"\n{len(reports)} checks ({n_probe} probes), {n_bad} violations. Wrote {path}"
    )
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()

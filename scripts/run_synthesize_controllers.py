#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthesise a controller ensemble on the main three-qubit problem.

Runs n_opt L-BFGS-B / GRAPE optimisations from random initial controls on
the problem in --problem-mat (default: the main ensemble's problem9.mat,
which is copied, not modified). Options: --n-opt, --seed (first seed; run i
uses seed + i), --sigma (initial control scale), --tf, --tau, --maxiter,
--ftol, --out, --problem-mat.

Writes to results/synth-python/ (or --out):
    problem9.mat: copy of the source problem.
    controllers.csv: no header, one row per run in the load_controllers
        format: problem id, run id, tf, tau, error, packed controls.
    meta.json: optimiser settings, source path, the analysis filter, the
        number of runs with error <= DEFAULT_MAX_ERROR, and error min, max,
        median.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from qrobustness import load_problem, optimize_controller, pack_controls

from _drivers import DEFAULT_MAX_ERROR

ROOT = Path(__file__).resolve().parents[1]
PAPER_CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
OUT_DEFAULT = ROOT / "results/synth-python"

TF = 15.0
TAU = 32
PROBLEM_ID = 9


def write_controllers_csv(path: Path, rows: list[dict]) -> None:
    """Write the ensemble in the flat format that load_controllers reads."""
    lines = []
    for r in rows:
        vals = [
            str(PROBLEM_ID),
            str(r["run_id"]),
            f"{r['tf']:g}",
            str(r["tau"]),
            repr(float(r["error"])),
        ]
        x = pack_controls(r["u1"], r["u2"])
        vals.extend(repr(float(v)) for v in x)
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Synthesise the ensemble and store it with its provenance."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-opt", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--tf", type=float, default=TF)
    ap.add_argument("--tau", type=int, default=TAU)
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--ftol", type=float, default=1e-12)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument(
        "--problem-mat",
        type=Path,
        default=PAPER_CTRL / "problem9.mat",
        help="Source problem.mat (copied into out/; paper set left untouched)",
    )
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    dest_mat = out / "problem9.mat"
    shutil.copy2(args.problem_mat, dest_mat)

    problem = load_problem(dest_mat)
    rows = []
    for i in range(args.n_opt):
        seed_i = args.seed + i
        res = optimize_controller(
            problem["H0"],
            problem["H1"],
            problem["H2"],
            problem["Uf"],
            args.tf,
            args.tau,
            sigma=args.sigma,
            seed=seed_i,
            maxiter=args.maxiter,
            ftol=args.ftol,
        )
        rows.append(
            {
                "run_id": i + 1,
                "tf": args.tf,
                "tau": args.tau,
                "error": res.error,
                "u1": res.u1,
                "u2": res.u2,
                "fid": res.fid,
                "fid_init": res.fid_init,
                "n_iter": res.n_iter,
                "success": res.success,
            }
        )
        print(
            f"[{i + 1}/{args.n_opt}] seed={seed_i} "
            f"fid_init={res.fid_init:.6g} fid={res.fid:.6g} err={res.error:.3e} "
            f"iters={res.n_iter}",
            flush=True,
        )

    csv_path = out / "controllers.csv"
    write_controllers_csv(csv_path, rows)

    errs = np.array([r["error"] for r in rows])
    meta = {
        "method": "L-BFGS-B",
        "gradient": "GRAPE",
        "seed": args.seed,
        "n_opt": args.n_opt,
        "tf": args.tf,
        "tau": args.tau,
        "sigma": args.sigma,
        "maxiter": args.maxiter,
        "ftol": args.ftol,
        "problem_mat_source": str(args.problem_mat.resolve()),
        "analysis_filter": (
            f"eps0 <= {DEFAULT_MAX_ERROR:g} "
            "(paper; applied by load_controllers, not here)"
        ),
        # Key name shared with the MATLAB peer; the count uses DEFAULT_MAX_ERROR.
        "n_accepted_1e-4": int(np.sum(errs <= DEFAULT_MAX_ERROR)),
        "error_min": float(errs.min()),
        "error_max": float(errs.max()),
        "error_median": float(np.median(errs)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {csv_path} and {out / 'meta.json'}")
    print(
        f"Accepted with eps<={DEFAULT_MAX_ERROR:g}: "
        f"{meta['n_accepted_1e-4']}/{args.n_opt}"
    )


if __name__ == "__main__":
    main()

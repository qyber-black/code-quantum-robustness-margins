#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthesize an ensemble of fidelity-maximising controllers -> results/synth-python/.

Does not modify data/controllers/problem9_tf15_K32_quasi-newton/.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from qrobustness import load_problem, optimize_controller, pack_controls

ROOT = Path(__file__).resolve().parents[1]
PAPER_CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
OUT_DEFAULT = ROOT / "results/synth-python"

TF = 15.0
TAU = 32
PROBLEM_ID = 9


def write_controllers_csv(path: Path, rows: list[dict]) -> None:
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
    ap = argparse.ArgumentParser()
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
            f"[{i+1}/{args.n_opt}] seed={seed_i} "
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
        "analysis_filter": "eps0 <= 1e-4 (paper; applied by load_controllers, not here)",
        "n_accepted_1e-4": int(np.sum(errs <= 1e-4)),
        "error_min": float(errs.min()),
        "error_max": float(errs.max()),
        "error_median": float(np.median(errs)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {csv_path} and {out / 'meta.json'}")
    print(f"Accepted with eps<=1e-4: {meta['n_accepted_1e-4']}/{args.n_opt}")


if __name__ == "__main__":
    main()

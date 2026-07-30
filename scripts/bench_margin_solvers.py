#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Benchmark selectable margin solvers against Algorithm 1."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from qrobustness import (
    dH_structure,
    differential_sensitivity,
    iterative_margin,
    lipschitz_constant,
    load_controllers,
    load_problem,
    make_fidelity_fn,
    perturbed_hamiltonians,
    structure_constant,
)

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
OUT = ROOT / "results/bench-margin-solvers"

METHODS = (
    "algorithm1",
    "lipschitz_brent",
    "lipschitz_toms748",
    "doubling",
    "newton_probe",
)


def _run_one(name, fidelity_fn, L, FT, eta, method, zeta_fn=None):
    kwargs = dict(
        mu0=0.0,
        eta=eta,
        method=method,
        return_diagnostics=True,
    )
    if method == "newton_probe":
        kwargs["zeta_fn"] = zeta_fn
    t0 = time.perf_counter()
    res = iterative_margin(fidelity_fn, L, FT, **kwargs)
    wall = time.perf_counter() - t0
    cert_m = fidelity_fn(-res.M) >= FT - 1e-12
    cert_p = fidelity_fn(res.M) >= FT - 1e-12
    return {
        "case": name,
        "method": method,
        "M": res.M,
        "M_minus": res.M_minus,
        "M_plus": res.M_plus,
        "n_evals": res.n_evals,
        "n_steps": res.n_steps,
        "wall_s": wall,
        "cert_ok": cert_m and cert_p,
    }


def synthetic_cases():
    FT = 0.99
    eta = 1e-8
    L = 0.05
    mono = lambda mu: max(0.0, 1.0 - 0.05 * abs(mu))

    def zeta_mono(mu: float) -> float:
        if mu > 0:
            return -0.05
        if mu < 0:
            return 0.05
        return -0.05

    # Non-monotone: dips below FT then recovers (stresses aggressive probes).
    def nonmono(mu: float) -> float:
        a = abs(mu)
        if a < 0.1:
            return 1.0 - 0.05 * a
        if a < 0.15:
            return 0.985  # below FT=0.99
        return max(0.0, 0.995 - 0.01 * (a - 0.15))

    def zeta_nonmono(mu: float) -> float:
        # Rough local slope for probing only.
        h = 1e-6
        return (nonmono(mu + h) - nonmono(mu - h)) / (2 * h)

    rows = []
    for method in METHODS:
        rows.append(_run_one("synthetic_mono", mono, L, FT, eta, method, zeta_mono))
        rows.append(_run_one("synthetic_nonmono", nonmono, L, FT, eta, method, zeta_nonmono))
    # Large-L overshoot stress on monotone landscape.
    for method in METHODS:
        rows.append(_run_one("synthetic_mono_L10", mono, 10.0, FT, 1e-6, method, zeta_mono))
    return rows


def case_study_rows(n_controllers: int = 3):
    FT = 0.999
    eta = 1e-6
    problem = load_problem(CTRL / "problem9.mat")
    controllers = load_controllers(CTRL / "controllers.csv", 1e-4)[:n_controllers]
    rows = []
    for ci, c in enumerate(controllers):
        dt = c["tf"] / c["tau"]
        for structure, kind in (("H0", "drift"), ("H1", "control"), ("H2", "control")):
            if kind == "drift":
                C = structure_constant("drift", problem["H0"], dt, c["tau"])
            else:
                Hm = problem[structure]
                um = c["u1"] if structure == "H1" else c["u2"]
                C = structure_constant("control", Hm, dt, c["tau"], um)
            L = lipschitz_constant(FT, problem["dim"], C)
            fid_fn = make_fidelity_fn(
                problem["H0"],
                problem["H1"],
                problem["H2"],
                c["u1"],
                c["u2"],
                problem["Uf"],
                dt,
                structure,
            )

            def zeta_fn(mu, structure=structure, c=c, dt=dt):
                H_list = perturbed_hamiltonians(
                    problem["H0"],
                    problem["H1"],
                    problem["H2"],
                    c["u1"],
                    c["u2"],
                    structure,
                    mu,
                )
                dH = dH_structure(
                    problem["H0"],
                    problem["H1"],
                    problem["H2"],
                    c["u1"],
                    c["u2"],
                    structure,
                )
                return differential_sensitivity(H_list, dH, dt, problem["Uf"], n_quad=16)

            name = f"ctrl{ci}_{structure}"
            for method in METHODS:
                rows.append(_run_one(name, fid_fn, L, FT, eta, method, zeta_fn))
    return rows


def attach_baseline_delta(rows):
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case"], {})[r["method"]] = r
    out = []
    for r in rows:
        base = by_case[r["case"]].get("algorithm1")
        if base is None:
            r = dict(r)
            r["dM"] = ""
            r["eval_ratio"] = ""
        else:
            r = dict(r)
            r["dM"] = abs(r["M"] - base["M"])
            r["eval_ratio"] = r["n_evals"] / base["n_evals"] if base["n_evals"] else ""
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--controllers", type=int, default=3)
    ap.add_argument("--skip-case-study", action="store_true")
    args = ap.parse_args()

    rows = synthetic_cases()
    if not args.skip_case_study:
        rows.extend(case_study_rows(args.controllers))
    rows = attach_baseline_delta(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "bench_margin_solvers.csv"
    fields = [
        "case",
        "method",
        "M",
        "M_minus",
        "M_plus",
        "n_evals",
        "n_steps",
        "wall_s",
        "cert_ok",
        "dM",
        "eval_ratio",
    ]
    with csv_path.open("w", newline="") as f:
        # lineterminator: LF everywhere, matching the MATLAB peers.
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"Wrote {csv_path}")
    print(
        f"{'case':<22} {'method':<20} {'n_evals':>8} {'eval_ratio':>10} " f"{'dM':>12} {'cert':>5}"
    )
    for r in rows:
        er = r["eval_ratio"]
        er_s = f"{er:.3f}" if isinstance(er, float) else str(er)
        dm = r["dM"]
        dm_s = f"{dm:.3e}" if isinstance(dm, float) else str(dm)
        print(
            f"{r['case']:<22} {r['method']:<20} {r['n_evals']:>8} {er_s:>10} "
            f"{dm_s:>12} {str(r['cert_ok']):>5}"
        )


if __name__ == "__main__":
    main()

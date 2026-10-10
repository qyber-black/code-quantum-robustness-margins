#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Accuracy, status and cost of the directional margin brackets.

Computes, for the xQRM paper (Numerical evaluation, targeted checks of the
continuation), the directional margin M and witness M_upper on the main
ensemble along the six coordinate and eight diagonal rays at relative bracket
1e-8, with the evaluation band 0 and VIOLATION_TOL, and on the coordinate
rays also the scalar Lipschitz-step continuation with L_j. Options: --FT,
--out, --jobs (worker processes), --first N (first N controllers only).
Writes results/bracket-audit-python/:

brackets_<FT>.csv
    One row per controller, direction, rule and band.
    controller, direction, rule, band: instance.
    M, M_upper, rel_width: bracket and (M_upper - M)/M.
    reason, n_unresolved, n_evals: status, unresolved probes, evaluations.
timing_<FT>.csv
    Wall-clock cost on the first TIMING_CONTROLLERS controllers, REPEATS
    serial repetitions each.
    controller, repeats, n_evals_dir: instance and diagonal-run evaluations.
    t_preproc_*, t_eval_*, t_dir_*: median and IQR of preprocessing (constants,
        Grams, gauges), one fidelity evaluation, one angular directional run.
        Preprocessing and evaluation samples are means over batches of
        PREPROC_REPEATS and EVAL_REPEATS calls, after WARMUP untimed calls.
    Only the t_* columns vary between runs (volatile in the reproduction
    check).
environment.json
    CPU, platform, Python, NumPy, BLAS and thread settings of the timing run.
"""

from __future__ import annotations

import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from qrobustness import iterative_margin
from qrobustness import multiparam as mp
from qrobustness.core import gate_fidelity, propagator

from _drivers import (
    DEFAULT_ETA,
    ROOT,
    VIOLATION_TOL,
    base_parser,
    load_ensemble,
    three_structure_specs,
    write_rows,
)

OUT_DIR = ROOT / "results/bracket-audit-python"
MARGIN_TOL = 1e-8
BANDS = (0.0, VIOLATION_TOL)
TIMING_CONTROLLERS = 5
REPEATS = 7
EVAL_REPEATS = 50
#: Preprocessing calls per timed sample (one call takes milliseconds).
PREPROC_REPEATS = 10
#: Untimed calls before each timed measurement (first-call costs).
WARMUP = 3


def _setup(problem, c, fT):
    dt = c["tf"] / c["tau"]
    Hs = [
        problem["H0"] + a * problem["H1"] + b * problem["H2"]
        for a, b in zip(c["u1"], c["u2"], strict=True)
    ]
    struct = [
        [problem["H0"]] * len(Hs),
        [a * problem["H1"] for a in c["u1"]],
        [b * problem["H2"] for b in c["u2"]],
    ]
    _, L = mp.structure_constants(
        three_structure_specs(problem, c), dt, len(Hs), fT, problem["dim"]
    )
    G = mp.joint_gauge(struct, dt)
    A = mp.angular_gauge(struct, dt)

    def fid(x):
        return gate_fidelity(
            propagator(
                [
                    h + x[0] * s0 + x[1] * s1 + x[2] * s2
                    for h, s0, s1, s2 in zip(Hs, *struct, strict=True)
                ],
                dt,
            ),
            problem["Uf"],
        )

    return dt, Hs, struct, L, G, A, fid


def _directions():
    names, dirs = [], []
    for j, d in enumerate(np.eye(3)):
        names.append(f"+e{j}")
        dirs.append(d)
        names.append(f"-e{j}")
        dirs.append(-d)
    for d in mp.diagonal_directions(3):
        names.append("diag" + "".join("p" if x > 0 else "m" for x in d))
        dirs.append(d)
    return names, dirs


def _width(res):
    if not np.isfinite(res.M_upper_plus) or res.M_plus <= 0:
        return float("inf")
    return (res.M_upper_plus - res.M_plus) / res.M_plus


def _audit_controller(args):
    idx, problem, c, fT = args
    _, _, _, L, G, A, fid = _setup(problem, c, fT)
    dim = problem["dim"]
    rows = []
    for name, d in zip(*_directions(), strict=True):
        runs = [("angular", None)]
        if name.endswith(("e0", "e1", "e2")):
            runs.append(("precursor", int(name[-1])))
        for rule, axis in runs:
            for band in BANDS:
                if rule == "angular":
                    res = mp.directional_margin(
                        fid,
                        L,
                        fT,
                        d,
                        L_dir=G.L_dir(d, fT, dim),
                        angular_gauge=A,
                        eta=DEFAULT_ETA,
                        margin_tol=MARGIN_TOL,
                        eval_tol=band,
                        return_diagnostics=True,
                    )
                else:
                    sign = 1.0 if name[0] == "+" else -1.0
                    e = np.zeros(3)
                    e[axis] = sign

                    def ray(s, e=e):
                        return fid(s * e)

                    res = iterative_margin(
                        ray,
                        L[axis],
                        fT,
                        eta=DEFAULT_ETA,
                        margin_tol=MARGIN_TOL,
                        eval_tol=band,
                        return_diagnostics=True,
                    )
                rows.append(
                    {
                        "controller": idx,
                        "direction": name,
                        "rule": rule,
                        "band": band,
                        "M": res.M_plus,
                        "M_upper": res.M_upper_plus,
                        "rel_width": _width(res),
                        "reason": res.reason_plus,
                        "n_unresolved": res.n_unresolved,
                        "n_evals": res.n_evals_plus,
                    }
                )
    return rows


def _timed(fn, repeats):
    """Wall-clock times of ``repeats`` calls, after WARMUP untimed calls
    (first-call imports and caches would otherwise inflate the spread)."""
    for _ in range(WARMUP):
        fn()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return np.array(out)


def timing(problem, controllers, fT):
    rows = []
    d = mp.diagonal_directions(3)[0]
    for idx, c in enumerate(controllers[:TIMING_CONTROLLERS], start=1):
        t_pre = _timed(
            lambda c=c: [_setup(problem, c, fT) for _ in range(PREPROC_REPEATS)],
            REPEATS,
        )
        t_pre = t_pre / PREPROC_REPEATS
        _, _, _, L, G, A, fid = _setup(problem, c, fT)
        x = np.array([1e-3, -1e-3, 1e-3])
        t_eval = _timed(
            lambda fid=fid, x=x: [fid(x) for _ in range(EVAL_REPEATS)], REPEATS
        )
        t_eval = t_eval / EVAL_REPEATS
        res = None

        def run(L=L, G=G, A=A, fid=fid):
            nonlocal res
            res = mp.directional_margin(
                fid,
                L,
                fT,
                d,
                L_dir=G.L_dir(d, fT, problem["dim"]),
                angular_gauge=A,
                eta=DEFAULT_ETA,
                margin_tol=MARGIN_TOL,
                return_diagnostics=True,
            )

        t_dir = _timed(run, REPEATS)
        rows.append(
            {
                "controller": idx,
                "repeats": REPEATS,
                "n_evals_dir": res.n_evals,
                "t_preproc_med": float(np.median(t_pre)),
                "t_preproc_iqr": float(np.subtract(*np.percentile(t_pre, [75, 25]))),
                "t_eval_med": float(np.median(t_eval)),
                "t_eval_iqr": float(np.subtract(*np.percentile(t_eval, [75, 25]))),
                "t_dir_med": float(np.median(t_dir)),
                "t_dir_iqr": float(np.subtract(*np.percentile(t_dir, [75, 25]))),
            }
        )
    return rows


def environment():
    cpu = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    blas = np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {})
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": blas.get("name", "unknown"),
        "blas_version": blas.get("version", "unknown"),
        "threads": {
            k: os.environ.get(k, "unset")
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "timing": "serial, one process, after the parallel audit has finished",
    }


def main() -> None:
    ap = base_parser(OUT_DIR, __doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--first", type=int, default=None, help="first N controllers")
    args = ap.parse_args()
    fT, out = args.FT, args.out
    problem, controllers = load_ensemble()
    if args.first:
        controllers = controllers[: args.first]
    tasks = [(i, problem, c, fT) for i, c in enumerate(controllers, start=1)]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        rows = [r for part in ex.map(_audit_controller, tasks) for r in part]
    write_rows(out / f"brackets_{fT}.csv", rows)
    write_rows(out / f"timing_{fT}.csv", timing(problem, controllers, fT))
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(json.dumps(environment(), indent=1) + "\n")
    print(f"Wrote {out / 'environment.json'}")


if __name__ == "__main__":
    main()

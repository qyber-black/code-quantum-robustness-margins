#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fidelity scan over the (mu_1, mu_2) control-structure plane.

For one controller of the shipped ensemble (default: controller 1),
scans the gate fidelity over a grid in the (H1, H2) perturbation plane
(mu_0 = 0), for the true-safe-region figure of the paper: the F = F_T
contour is the ground truth against which the certified cross-polytope
slice and the in-plane directional margins are drawn.

Writes results/multiparameter-margin-python/slice_ctrl<i>_<FT>.npz
(grid axes, fidelity matrix, Lipschitz constants).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from qrobustness.core import gate_fidelity, propagator
from qrobustness import multiparam as mp

from _drivers import base_parser, load_ensemble, three_structure_specs

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/multiparameter-margin-python"


def main() -> None:
    ap = base_parser(OUT_DIR)
    ap.add_argument("--controller", type=int, default=1)
    ap.add_argument("--n", type=int, default=121, help="grid points per axis")
    ap.add_argument(
        "--span",
        type=float,
        default=2.5,
        help="half-width in units of the larger axis margin",
    )
    args = ap.parse_args()

    problem, controllers = load_ensemble()
    c = controllers[args.controller - 1]
    dt = c["tf"] / c["tau"]
    tau = c["tau"]

    specs = three_structure_specs(problem, c)
    _, L = mp.structure_constants(specs, dt, tau, args.FT, problem["dim"])

    dH1 = [c["u1"][k] * problem["H1"] for k in range(tau)]
    dH2 = [c["u2"][k] * problem["H2"] for k in range(tau)]
    H_list = [
        problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
        for k in range(tau)
    ]

    # Scale from the certified single-axis margins (mp CSV not required).
    # The filename must follow --FT: reading the 0.999 table while computing
    # every constant at another threshold silently mixed the two.
    mp_path = args.out / f"multiparam_{args.FT:g}.csv"
    if not mp_path.exists():
        raise SystemExit(
            f"ERROR: missing {mp_path}; run run_multiparameter_case_study.py "
            f"--FT {args.FT:g} first"
        )
    with mp_path.open() as fh:
        mp_rows = list(csv.DictReader(fh))
    row = mp_rows[args.controller - 1]
    half = args.span * max(
        min(float(row["M_+e1"]), float(row["M_-e1"])),
        min(float(row["M_+e2"]), float(row["M_-e2"])),
    )
    ax = np.linspace(-half, half, args.n)
    F = np.empty((args.n, args.n))
    for i, m1 in enumerate(ax):
        for j, m2 in enumerate(ax):
            Hp = [H_list[k] + m1 * dH1[k] + m2 * dH2[k] for k in range(tau)]
            F[i, j] = gate_fidelity(propagator(Hp, dt), problem["Uf"])
        print(f"row {i + 1}/{args.n}", flush=True)

    # In-plane free-region boundaries of the joint Frobenius gauge and
    # the static Choi-angular gauge (radial closed forms; the three
    # nested free regions of the paper's slice figure).
    from qrobustness.core import lipschitz_constant

    dHs = [[problem["H0"]] * tau, dH1, dH2]
    G = mp.joint_gauge(dHs, dt)
    AG = mp.angular_gauge(dHs, dt)
    surplus = c["fid"] - args.FT
    ang_budget = AG.budget(c["fid"], args.FT)
    thetas = np.linspace(0.0, 2.0 * np.pi, 257)
    r_joint = np.empty_like(thetas)
    r_ang = np.empty_like(thetas)
    for t, th in enumerate(thetas):
        d = np.array([0.0, np.cos(th), np.sin(th)])
        Lg = lipschitz_constant(args.FT, problem["dim"], G.C(d))
        r_joint[t] = surplus / Lg if Lg > 0 else np.inf
        Ca = AG.C(d)
        r_ang[t] = ang_budget / Ca if Ca > 0 else np.inf

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"slice_ctrl{args.controller}_{args.FT:g}.npz"
    np.savez(
        path,
        mu1=ax,
        mu2=ax,
        F=F,
        L=np.asarray(L),
        fid=c["fid"],
        FT=args.FT,
        theta=thetas,
        r_joint=r_joint,
        r_angular=r_ang,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

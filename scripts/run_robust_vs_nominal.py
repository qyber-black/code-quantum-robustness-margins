#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Certified margins as a controller-selection tool: robust vs nominal.

On the CNOT model of run_cnot_case_study.py, synthesises an ensemble
of ensemble-robustified controllers (average fidelity over the nine
sign-pattern samples of the three multiplicative structures at
delta_0 = 0.01, plus the nominal) alongside the plain nominal-GRAPE
ensemble, and compares the certified margins: iterated M per
structure, joint gauge inradius, and r_FS.  If the certified margins
separate the two families, the margin is demonstrably useful for
controller selection, not only post-hoc analysis.

Writes results/cnot-python/robust_vs_nominal_<FT>.csv.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from qrobustness import iterative_margin, lipschitz_constant, structure_constant

from _drivers import base_parser, cnot_model, write_rows
from qrobustness.core import gate_fidelity, propagator
from qrobustness import multiparam as mp
from qrobustness.synthesis import grape, grape_robust
from qrobustness.timevarying import (
    adversarial_upper_bound,
    fs_margin,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/cnot-python"

H0, X1, X2, CNOT = cnot_model()
TF, TAU = 4.0, 20
DT = TF / TAU
N = 4
SEED0 = 20260810
N_EACH = 20
DELTA0 = 0.01
STRUCTURES = ("H0", "X1", "X2")


def margins_for(u, ft, adversary=False, seed=0):
    H_list = [H0 + u[0, k] * X1 + u[1, k] * X2 for k in range(TAU)]
    F0 = gate_fidelity(propagator(H_list, DT), CNOT)
    if F0 <= ft:
        return None
    dHs = {
        "H0": [H0] * TAU,
        "X1": [u[0, k] * X1 for k in range(TAU)],
        "X2": [u[1, k] * X2 for k in range(TAU)],
    }
    C = {
        "H0": structure_constant("drift", H0, DT, TAU),
        "X1": structure_constant("control", X1, DT, TAU, u[0]),
        "X2": structure_constant("control", X2, DT, TAU, u[1]),
    }
    L = np.array([lipschitz_constant(ft, N, C[t]) for t in STRUCTURES])
    out = {"fid": F0}
    for j, tag in enumerate(STRUCTURES):
        dH = dHs[tag]

        def fid_fn(mu, dH=dH):
            return gate_fidelity(
                propagator([H_list[k] + mu * dH[k] for k in range(TAU)], DT), CNOT
            )

        out[f"M_{tag}"] = float(
            iterative_margin(fid_fn, L[j], ft, mu0=0.0, eta=1e-6, margin_tol=1e-6).M
        )
        out[f"rfs_{tag}"] = fs_margin(dH, DT, F0, ft).r_fs
        if adversary and tag != "H0":
            # Numerical upper witness on the true trajectory margin
            # M_tv: the smallest sup-norm budget at which the standard
            # adversary exhibits a violating trajectory (evidence layer
            # for the family comparison, not a certificate).
            br = adversarial_upper_bound(
                H_list,
                dH,
                DT,
                CNOT,
                ft,
                out[f"rfs_{tag}"],
                2.0 * out[f"M_{tag}"],
                rel_tol=5e-2,
                seed=seed + 1000 * STRUCTURES.index(tag),
            )
            out[f"mub_{tag}"] = br.m_ub
    G = mp.joint_gauge([dHs[t] for t in STRUCTURES], DT)
    sphere = mp.sphere_directions(3, 100, seed=0)
    out["inradius_gauge"] = min(G.boundary_radius(d, F0 - ft, ft, N) for d in sphere)
    return out


def main() -> None:
    ap = base_parser(OUT_DIR)
    ap.add_argument(
        "--adversary",
        action="store_true",
        help="also bracket the true trajectory margin M_tv "
        "with adversarial upper witnesses (mub_X1, mub_X2)",
    )
    args = ap.parse_args()
    ft = args.FT

    samples = [np.zeros(3)] + [
        DELTA0 * np.array(s) for s in itertools.product([-1, 1], repeat=3)
    ]

    rows = []
    for kind in ("nominal", "robust"):
        kept = 0
        i = 0
        while kept < N_EACH and i < 3 * N_EACH:
            seed = SEED0 + i
            i += 1
            if kind == "nominal":
                r = grape(H0, [X1, X2], CNOT, TF, TAU, seed=seed)
            else:
                r = grape_robust(
                    H0, [X1, X2], CNOT, TF, TAU, seed=seed, sample_deltas=samples
                )
            if r.error > 1e-4:
                continue
            m = margins_for(r.u, ft, adversary=args.adversary, seed=seed)
            if m is None:
                continue
            kept += 1
            rows.append({"kind": kind, "seed": seed, "err": r.error, **m})
            print(
                f"{kind} {kept}/{N_EACH}: eps0={r.error:.2e} "
                f"M_X1={m['M_X1']:.3e} gauge_inr={m['inradius_gauge']:.3e}",
                flush=True,
            )

    out = args.out
    path = write_rows(out / f"robust_vs_nominal_{ft:g}.csv", rows)

    keys = (
        ["M_" + t for t in STRUCTURES]
        + ["rfs_" + t for t in STRUCTURES]
        + ["inradius_gauge"]
    )
    if args.adversary:
        keys += ["mub_X1", "mub_X2"]
    for key in keys:
        a = np.array([r[key] for r in rows if r["kind"] == "nominal"])
        b = np.array([r[key] for r in rows if r["kind"] == "robust"])
        print(
            f"{key}: nominal med {np.median(a):.3e}  robust med "
            f"{np.median(b):.3e}  ratio {np.median(b) / np.median(a):.2f}"
        )


if __name__ == "__main__":
    main()

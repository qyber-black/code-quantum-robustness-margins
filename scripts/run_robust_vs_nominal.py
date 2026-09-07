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

from _drivers import (
    DEFAULT_ETA,
    DEFAULT_MAX_ERROR,
    base_parser,
    cnot_model,
    write_rows,
)
from qrobustness.core import gate_fidelity, propagator
from qrobustness import multiparam as mp
from qrobustness.synthesis import grape, grape_robust
from qrobustness.timevarying import (
    adversarial_upper_bound,
    toggling_frame_integral,
    fs_margin,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/cnot-python"
#: Safe-radius continuation step, matched to the other drivers.
ETA = DEFAULT_ETA
#: Bracket refinement, 1e-8 as in every other closed-system driver. This
#: was 1e-6 -- the open-system value -- so the margins here were certified
#: to a looser bracket than the ones they are compared against.
MARGIN_TOL = 1e-8

H0, X1, X2, CNOT = cnot_model()
TF, TAU = 4.0, 20
DT = TF / TAU
#: Hilbert-space dimension of the two-qubit model.
DIM = 4
SEED0 = 20260810
N_EACH = 20
DELTA0 = 0.01
STRUCTURES = ("H0", "X1", "X2")
N_PARAMS = len(STRUCTURES)

#: Synthesis attempts allowed per family before giving up on the quota.
ATTEMPTS_PER_KEPT = 3

#: Sampled inradius of the joint gauge region. Fewer directions than the
#: dedicated joint-gauge driver uses: here the inradius is one column
#: among many, not the result being reported.
N_SPHERE = 100
SPHERE_SEED = 0

#: Adversarial witness search: bracket up to twice the iterated margin, to
#: a relative width of 5%, with a per-structure seed stride.
ADVERSARY_SPAN = 2.0
ADVERSARY_RTOL = 5e-2
ADVERSARY_SEED_STRIDE = 1000


def margins_for(u, ft, adversary=False, seed=0):
    """Every certificate for one controller, or None if it misses FT."""
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
    L = np.array([lipschitz_constant(ft, DIM, C[t]) for t in STRUCTURES])
    out = {"fid": F0}
    for j, tag in enumerate(STRUCTURES):
        dH = dHs[tag]

        def fid_fn(mu, dH=dH):
            return gate_fidelity(
                propagator([H_list[k] + mu * dH[k] for k in range(TAU)], DT), CNOT
            )

        out[f"M_{tag}"] = float(
            iterative_margin(
                fid_fn, L[j], ft, mu0=0.0, eta=ETA, margin_tol=MARGIN_TOL
            ).M
        )
        out[f"rfs_{tag}"] = fs_margin(dH, DT, F0, ft).r_fs
        if adversary and tag != "H0":
            # Numerical upper witness on the true trajectory margin
            # M_tv: a sup-norm budget at which the standard adversary
            # exhibits a violating trajectory -- found, not least, since
            # the search is a heuristic local one (evidence layer for the
            # family comparison, not a certificate).
            br = adversarial_upper_bound(
                H_list,
                dH,
                DT,
                CNOT,
                ft,
                out[f"rfs_{tag}"],
                ADVERSARY_SPAN * out[f"M_{tag}"],
                rel_tol=ADVERSARY_RTOL,
                seed=seed + ADVERSARY_SEED_STRIDE * STRUCTURES.index(tag),
            )
            out[f"madv_{tag}"] = br.m_adv
            # Re-evaluate the witness through the independent propagator
            # route. The adversary works in the eigenbasis; this rebuilds
            # the same trajectory with expm, so a violation that survives
            # is not an artefact of one route's arithmetic. Recorded, not
            # asserted: the appendix quotes these controllers.
            if br.delta_adv is None:
                out[f"madvF_{tag}"] = float("nan")
            else:
                out[f"madvF_{tag}"] = float(
                    gate_fidelity(
                        propagator(
                            [
                                H_list[k] + float(br.delta_adv[k]) * dH[k]
                                for k in range(TAU)
                            ],
                            DT,
                        ),
                        CNOT,
                    )
                )
    # What static robustification is implicitly minimising: the coherent
    # sum over the gate in the toggling frame. The free certificates
    # charge the pulse area instead, so recording both separates
    # "arranged cancellation" from "spent amplitude".
    for tag in STRUCTURES:
        out[f"cancel_{tag}"] = float(
            np.linalg.norm(toggling_frame_integral(H_list, dHs[tag], DT), "fro")
        )
    G = mp.joint_gauge([dHs[t] for t in STRUCTURES], DT)
    sphere = mp.sphere_directions(N_PARAMS, N_SPHERE, seed=SPHERE_SEED)
    out["inradius_gauge"] = min(G.boundary_radius(d, F0 - ft, ft, DIM) for d in sphere)
    return out


def main() -> None:
    """Synthesise both families and compare their certified margins."""
    ap = base_parser(OUT_DIR, description=__doc__)
    ap.add_argument(
        "--adversary",
        action="store_true",
        help="also bracket the true trajectory margin M_tv "
        "with adversarial upper witnesses (madv_X1, madv_X2)",
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
        while kept < N_EACH and i < ATTEMPTS_PER_KEPT * N_EACH:
            seed = SEED0 + i
            i += 1
            if kind == "nominal":
                r = grape(H0, [X1, X2], CNOT, TF, TAU, seed=seed)
            else:
                r = grape_robust(
                    H0, [X1, X2], CNOT, TF, TAU, seed=seed, sample_deltas=samples
                )
            if r.error > DEFAULT_MAX_ERROR:
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
    write_rows(out / f"robust_vs_nominal_{ft:g}.csv", rows)

    keys = (
        ["M_" + t for t in STRUCTURES]
        + ["rfs_" + t for t in STRUCTURES]
        + ["cancel_" + t for t in STRUCTURES]
        + ["inradius_gauge"]
    )
    if args.adversary:
        keys += ["madv_X1", "madv_X2"]
    for key in keys:
        a = np.array([r[key] for r in rows if r["kind"] == "nominal"])
        b = np.array([r[key] for r in rows if r["kind"] == "robust"])
        print(
            f"{key}: nominal med {np.median(a):.3e}  robust med "
            f"{np.median(b):.3e}  ratio {np.median(b) / np.median(a):.2f}"
        )


if __name__ == "__main__":
    main()

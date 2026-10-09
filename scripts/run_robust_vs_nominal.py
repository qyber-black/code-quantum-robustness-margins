#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Certified margins of robustified CNOT controllers and nominal ones.

On the CNOT model of _drivers.cnot_model, synthesises N_EACH plain GRAPE
controllers and N_EACH ensemble-robustified ones (average fidelity over the
nominal and the eight sign patterns of the three multiplicative structures
at delta_0 = 0.01), and computes per controller, for the structures H0, X1,
X2, M with its witness M_upper, r_FS, the toggling-frame cancellation and the
sampled C_joint gauge inradius; with --adversary also the adversarial
witness m_adv on the time-varying margin for X1 and X2 (the xQRM paper,
Numerical evaluation, robustness is uncertainty-class dependent).
Options: --FT, --out, --adversary.

Writes results/cnot-python/robust_vs_nominal_<FT>.csv:
    kind, seed, err, fid: family (nominal or robust), seed, nominal error and
        fidelity.
    budget, area_X1, area_X2: arccos FT - theta_0 and the pulse areas
        sum_k |u_j(k)| (r_FS factors into these).
    M_<s>, Mupper_<s>, rfs_<s>: M, M_upper on the side that sets M, r_FS.
    madv_<s>, madvF_<s> (X1, X2, --adversary only): m_adv and the fidelity of
        the witness re-evaluated with the expm propagator (NaN if none).
    cancel_<s>: Frobenius norm of the toggling-frame integral of the structure.
    inradius_gauge: sampled inradius of the C_joint gauge region.
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
#: Continuation hand-over surplus eta, as in the other drivers.
ETA = DEFAULT_ETA
#: Relative bracket tolerance, 1e-8 as in every closed-system driver.
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

#: Directions for the sampled joint-gauge inradius.
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
    # r_FS = (arccos FT - theta_0) / s_j with s_j proportional to the pulse
    # area, so both factors are recorded per controller.
    out = {
        "fid": F0,
        "budget": float(np.arccos(ft) - np.arccos(min(F0, 1.0))),
        "area_X1": float(np.sum(np.abs(u[0]))),
        "area_X2": float(np.sum(np.abs(u[1]))),
    }
    for j, tag in enumerate(STRUCTURES):
        dH = dHs[tag]

        def fid_fn(mu, dH=dH):
            return gate_fidelity(
                propagator([H_list[k] + mu * dH[k] for k in range(TAU)], DT), CNOT
            )

        res = iterative_margin(
            fid_fn, L[j], ft, mu0=0.0, eta=ETA, margin_tol=MARGIN_TOL
        )
        out[f"M_{tag}"] = float(res.M)
        # Unsafe witness M_upper on the side that sets M.
        side = "minus" if res.M_minus <= res.M_plus else "plus"
        out[f"Mupper_{tag}"] = float(getattr(res, f"M_upper_{side}"))
        out[f"rfs_{tag}"] = fs_margin(dH, DT, F0, ft).r_fs
        if adversary and tag != "H0":
            # Adversarial upper witness m_adv on M_tv: a budget at which a
            # violating trajectory was found (not necessarily the least).
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
                n_starts=4,
                starts="legacy",
                maxiter=200,
            )
            out[f"madv_{tag}"] = br.m_adv
            # Re-evaluate the witness with the expm propagator (the adversary
            # works in the eigenbasis).
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
    # Frobenius norm of the toggling-frame integral of each structure.

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

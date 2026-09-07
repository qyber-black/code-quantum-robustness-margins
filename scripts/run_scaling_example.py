#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Four-qubit scaling example: closed-system certificates at N = 16.

System: 4-qubit Ising chain with symmetry-breaking local detunings,

    H0 = 2 pi J sum_q Z_q Z_{q+1} + 2 pi sum_q d_q Z_q,
    J = 0.5,  d = (0.11, -0.07, 0.05, -0.13)

(without the detunings the global spin flip X^{ox4} commutes with every
generator and generic targets are unreachable), individually
addressable x drives X_1..X_4 (p = 5 structures: multiplicative drift
plus the four controls), a fixed Haar-random target (seed 20260801),
t_f = 24, tau = 96.  Controllers are
synthesised in-repository (grape_ensemble, deterministic seeds).

For each controller and structure: iterated margin M with bracket,
uniform time-varying radii r_0 and r_FS, universal-bound margins
M^K / M^K_tv (angular absorption), joint 5-parameter polytope inradii,
and the off-nominal fidelity evaluations each certificate class costs --
the point of the example, and the one cost measure that does not move
with machine load.

Writes results/scaling-python/scaling4q_margins_<FT>.csv and
data/controllers/chain4q_tf24_K96_lbfgs/controllers.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import unitary_group

from qrobustness import (
    iterative_margin,
    lipschitz_constant,
    structure_constant,
)
from qrobustness.core import gate_fidelity, propagator
from qrobustness import kosut
from qrobustness import multiparam as mp
from qrobustness.synthesis import grape_ensemble
from qrobustness.timevarying import fs_margin, uniform_margin

from _drivers import (
    DEFAULT_ETA,
    DEFAULT_MAX_ERROR,
    EYE2,
    PAULI_X,
    PAULI_Z,
    base_parser,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/scaling-python"
DATA_DIR = ROOT / "data/controllers/chain4q_tf24_K96_lbfgs"

NQ = 4
DIM = 2**NQ

#: Ising coupling and the target's Haar seed, both quoted in the docstring
#: above and used once each below.
J_COUPLING = 0.5
TARGET_SEED = 20260801


def local(op, q):
    """Embed a single-qubit operator on qubit ``q`` of the chain."""
    mats = [EYE2] * NQ
    mats[q] = op
    M = mats[0]
    for m in mats[1:]:
        M = np.kron(M, m)
    return M


DETUNE = (0.11, -0.07, 0.05, -0.13)
H0 = 2 * np.pi * J_COUPLING * sum(
    local(PAULI_Z, q) @ local(PAULI_Z, q + 1) for q in range(NQ - 1)
) + 2 * np.pi * sum(d * local(PAULI_Z, q) for q, d in enumerate(DETUNE))
XS = [local(PAULI_X, q) for q in range(NQ)]
UF = np.asarray(unitary_group.rvs(DIM, random_state=TARGET_SEED), dtype=complex)

TF = 24.0
TAU = 96
DT = TF / TAU
ETA = DEFAULT_ETA
MARGIN_TOL = 1e-8
SEED0 = 20260802
N_ATTEMPTS = 20
N_KEEP = 10
STRUCTURES = ("H0",) + tuple(f"X{q + 1}" for q in range(NQ))


def main() -> None:
    """Synthesise the four-qubit ensemble and certify every structure."""
    ap = base_parser(OUT_DIR, description=__doc__)
    ap.add_argument("--maxiter", type=int, default=3000)
    args = ap.parse_args()
    ft = args.FT

    print("Synthesising ensemble...", flush=True)
    ens = grape_ensemble(
        H0,
        XS,
        UF,
        TF,
        TAU,
        n_attempts=N_ATTEMPTS,
        max_error=DEFAULT_MAX_ERROR,
        seed0=SEED0,
        maxiter=args.maxiter,
        verbose=True,
    )[:N_KEEP]
    print(f"kept {len(ens)} controllers", flush=True)
    if not ens:
        raise SystemExit("no controller reached the error threshold")

    # The synthesised ensemble is an input artefact, not a result, so it
    # normally lands in data/. But a scratch run (--out) must not touch the
    # committed ensemble: check_reproducible refuses drivers that lack --out
    # precisely to protect the reference tree, and writing outside --out
    # defeated that guard. Anchor it to --out whenever one is given.
    data_dir = DATA_DIR if args.out == OUT_DIR else args.out / DATA_DIR.name
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "controllers.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            ["seed", "fid", "err"]
            + [f"u{j + 1}_{k}" for j in range(NQ) for k in range(TAU)]
        )
        for r in ens:
            w.writerow(
                [r.seed, f"{r.fidelity:.16e}", f"{r.error:.16e}"]
                + [f"{v:.16e}" for v in r.u.ravel()]
            )
    np.savez(
        data_dir / "problem.npz",
        H0=H0,
        Uf=UF,
        tf=TF,
        tau=TAU,
        **{f"X{q + 1}": XS[q] for q in range(NQ)},
    )

    rows = []
    for ci, r in enumerate(ens):
        u = r.u
        H_list = [H0 + sum(u[j, k] * XS[j] for j in range(NQ)) for k in range(TAU)]
        F0 = gate_fidelity(propagator(H_list, DT), UF)
        dHs = {"H0": [H0] * TAU}
        for j in range(NQ):
            dHs[f"X{j + 1}"] = [u[j, k] * XS[j] for k in range(TAU)]

        C = {"H0": structure_constant("drift", H0, DT, TAU)}
        for j in range(NQ):
            C[f"X{j + 1}"] = structure_constant("control", XS[j], DT, TAU, u[j])
        L = np.array([lipschitz_constant(ft, DIM, C[t]) for t in STRUCTURES])

        row = {"controller": ci + 1, "seed": r.seed, "fid": F0, "err": 1.0 - F0}
        n_evals_iter = n_steps_iter = 0
        for j, tag in enumerate(STRUCTURES):
            dH = dHs[tag]

            def fid_fn(mu, dH=dH, H_list=H_list):
                return gate_fidelity(
                    propagator([H_list[k] + mu * dH[k] for k in range(TAU)], DT), UF
                )

            res = iterative_margin(
                fid_fn,
                L[j],
                ft,
                mu0=0.0,
                eta=ETA,
                margin_tol=MARGIN_TOL,
                return_diagnostics=True,
            )
            # Deterministic cost measure. The evaluation counter is built
            # unconditionally inside iterative_margin, so asking for it
            # changes what is reported and not what is computed.
            n_evals_iter += res.n_evals
            n_steps_iter += res.n_steps
            fs = fs_margin(dH, DT, F0, ft)
            rates = kosut.uncertainty_rates(H_list, dH, DT)
            KM = kosut.margin(rates, ft, nominal_error=1.0 - F0)
            KMtv = kosut.margin(
                rates, ft, nominal_error=1.0 - F0, uncertainty="trajectory"
            )

            row[f"M_{tag}"] = float(res.M)
            row[f"r0_{tag}"] = uniform_margin(L[j], F0, ft)
            row[f"rfs_{tag}"] = fs.r_fs
            row[f"KM_{tag}"] = KM
            row[f"KMtv_{tag}"] = KMtv
        P = mp.safe_polytope(np.zeros(len(L)), L, F0, ft)
        row["inradius_l2"] = P.inradius_l2
        # Off-nominal fidelity evaluations are the reproducible cost
        # measure. No wall-clock is recorded: it depends on machine load,
        # so it is not reproducible and nothing in the paper quotes it.
        # The geometric certificate and the universal-bound measures need
        # none: both are closed form in the structure constants.
        row["n_evals_iter"] = n_evals_iter
        row["n_steps_iter"] = n_steps_iter
        row["n_evals_fs"] = 0
        row["n_evals_kosut"] = 0
        rows.append(row)
        print(
            f"ctrl {ci + 1}/{len(ens)}: fid={F0:.8f} "
            f"M_X1={row['M_X1']:.3e} rfs_X1={row['rfs_X1']:.3e} "
            f"[{n_evals_iter} evals]",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"scaling4q_margins_{ft:g}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")

    print(f"\nSummary ({len(rows)} controllers, N={DIM}, FT={ft:g})")
    for tag in STRUCTURES:
        M = np.array([r[f"M_{tag}"] for r in rows])
        rfs = np.array([r[f"rfs_{tag}"] for r in rows])
        r0 = np.array([r[f"r0_{tag}"] for r in rows])
        KM = np.array([r[f"KM_{tag}"] for r in rows])
        KMtv = np.array([r[f"KMtv_{tag}"] for r in rows])
        print(
            f"{tag:>3}: med M={np.median(M):.3e} "
            f"M/KM={np.median(M / KM):.2f} rfs/r0={np.median(rfs / r0):.2f} "
            f"rfs/KMtv={np.median(rfs / KMtv):.2f} "
            f"rfs>KMtv frac={np.mean(rfs > KMtv):.2f}"
        )
    ne = np.array([r["n_evals_iter"] for r in rows])
    print(
        f"n_evals_iter: median {np.median(ne):.0f} off-nominal fidelity "
        f"evaluations per controller ({len(STRUCTURES)} structures)"
    )


if __name__ == "__main__":
    main()

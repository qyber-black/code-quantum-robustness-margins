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
and wall-clock cost per certificate class (the point of the example).

Writes results/scaling-python/scaling4q_margins_<FT>.csv and
data/controllers/chain4q_tf24_K96_lbfgs/controllers.csv.
"""

from __future__ import annotations

import argparse
import csv
import time
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

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/scaling-python"
DATA_DIR = ROOT / "data/controllers/chain4q_tf24_K96_lbfgs"

NQ = 4
N = 2**NQ
Z = np.diag([1.0, -1.0]).astype(complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def local(op, q):
    mats = [I2] * NQ
    mats[q] = op
    M = mats[0]
    for m in mats[1:]:
        M = np.kron(M, m)
    return M


DETUNE = (0.11, -0.07, 0.05, -0.13)
H0 = 2 * np.pi * 0.5 * sum(
    local(Z, q) @ local(Z, q + 1) for q in range(NQ - 1)
) + 2 * np.pi * sum(d * local(Z, q) for q, d in enumerate(DETUNE))
XS = [local(X, q) for q in range(NQ)]
UF = np.asarray(unitary_group.rvs(N, random_state=20260801), dtype=complex)

TF = 24.0
TAU = 96
DT = TF / TAU
ETA = 1e-6
MARGIN_TOL = 1e-8
SEED0 = 20260802
N_ATTEMPTS = 20
N_KEEP = 10
STRUCTURES = ("H0",) + tuple(f"X{q + 1}" for q in range(NQ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--FT", type=float, default=0.999)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--maxiter", type=int, default=3000)
    args = ap.parse_args()
    ft = args.FT

    t0 = time.perf_counter()
    print("Synthesising ensemble...", flush=True)
    ens = grape_ensemble(
        H0,
        XS,
        UF,
        TF,
        TAU,
        n_attempts=N_ATTEMPTS,
        max_error=1e-4,
        seed0=SEED0,
        maxiter=args.maxiter,
        verbose=True,
    )[:N_KEEP]
    t_synth = time.perf_counter() - t0
    print(f"kept {len(ens)} controllers ({t_synth:.1f} s)", flush=True)
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
        L = np.array([lipschitz_constant(ft, N, C[t]) for t in STRUCTURES])

        row = {"controller": ci + 1, "seed": r.seed, "fid": F0, "err": 1.0 - F0}
        t_iter = t_fs = t_kosut = 0.0
        n_evals_iter = n_steps_iter = 0
        for j, tag in enumerate(STRUCTURES):
            dH = dHs[tag]

            def fid_fn(mu, dH=dH):
                return gate_fidelity(
                    propagator([H_list[k] + mu * dH[k] for k in range(TAU)], DT), UF
                )

            t1 = time.perf_counter()
            res = iterative_margin(
                fid_fn,
                L[j],
                ft,
                mu0=0.0,
                eta=ETA,
                margin_tol=MARGIN_TOL,
                return_diagnostics=True,
            )
            t_iter += time.perf_counter() - t1
            # Deterministic cost measure. The evaluation counter is built
            # unconditionally inside iterative_margin, so asking for it
            # changes what is reported and not what is computed.
            n_evals_iter += res.n_evals
            n_steps_iter += res.n_steps
            t1 = time.perf_counter()
            fs = fs_margin(dH, DT, F0, ft)
            t_fs += time.perf_counter() - t1
            t1 = time.perf_counter()
            rates = kosut.uncertainty_rates(H_list, dH, DT)
            KM = kosut.margin(rates, ft, nominal_error=1.0 - F0)
            KMtv = kosut.margin(
                rates, ft, nominal_error=1.0 - F0, uncertainty="trajectory"
            )
            t_kosut += time.perf_counter() - t1

            row[f"M_{tag}"] = float(res.M)
            row[f"r0_{tag}"] = uniform_margin(L[j], F0, ft)
            row[f"rfs_{tag}"] = fs.r_fs
            row[f"KM_{tag}"] = KM
            row[f"KMtv_{tag}"] = KMtv
        P = mp.safe_polytope(np.zeros(len(L)), L, F0, ft)
        row["inradius_l2"] = P.inradius_l2
        # Off-nominal fidelity evaluations are the reproducible cost
        # measure; the wall-clock beside them depends on machine load and
        # is diagnostic only (see VOLATILE_COLUMNS in check_reproducible).
        # The geometric certificate and the universal-bound measures need
        # none: both are closed form in the structure constants.
        row["n_evals_iter"] = n_evals_iter
        row["n_steps_iter"] = n_steps_iter
        row["n_evals_fs"] = 0
        row["n_evals_kosut"] = 0
        row["t_iter_s"] = t_iter
        row["t_fs_s"] = t_fs
        row["t_kosut_s"] = t_kosut
        rows.append(row)
        print(
            f"ctrl {ci + 1}/{len(ens)}: fid={F0:.8f} "
            f"M_X1={row['M_X1']:.3e} rfs_X1={row['rfs_X1']:.3e} "
            f"[iter {t_iter:.1f}s, fs {t_fs * 1e3:.0f}ms, "
            f"kosut {t_kosut:.1f}s]",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"scaling4q_margins_{ft:g}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")

    print(
        f"\nSummary ({len(rows)} controllers, N={N}, FT={ft:g}; "
        f"synthesis {t_synth:.0f} s)"
    )
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
    for key in ("t_iter_s", "t_fs_s", "t_kosut_s"):
        v = np.array([r[key] for r in rows])
        print(
            f"{key}: median {np.median(v):.2f} s per controller "
            f"({len(STRUCTURES)} structures)"
        )


if __name__ == "__main__":
    main()

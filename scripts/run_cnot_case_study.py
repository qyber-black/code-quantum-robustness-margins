#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two-qubit CNOT case study: synthesis plus the full margin suite.

System: always-on Ising coupling with fixed opposite detunings,

    H0 = 2 pi J ZZ + pi D (Z1 - Z2),   J = 0.5, D = 0.1,

controls X1, X2 (individually addressable x drives), target CNOT,
t_f = 4, tau = 20.  Controllers are synthesised in-repository by
qrobustness.synthesis.grape_ensemble with deterministic seeds, so the
whole study reproduces from this script alone.

Per controller and structure (drift H0, controls X1, X2,
multiplicative as in the main study) the script computes the iterated
margin M with bracket, the uniform time-varying radii r_0 and r_FS,
the universal-bound margins M^K / M^K_tv (angular absorption), the
joint polytope inradii, and the common-rate local-dephasing margin
(certified M_gamma vs the bisected true crossing gamma*).

Writes results/cnot-python/cnot_margins_<FT>.csv and
data/controllers/cnot_tf4_K20_lbfgs/controllers.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from qrobustness import (
    iterative_margin,
    lipschitz_constant,
    structure_constant,
)
from qrobustness.core import gate_fidelity, propagator
from qrobustness import kosut
from qrobustness import lindblad as lb
from qrobustness import multiparam as mp
from qrobustness.synthesis import grape_ensemble
from qrobustness.timevarying import fs_margin, uniform_margin

from _drivers import (
    DEFAULT_ETA,
    DEFAULT_MAX_ERROR,
    EYE2,
    PAULI_Z,
    base_parser,
    cnot_model,
    true_crossing,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/cnot-python"
DATA_DIR = ROOT / "data/controllers/cnot_tf4_K20_lbfgs"

H0, X1, X2, CNOT = cnot_model()

TF = 4.0
TAU = 20
DT = TF / TAU
#: Hilbert-space dimension of the two-qubit model.
DIM = 4
ETA = DEFAULT_ETA
MARGIN_TOL = 1e-8
#: Bracket refinement for the open-system margin. Looser than the
#: closed-system MARGIN_TOL because each probe is a Liouville-space
#: propagation, which is orders of magnitude dearer.
MARGIN_TOL_OPEN = 1e-6
SEED0 = 20260801
N_ATTEMPTS = 60
N_KEEP = 50
STRUCTURES = ("H0", "X1", "X2")


def hamiltonians(u):
    """Per-interval nominal Hamiltonians for a control pair."""
    return [H0 + u[0, k] * X1 + u[1, k] * X2 for k in range(TAU)]


def structure_lists(u):
    """Per-interval perturbation structures, multiplicative as in the
    main study: drift delta*H0, control delta*u_jk*Xj."""
    return {
        "H0": [H0] * TAU,
        "X1": [u[0, k] * X1 for k in range(TAU)],
        "X2": [u[1, k] * X2 for k in range(TAU)],
    }


def main() -> None:
    """Synthesise the CNOT ensemble and compute the full margin suite."""
    ap = base_parser(OUT_DIR, description=__doc__)
    args = ap.parse_args()
    ft = args.FT
    ft_pro = ft**2

    print("Synthesising ensemble...", flush=True)
    ens = grape_ensemble(
        H0,
        [X1, X2],
        CNOT,
        TF,
        TAU,
        n_attempts=N_ATTEMPTS,
        max_error=DEFAULT_MAX_ERROR,
        seed0=SEED0,
        verbose=True,
    )[:N_KEEP]
    print(f"kept {len(ens)} controllers", flush=True)

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
            + [f"u1_{k}" for k in range(TAU)]
            + [f"u2_{k}" for k in range(TAU)]
        )
        for r in ens:
            w.writerow(
                [r.seed, f"{r.fidelity:.16e}", f"{r.error:.16e}"]
                + [f"{v:.16e}" for v in r.u[0]]
                + [f"{v:.16e}" for v in r.u[1]]
            )
    np.savez(data_dir / "problem.npz", H0=H0, X1=X1, X2=X2, Uf=CNOT, tf=TF, tau=TAU)

    # Dissipative structure: local dephasing, common rate.
    G_gamma = lb.dissipator(np.kron(PAULI_Z, EYE2)) + lb.dissipator(
        np.kron(EYE2, PAULI_Z)
    )
    dn = lb.diamond_norm(G_gamma)
    print(f"dnorm(dephasing) = {dn.value:.6f} ({dn.status})", flush=True)
    L_gamma = lb.rate_lipschitz(dn.value, TF)

    rows = []
    for ci, r in enumerate(ens):
        u = r.u
        H_list = hamiltonians(u)
        F0 = gate_fidelity(propagator(H_list, DT), CNOT)
        dHs = structure_lists(u)

        # Lipschitz constants per structure.
        C = {
            "H0": structure_constant("drift", H0, DT, TAU),
            "X1": structure_constant("control", X1, DT, TAU, u[0]),
            "X2": structure_constant("control", X2, DT, TAU, u[1]),
        }
        L = np.array([lipschitz_constant(ft, DIM, C[t]) for t in STRUCTURES])

        row = {"controller": ci + 1, "seed": r.seed, "fid": F0, "err": 1.0 - F0}
        for j, tag in enumerate(STRUCTURES):
            dH = dHs[tag]

            def fid_fn(mu, dH=dH, H_list=H_list):
                return gate_fidelity(
                    propagator([H_list[k] + mu * dH[k] for k in range(TAU)], DT), CNOT
                )

            res = iterative_margin(
                fid_fn, L[j], ft, mu0=0.0, eta=ETA, margin_tol=MARGIN_TOL
            )
            M = float(res.M)
            r0 = uniform_margin(L[j], F0, ft)
            fs = fs_margin(dH, DT, F0, ft, r0=r0)
            rates = kosut.uncertainty_rates(H_list, dH, DT)
            row[f"M_{tag}"] = M
            row[f"r0_{tag}"] = r0
            row[f"rfs_{tag}"] = fs.r_fs
            row[f"KM_{tag}"] = kosut.margin(rates, ft, nominal_error=1.0 - F0)
            row[f"KMtv_{tag}"] = kosut.margin(
                rates, ft, nominal_error=1.0 - F0, uncertainty="trajectory"
            )
        # Joint polytope inradii.
        P = mp.safe_polytope(np.zeros(3), L, F0, ft)
        row["inradius_l2"] = P.inradius_l2
        row["inradius_linf"] = P.inradius_linf

        # Common-rate dephasing margin and true crossing.
        GH_list = [lb.hamiltonian_superop(H) for H in H_list]

        def F_pro(gamma: float, GH_list=GH_list) -> float:
            G_list = [GH + float(gamma) * G_gamma for GH in GH_list]
            return lb.process_fidelity(lb.channel(G_list, DT), CNOT)

        F0p = F_pro(0.0)
        if F0p > ft_pro:
            om = lb.open_margin(F_pro, L_gamma, ft_pro, margin_tol=MARGIN_TOL_OPEN)
            row["M_gamma"] = om.M_plus
            row["r0_gamma"] = (F0p - ft_pro) / L_gamma
            row["gamma_star"] = true_crossing(F_pro, ft_pro, om.M_plus)
        else:
            row["M_gamma"] = row["r0_gamma"] = row["gamma_star"] = float("nan")

        rows.append(row)
        print(
            f"ctrl {ci + 1:2d}/{len(ens)}: fid={F0:.8f} "
            f"M_X1={row['M_X1']:.3e} rfs_X1={row['rfs_X1']:.3e} "
            f"KMtv_X1={row['KMtv_X1']:.3e} "
            f"M_gamma={row['M_gamma']:.3e}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"cnot_margins_{ft:g}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")

    print(f"\nSummary ({len(rows)} controllers, FT={ft:g})")
    for tag in STRUCTURES:
        M = np.array([r[f"M_{tag}"] for r in rows])
        rfs = np.array([r[f"rfs_{tag}"] for r in rows])
        r0 = np.array([r[f"r0_{tag}"] for r in rows])
        KM = np.array([r[f"KM_{tag}"] for r in rows])
        KMtv = np.array([r[f"KMtv_{tag}"] for r in rows])
        print(
            f"{tag:>3}: med M={np.median(M):.3e} M/KM={np.median(M / KM):.2f} "
            f"rfs/r0={np.median(rfs / r0):.2f} "
            f"rfs/KMtv={np.median(rfs / KMtv):.2f} "
            f"rfs>KMtv frac={np.mean(rfs > KMtv):.2f}"
        )
    Mg = np.array([r["M_gamma"] for r in rows])
    gs = np.array([r["gamma_star"] for r in rows])
    ok = np.isfinite(Mg)
    print(
        f"dephasing: med M_gamma={np.median(Mg[ok]):.3e} "
        f"med gamma*/M_gamma={np.median(gs[ok] / Mg[ok]):.3f}"
    )


if __name__ == "__main__":
    main()

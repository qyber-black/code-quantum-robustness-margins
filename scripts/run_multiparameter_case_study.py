#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Joint static and time-varying margins, p = 3, on the main ensemble.

For each controller, with the structures H0, H1, H2, computes the
Lipschitz constants L_j, the safe-polytope axis radii and inradii, the
uniform time-varying radius r_0 per structure and jointly, and the
directional margin M with witness M_upper on the six axis and eight
diagonal directions using the C_joint directional constant (or the
C^stat_FS angular step with --step angular). With --adversary N it also
brackets the time-varying margin for H1 on the first N controllers,
r_FS <= M_tv <= m_adv, together with the constant-class bracket [M, M_upper]
and the constancy-gap endpoints from the same run. (The xQRM paper, Scenario
J, and Numerical evaluation, joint coherent margins and time variation.)

Options: --FT, --out, --max-error, --controllers N (first N; 0 = all),
--adversary N, --step (lipschitz or angular).

Writes results/multiparameter-margin-python/:
multiparam_<FT>.csv (multiparam_<FT>_angular.csv with --step angular)
    controller, fid, err: instance and nominal fidelity / error.
    L_H0, L_H1, L_H2: L_j.
    poly_r_H0, poly_r_H1, poly_r_H2, inradius_linf, inradius_l2: safe-polytope
        axis radii and inradii.
    r0_joint, r0_H0, r0_H1, r0_H2: r_0 jointly and per structure.
    M_<d>, Mupper_<d>, nev_<d>: M, M_upper and evaluation count for d in
        +e0..+e2, -e0..-e2 and diag<ppp..mmm>.
    rang_<d> (--step angular only): one-step angular radius
        (arccos F_T - arccos F_0) / C_FS(d) at the nominal point.
tv_bracket_<FT>.csv (only with --adversary)
    controller, structure, FT: instance.
    r0, r_fs: r_0 and r_FS.
    M_const, M_const_upper: constant-class bracket [M, M_upper].
    m_adv, F_at_adv, adv_violated: adversarial upper witness m_adv, the
        fidelity found there, and whether a violation was exhibited.
    gap_lower, gap_upper: constancy-gap endpoints (empty if unavailable).
    n_adversary_evals: fidelity evaluations of the adversarial searches.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from qrobustness import (
    dH_structure,
    iterative_margin,
    make_fidelity_fn,
)
from qrobustness import multiparam as mp
from qrobustness.lengthspace import margin_from
from qrobustness import timevarying as tv

from _drivers import DEFAULT_FT, DEFAULT_MAX_ERROR, load_ensemble, three_structure_specs

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/multiparameter-margin-python"
FT = DEFAULT_FT
MARGIN_TOL = 1e-8
STRUCTURES = ("H0", "H1", "H2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--FT", type=float, default=FT)
    ap.add_argument("--max-error", type=float, default=DEFAULT_MAX_ERROR)
    ap.add_argument(
        "--controllers",
        type=int,
        default=0,
        help="limit to the first n controllers (0 = all)",
    )
    ap.add_argument(
        "--adversary",
        type=int,
        default=0,
        help="run the time-varying adversary for the first n controllers",
    )
    ap.add_argument(
        "--step",
        choices=("lipschitz", "angular"),
        default="lipschitz",
        help="Directional continuation rule: the Lipschitz "
        "surplus step (default; writes multiparam_<FT>.csv) or "
        "the angular step, never shorter "
        "(writes multiparam_<FT>_angular.csv)",
    )
    args = ap.parse_args()
    ft = args.FT

    problem, controllers = load_ensemble(max_error=args.max_error)
    if args.controllers:
        controllers = controllers[: args.controllers]

    directions = np.vstack([mp.axis_directions(3), mp.diagonal_directions(3)])
    dir_names = (
        [f"+e{j}" for j in range(3)]
        + [f"-e{j}" for j in range(3)]
        + [
            "diag" + "".join("p" if s > 0 else "m" for s in d * np.sqrt(3))
            for d in mp.diagonal_directions(3)
        ]
    )

    rows = []
    tv_rows = []
    for ci, c in enumerate(controllers):
        dt = c["tf"] / c["tau"]
        specs = three_structure_specs(problem, c)
        C, L = mp.structure_constants(specs, dt, c["tau"], ft, problem["dim"])
        fn = mp.make_multiparam_fidelity_fn(
            problem["H0"],
            problem["H1"],
            problem["H2"],
            c["u1"],
            c["u2"],
            problem["Uf"],
            dt,
            STRUCTURES,
        )
        F0 = fn(np.zeros(3))
        P = mp.safe_polytope(np.zeros(3), L, F0, ft)

        row = {
            "controller": ci + 1,
            "fid": c["fid"],
            "err": c["error"],
            "L_H0": L[0],
            "L_H1": L[1],
            "L_H2": L[2],
            "poly_r_H0": P.axis_radii[0],
            "poly_r_H1": P.axis_radii[1],
            "poly_r_H2": P.axis_radii[2],
            "inradius_linf": P.inradius_linf,
            "inradius_l2": P.inradius_l2,
            "r0_joint": tv.uniform_margin(L, F0, ft),
        }
        for j, tag in enumerate(STRUCTURES):
            row[f"r0_{tag}"] = tv.uniform_margin(L[j], F0, ft)
        dHs = [
            [problem["H0"]] * c["tau"],
            [c["u1"][k] * problem["H1"] for k in range(c["tau"])],
            [c["u2"][k] * problem["H2"] for k in range(c["tau"])],
        ]
        G = mp.joint_gauge(dHs, dt)
        AG = mp.angular_gauge(dHs, dt) if args.step == "angular" else None
        for d, name in zip(directions, dir_names, strict=True):
            # Directional constant of the joint gauge; with --step angular
            # the iteration steps by the angular radius instead.
            L_dir = G.L_dir(d, ft, problem["dim"])
            res = mp.directional_margin(
                fn,
                L,
                ft,
                d,
                margin_tol=MARGIN_TOL,
                L_dir=L_dir,
                angular_gauge=AG,
                return_diagnostics=True,
            )
            row[f"M_{name}"] = res.M
            row[f"Mupper_{name}"] = res.M_upper
            row[f"nev_{name}"] = res.n_evals
            if AG is not None:
                row[f"rang_{name}"] = margin_from(
                    float(np.arccos(ft)) - float(np.arccos(min(F0, 1.0))), AG.C(d)
                )
        rows.append(row)
        print(
            f"controller {ci + 1}/{len(controllers)}  "
            f"inradius_l2={row['inradius_l2']:.3e}  "
            f"M_worst={min(row[f'M_{n}'] for n in dir_names):.3e}",
            flush=True,
        )

        if ci < args.adversary:
            tau = c["tau"]
            H_list = [
                problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
                for k in range(tau)
            ]
            Hhat = dH_structure(
                problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], "H1"
            )
            scalar_fn = make_fidelity_fn(
                problem["H0"],
                problem["H1"],
                problem["H2"],
                c["u1"],
                c["u2"],
                problem["Uf"],
                dt,
                "H1",
            )
            F0_scalar = scalar_fn(0.0)
            r0 = tv.uniform_margin(L[1], F0_scalar, ft)
            # Constant-class bracket at the directional tolerance. M_upper is
            # infinite unless an unsafe point was evaluated; the gap's upper
            # endpoint is then left empty.
            res_const = iterative_margin(scalar_fn, L[1], ft, margin_tol=MARGIN_TOL)
            M = res_const.M
            M_upper = res_const.M_upper
            # Lower trajectory certificate r_FS for the same instance.
            fsm = tv.fs_margin(Hhat, dt, F0_scalar, ft, r0)
            br = tv.adversarial_upper_bound(
                H_list,
                Hhat,
                dt,
                problem["Uf"],
                ft,
                r0,
                2.0 * M,
                rel_tol=2e-2,
                seed=1000 + ci,
                n_starts=4,
                starts="legacy",
                maxiter=200,
            )
            # delta_adv is set only when a violating trajectory was found.

            adv_violated = br.delta_adv is not None
            gap_lo = max(0.0, M - br.m_adv) if adv_violated else None
            gap_hi = (M_upper - fsm.r_fs) if np.isfinite(M_upper) else None
            if gap_lo is not None and gap_hi is not None and gap_lo > gap_hi:
                raise SystemExit(
                    f"ERROR: controller {ci + 1} structure H1 at FT={ft:g} gives "
                    f"gap_lower={gap_lo:.6e} > gap_upper={gap_hi:.6e}. The two "
                    "endpoints come from inconsistent records (M_const/m_adv "
                    "against M_const_upper/r_FS); the inputs must be fixed "
                    "rather than the interval clipped."
                )
            tv_rows.append(
                {
                    "controller": ci + 1,
                    "structure": "H1",
                    "FT": ft,
                    "r0": br.r0,
                    "r_fs": fsm.r_fs,
                    "M_const": M,
                    "M_const_upper": M_upper,
                    "m_adv": br.m_adv,
                    "F_at_adv": br.F_at_adv,
                    "adv_violated": int(adv_violated),
                    "gap_lower": "" if gap_lo is None else gap_lo,
                    "gap_upper": "" if gap_hi is None else gap_hi,
                    "n_adversary_evals": br.n_evals,
                }
            )
            print(
                f"  tv bracket H1: r_FS={fsm.r_fs:.3e} <= M_tv <= {br.m_adv:.3e}"
                f"  (M_const in [{M:.3e}, {M_upper:.3e}]; constancy gap in ["
                + ("n/a" if gap_lo is None else f"{gap_lo:.3e}")
                + ", "
                + ("n/a" if gap_hi is None else f"{gap_hi:.3e}")
                + "])",
                flush=True,
            )

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_angular" if args.step == "angular" else ""
    path = out / f"multiparam_{ft:g}{suffix}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")
    if tv_rows:
        path = out / f"tv_bracket_{ft:g}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=list(tv_rows[0].keys()), lineterminator="\n"
            )
            w.writeheader()
            w.writerows(tv_rows)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

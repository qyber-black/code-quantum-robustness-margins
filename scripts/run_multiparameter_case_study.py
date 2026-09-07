#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-parameter and time-varying margins for the paper-2 case study.

Per controller (p = 3 joint structures H0, H1, H2):
  - certified safe-polytope radii (per axis, l2 and linf inradii);
  - directional margins along the 6 axis and 8 diagonal directions
    (Euclidean-normalised), with margin_tol brackets;
  - uniform time-varying margins r0 per structure and joint;
  - optionally (--adversary N) time-varying brackets [r_FS, m_adv] for
    the first N controllers on structure H1, together with the
    constant-class bracket [M_const, M_const_upper] at the same
    controller, structure and threshold, and the constancy-gap endpoints
    formed from them. Everything the gap needs sits in one record, so no
    later join can pair margins computed to different tolerances.

Writes results/multiparameter-margin-python/multiparam_<FT>.csv and
tv_bracket_<FT>.csv.
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
        "surplus step (default; reproduces the historical "
        "output byte-for-byte) or the dominating Choi-angular "
        "step (same crossings, fewer evaluations; writes "
        "multiparam_<FT>_angular.csv)",
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
            # Sharper directional constant of the joint gauge
            # (Theorem gauge); the separable sum_j L_j |d_j| is the
            # fallback relaxation.  With --step angular the iteration
            # steps and certifies with the dominating Choi-angular
            # radius instead.
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
            # The constancy gap needs BOTH ends of the constant-class
            # bracket, so the scalar margin is run with the same
            # tolerance as the directional ones. M_upper is finite only
            # when an unsafe constant perturbation was actually
            # evaluated; a domain edge or an exhausted search leaves it
            # infinite, and the gap's upper endpoint is then unavailable
            # rather than clipped to something finite.
            res_const = iterative_margin(scalar_fn, L[1], ft, margin_tol=MARGIN_TOL)
            M = res_const.M
            M_upper = res_const.M_upper
            # The trajectory lower certificate for the same controller,
            # structure and threshold, computed here rather than joined
            # in later from a separate run.
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
            )
            # An upper witness exists only if the adversary exhibited a
            # violating trajectory; adversarial_upper_bound returns
            # delta_adv exactly then.
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
                    "n_adversary_calls": br.n_evals,
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

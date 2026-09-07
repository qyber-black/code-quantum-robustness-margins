#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Combined-structure gauge region versus the weighted cross-polytope.

For every controller of the shipped ensemble, evaluates the certified
free region of the joint gauge C_joint(x) = sum_k Delta ||sum_j x_j
Hhat_j^(k)||_F against the separable cross-polytope: certified radii
along the eight normalised diagonals (where cancellations between
structures matter; on the axes the two coincide), and Euclidean
inradii estimated over a 200-direction sphere design.  No fidelity
evaluations are involved on either side.

Writes results/multiparameter-margin-python/joint_gauge_<FT>.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qrobustness import dH_structure

from _drivers import base_parser, load_ensemble, three_structure_specs, write_rows
from qrobustness import multiparam as mp
from qrobustness.timevarying import fs_margin, fs_margin_joint

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/multiparameter-margin-python"

STRUCTURES = ("H0", "H1", "H2")
N_PARAMS = len(STRUCTURES)

#: Directions used for the Euclidean inradius. A sampled minimum over a
#: sphere design is an ESTIMATE and is reported as one; the certified
#: inradius beside it is what any claim rests on. The seed is fixed so the
#: estimate is reproducible.
N_SPHERE = 200
SPHERE_SEED = 0


def main() -> None:
    """Compare the joint gauge region with the separable cross-polytope."""
    ap = base_parser(OUT_DIR, description=__doc__)
    args = ap.parse_args()
    ft = args.FT

    problem, controllers = load_ensemble()
    dim = problem["dim"]

    diagonals = mp.diagonal_directions(N_PARAMS)
    sphere = mp.sphere_directions(N_PARAMS, N_SPHERE, seed=SPHERE_SEED)

    rows = []
    for i, c in enumerate(controllers):
        dt = c["tf"] / c["tau"]
        dHs = [
            dH_structure(
                problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag
            )
            for tag in STRUCTURES
        ]
        G = mp.joint_gauge(dHs, dt)
        A = mp.angular_gauge(dHs, dt)
        specs = three_structure_specs(problem, c)
        _, L = mp.structure_constants(specs, dt, c["tau"], ft, dim)
        L = np.asarray(L)
        surplus = c["fid"] - ft

        diag_gain = []
        for d in diagonals:
            sep = surplus / float(np.abs(d) @ L)
            diag_gain.append(G.boundary_radius(d, surplus, ft, dim) / sep)
        # Trajectory class: vertex Gram-gauge vs separable budget for
        # the equal-budget box m = m (1,...,1) (Theorem fs, joint form).
        ell, budget = fs_margin_joint(dHs, dt, c["fid"], ft)
        s_sep = sum(fs_margin(dH, dt, c["fid"], ft).speed for dH in dHs)
        m_joint = budget / ell(np.ones(N_PARAMS))
        m_sep = budget / s_sep
        traj_gain = m_joint / m_sep

        r_gauge_est = min(
            G.boundary_radius(d, surplus, ft, dim) for d in sphere
        )  # sampled ESTIMATE only
        r_gauge_cert = G.inradius_certified(surplus, ft, dim)
        r_poly = min(surplus / float(np.abs(d) @ L) for d in sphere)
        rows.append(
            {
                "controller": i + 1,
                "fid": c["fid"],
                "diag_gain_min": float(np.min(diag_gain)),
                "diag_gain_med": float(np.median(diag_gain)),
                "diag_gain_max": float(np.max(diag_gain)),
                "inradius_gauge_est": r_gauge_est,
                "inradius_gauge_cert": r_gauge_cert,
                "inradius_poly": r_poly,
                "inradius_gain_est": r_gauge_est / r_poly,
                "inradius_gain_cert": r_gauge_cert / r_poly,
                "traj_box_joint": m_joint,
                "traj_box_sep": m_sep,
                "traj_gain": traj_gain,
                "ang_diag_gain": float(
                    np.median(
                        [
                            A.boundary_radius(d, c["fid"], ft)
                            / G.boundary_radius(d, surplus, ft, dim)
                            for d in diagonals
                        ]
                    )
                ),
                "ang_inradius_cert": A.inradius_certified(c["fid"], ft),
                "ang_inradius_gain": (
                    A.inradius_certified(c["fid"], ft) / r_gauge_cert
                ),
            }
        )
        print(
            f"ctrl {i + 1:2d}/{len(controllers)}: diag gain "
            f"{np.median(diag_gain):.3f}  inradius gain cert "
            f"{r_gauge_cert / r_poly:.3f} (est {r_gauge_est / r_poly:.3f})",
            flush=True,
        )

    out = args.out
    write_rows(out / f"joint_gauge_{ft:g}.csv", rows)
    for key in (
        "diag_gain_med",
        "inradius_gain_cert",
        "inradius_gain_est",
        "traj_gain",
        "ang_diag_gain",
        "ang_inradius_gain",
    ):
        v = np.array([r[key] for r in rows])
        print(f"{key}: median {np.median(v):.3f} min {v.min():.3f} max {v.max():.3f}")


if __name__ == "__main__":
    main()

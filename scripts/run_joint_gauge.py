#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The combined-structure gauge region against the weighted cross-polytope.

For every controller of the main ensemble, computes the C_joint gauge radii
against the separable cross-polytope along the eight diagonals, the
Euclidean inradii (certified, and a sampled estimate over N_SPHERE
directions), the equal-budget trajectory box of the joint form of r_FS
against its separable sum, and the angular (C^stat_FS) gains over the joint
gauge (the xQRM paper, Scenario J, the combined-structure gauge). No
fidelity evaluations. Options: --FT, --out.

Writes results/multiparameter-margin-python/joint_gauge_<FT>.csv:
    controller, fid: instance and nominal fidelity.
    diag_gain_min, _med, _max: gauge radius over polytope radius on the
        diagonals.
    inradius_gauge_est, inradius_gauge_cert, inradius_poly: inradii.
    inradius_gain_est, inradius_gain_cert: gauge inradius over polytope
        inradius.
    traj_box_joint, traj_box_sep, traj_gain: joint and separable
        equal-budget trajectory box and their ratio.
    ang_diag_gain, ang_inradius_cert, ang_inradius_gain: median angular over
        gauge radius on the diagonals; certified angular inradius and its
        ratio to the certified gauge inradius.
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

#: Directions for the sampled inradius estimate (not a certificate; the
#: certified inradius is reported beside it). Fixed seed.
N_SPHERE = 200
SPHERE_SEED = 0


def main() -> None:
    """Compare the joint gauge region against the separable cross-polytope."""
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
        # Trajectory class: vertex Gram gauge against the separable sum for
        # the equal-budget box m (1, ..., 1).

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

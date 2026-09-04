#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Amplitude damping and joint two-rate margins on the paper ensemble.

Extends the local-dephasing study to the second canonical dissipative
channel and to joint dissipative uncertainty:

* common-rate local amplitude damping, V_q = sigma_-^(q): certified
  margin M_amp on gamma_- against the bisected true crossing
  gamma_-^*;
* the joint two-rate certified cross-polytope
  L_z gamma_z + L_amp gamma_- <= F_pro(0) - F_T^2 (rates are
  nonnegative, so the certified region is a simplex in the positive
  quadrant), checked along the equal-rate diagonal against the true
  crossing g^* of gamma_z = gamma_- = g.

Writes results/lindblad-margin-python/open_amp_<FT>.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from qrobustness import lindblad as lb

from _drivers import load_ensemble

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/lindblad-margin-python"

FT = 0.999
MARGIN_TOL = 1e-6

SM = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)  # sigma_-
SZ = np.diag([1.0, -1.0]).astype(complex)
I2 = np.eye(2, dtype=complex)


def crossing(F, ft_pro, m_lo):
    lo, hi = m_lo, max(10 * m_lo, 1e-6)
    while F(hi) >= ft_pro and hi < 1e3:
        lo, hi = hi, 10 * hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if F(mid) >= ft_pro:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--FT", type=float, default=FT)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--controllers", type=int, default=0)
    args = ap.parse_args()
    ft_pro = args.FT**2

    problem, controllers = load_ensemble()
    if args.controllers:
        controllers = controllers[: args.controllers]
    nq = problem["n_qubits"]

    G_z = sum(lb.dissipator(V) for V in lb.local_ops(SZ, nq))
    G_amp = sum(lb.dissipator(V) for V in lb.local_ops(SM, nq))
    dn_z = lb.diamond_norm(G_z)
    dn_amp = lb.diamond_norm(G_amp)
    print(
        f"dnorm(dephasing) = {dn_z.value:.6f} ({dn_z.status}); "
        f"dnorm(amp damping) = {dn_amp.value:.6f} ({dn_amp.status})",
        flush=True,
    )

    rows = []
    for ci, c in enumerate(controllers):
        dt = c["tf"] / c["tau"]
        L_z = 0.5 * c["tf"] * dn_z.value
        L_amp = 0.5 * c["tf"] * dn_amp.value
        H_list = [
            problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
            for k in range(c["tau"])
        ]
        GH = [lb.hamiltonian_superop(H) for H in H_list]

        def F_pro(gz: float, ga: float) -> float:
            G_list = [G + gz * G_z + ga * G_amp for G in GH]
            return lb.process_fidelity(lb.channel(G_list, dt), problem["Uf"])

        F0 = F_pro(0.0, 0.0)
        if F0 <= ft_pro:
            continue
        surplus = F0 - ft_pro

        # Amplitude-damping margin and true crossing.
        om = lb.open_margin(
            lambda g: F_pro(0.0, g), L_amp, ft_pro, margin_tol=MARGIN_TOL
        )
        M_amp = float(om.M_plus)
        amp_star = crossing(lambda g: F_pro(0.0, g), ft_pro, M_amp)

        # Joint simplex: free certified equal-rate radius, its iterated
        # (directional) refinement, and the true crossing.
        g_cert = surplus / (L_z + L_amp)
        om_d = lb.open_margin(
            lambda g: F_pro(g, g), L_z + L_amp, ft_pro, margin_tol=MARGIN_TOL
        )
        M_diag = float(om_d.M_plus)
        g_star = crossing(lambda g: F_pro(g, g), ft_pro, M_diag)

        rows.append(
            {
                "controller": ci + 1,
                "fid": c["fid"],
                "err": c["error"],
                "F_pro_0": F0,
                "L_z": L_z,
                "L_amp": L_amp,
                "M_amp": M_amp,
                "r0_amp": surplus / L_amp,
                "amp_star": amp_star,
                "conservatism_amp": amp_star / M_amp,
                "g_cert_diag": g_cert,
                "M_diag": M_diag,
                "g_star_diag": g_star,
                "conservatism_diag": g_star / g_cert,
                "conservatism_diag_iter": g_star / M_diag,
            }
        )
        print(
            f"ctrl {ci + 1:2d}/{len(controllers)}: M_amp={M_amp:.3e} "
            f"amp*={amp_star:.3e} ({amp_star / M_amp:.3f}x)  "
            f"diag free={g_cert:.3e} iter={M_diag:.3e} "
            f"true={g_star:.3e} ({g_star / M_diag:.3f}x)",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"open_amp_{args.FT:g}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")

    for key in ("conservatism_amp", "conservatism_diag", "conservatism_diag_iter"):
        v = np.array([r[key] for r in rows])
        print(f"{key}: median {np.median(v):.3f}  max {v.max():.3f}")


if __name__ == "__main__":
    main()

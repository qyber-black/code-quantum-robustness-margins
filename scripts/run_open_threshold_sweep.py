#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Threshold sweep for the dissipative certificates.

Proposition (first-order rate response) predicts that the one-step
radius matches the resolved crossing for Pauli dephasing and sits at
half of it for amplitude damping, at first order.  This sweep tests
where the first-order picture ends: thresholds F_T in {0.99, 0.999,
0.9999} on the first ten controllers, both channels, reporting the
crossing-to-radius ratio (curvature and nominal error move it away
from the first-order value as the threshold, and hence the rate,
grows).

Writes results/lindblad-margin-python/open_threshold_sweep.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


from _drivers import load_ensemble, write_rows
from qrobustness import lindblad as lb

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/lindblad-margin-python"

SM = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
SZ = np.diag([1.0, -1.0]).astype(complex)
I2 = np.eye(2, dtype=complex)

THRESHOLDS = (0.99, 0.999, 0.9999)


def crossing(F, ft_pro, m_lo):
    lo, hi = m_lo, max(10 * m_lo, 1e-6)
    while F(hi) >= ft_pro and hi < 1e3:
        lo, hi = hi, 10 * hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if F(mid) >= ft_pro:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", type=int, default=10)
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help="write results here instead of the default tree",
    )
    args = ap.parse_args()

    problem, controllers = load_ensemble()
    controllers = controllers[: args.controllers]
    nq = problem["n_qubits"]

    chans = {
        "dephasing": sum(lb.dissipator(V) for V in lb.local_ops(SZ, nq)),
        "amp_damping": sum(lb.dissipator(V) for V in lb.local_ops(SM, nq)),
    }
    dns = {k: lb.diamond_norm(G) for k, G in chans.items()}
    for k, dn in dns.items():
        print(f"dnorm({k}) = {dn.value_certified:.6f}", flush=True)

    rows = []
    for ci, c in enumerate(controllers):
        dt = c["tf"] / c["tau"]
        H_list = [
            problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
            for k in range(c["tau"])
        ]
        GH = [lb.hamiltonian_superop(H) for H in H_list]
        for chan, G_chan in chans.items():
            L = 0.5 * c["tf"] * dns[chan].value_certified

            def F_pro(g: float) -> float:
                return lb.process_fidelity(
                    lb.channel([Gh + g * G_chan for Gh in GH], dt), problem["Uf"]
                )

            F0 = F_pro(0.0)
            for ft in THRESHOLDS:
                ft_pro = ft**2
                if F0 <= ft_pro:
                    continue
                r0 = (F0 - ft_pro) / L
                gs = crossing(F_pro, ft_pro, r0)
                rows.append(
                    {
                        "controller": ci + 1,
                        "channel": chan,
                        "FT": ft,
                        "r0": r0,
                        "gamma_star": gs,
                        "ratio": gs / r0,
                    }
                )
        print(f"ctrl {ci + 1}/{len(controllers)} done", flush=True)

    out = args.out
    path = write_rows(out / "open_threshold_sweep.csv", rows)

    for chan in chans:
        for ft in THRESHOLDS:
            v = np.array(
                [r["ratio"] for r in rows if r["channel"] == chan and r["FT"] == ft]
            )
            if v.size:
                print(
                    f"{chan} FT={ft}: crossing/r0 median {np.median(v):.4f} "
                    f"range [{v.min():.4f}, {v.max():.4f}]"
                )


if __name__ == "__main__":
    main()

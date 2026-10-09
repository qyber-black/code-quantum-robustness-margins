#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sweep the dissipative certificates over the fidelity threshold.

For the first N controllers of the main ensemble, both common-rate local
channels (dephasing, amplitude damping) and FT in {0.99, 0.999, 0.9999},
computes the one-step rate radius r_0 and the bisected true crossing gamma*
on F^pro with threshold FT^2, and their ratio (the xQRM paper, Numerical
evaluation, beyond dephasing). A controller enters a threshold's cohort only
if F^pro_0 - FT^2 > ELIGIBILITY_TOL, so the cohort shrinks as FT grows.
Options: --out, --controllers N (default 10); no --FT.

Writes results/lindblad-margin-python/:
open_threshold_sweep.csv
    controller, channel, FT, F0_pro: instance, threshold and F^pro_0.
    r0, gamma_star, ratio: one-step radius, true crossing, gamma_star / r0.
open_threshold_cohort.csv
    channel, FT, n_eligible, n_controllers: cohort size per threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _drivers import PAULI_Z, load_ensemble, true_crossing, write_rows
from qrobustness import lindblad as lb

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/lindblad-margin-python"

SM = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)  # sigma_-

THRESHOLDS = (0.99, 0.999, 0.9999)

#: Fixed number of bisection steps for the crossing (no early stop).
CROSSING_SWEEP_STEPS = 50

#: A controller joins a threshold's cohort only if F^pro_0 exceeds FT^2 by
#: more than this. The cohort changes with FT; its size is written per
#: threshold.
ELIGIBILITY_TOL = 1e-12


def main() -> None:
    """Sweep over the threshold and report the crossing-to-radius ratio."""
    # No --FT: the sweep runs over THRESHOLDS.
    ap = argparse.ArgumentParser(description=__doc__)
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
        "dephasing": sum(lb.dissipator(V) for V in lb.local_ops(PAULI_Z, nq)),
        "amp_damping": sum(lb.dissipator(V) for V in lb.local_ops(SM, nq)),
    }
    # Closed-form diamond norm 2n for both families; no SDP.
    dns = {k: lb.common_rate_local_dnorm(nq) for k in chans}
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
            L = lb.rate_lipschitz(dns[chan].value_certified, c["tf"])

            def F_pro(g: float, G_chan=G_chan, GH=GH, dt=dt) -> float:
                return lb.process_fidelity(
                    lb.channel([Gh + g * G_chan for Gh in GH], dt), problem["Uf"]
                )

            F0 = F_pro(0.0)
            for ft in THRESHOLDS:
                ft_pro = ft**2
                if F0 - ft_pro <= ELIGIBILITY_TOL:
                    # Not nominally safe at this threshold: not in the cohort.
                    continue
                r0 = (F0 - ft_pro) / L
                gs = true_crossing(F_pro, ft_pro, r0, steps=CROSSING_SWEEP_STEPS)
                rows.append(
                    {
                        "controller": ci + 1,
                        "channel": chan,
                        "FT": ft,
                        "F0_pro": F0,
                        "r0": r0,
                        "gamma_star": gs,
                        "ratio": gs / r0,
                    }
                )
        print(f"ctrl {ci + 1}/{len(controllers)} done", flush=True)

    out = args.out
    write_rows(out / "open_threshold_sweep.csv", rows)

    # Cohort size per channel and threshold.

    cohort = [
        {
            "channel": chan,
            "FT": ft,
            "n_eligible": sum(
                1 for r in rows if r["channel"] == chan and r["FT"] == ft
            ),
            "n_controllers": len(controllers),
        }
        for chan in chans
        for ft in THRESHOLDS
    ]
    write_rows(out / "open_threshold_cohort.csv", cohort)
    for r in cohort:
        print(
            f"cohort {r['channel']} FT={r['FT']}: "
            f"{r['n_eligible']}/{r['n_controllers']} eligible"
        )

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

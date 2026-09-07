#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gate-duration sweep: how every certificate scales with t_f.

Synthesises CNOT controllers (five per duration, deterministic seeds)
at t_f in {2, 4, 8, 16, 32} with proportional interval counts, and
reports the certified margins.  The honest expectation: the uniform
trajectory radii r_0 and r_FS and both universal-bound margins all
shrink roughly as 1/t_f at fixed structures (their integrated
constants grow with t_f), while the iterated constant margin M tracks
the actual landscape; any advantage between families lies in
structure-dependent constants, not in the absence of duration
dependence.

Writes results/cnot-python/duration_sweep_<FT>.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qrobustness import iterative_margin, lipschitz_constant, structure_constant

from _drivers import (
    DEFAULT_ETA,
    DEFAULT_MAX_ERROR,
    base_parser,
    cnot_model,
    write_rows,
)
from qrobustness.core import gate_fidelity, propagator
from qrobustness import kosut
from qrobustness.synthesis import grape
from qrobustness.timevarying import fs_margin, uniform_margin

ROOT = Path(__file__).resolve().parents[1]
#: Bracket refinement, matched to the other drivers.
MARGIN_TOL = 1e-8
#: Safe-radius continuation step, matched to the other drivers.
ETA = DEFAULT_ETA

OUT_DIR = ROOT / "results/cnot-python"

H0, X1, X2, CNOT = cnot_model()
#: Hilbert-space dimension of the two-qubit CNOT model.
DIM = 4
SEED0 = 20260820
N_PER_TF = 5
DURATIONS = (2.0, 4.0, 8.0, 16.0, 32.0)

#: Interval count per duration: proportional to t_f so the step size stays
#: comparable across the sweep, with a floor for the shortest gates.
INTERVALS_PER_UNIT_TIME = 5
MIN_INTERVALS = 10

#: Synthesis attempts allowed per duration before giving up on filling the
#: quota; GRAPE does not converge from every seed.
ATTEMPTS_PER_KEPT = 4


def main() -> None:
    """Synthesise controllers at each duration and record every certificate."""
    ap = base_parser(OUT_DIR, description=__doc__)
    args = ap.parse_args()
    ft = args.FT

    rows = []
    for tf in DURATIONS:
        tau = max(MIN_INTERVALS, int(INTERVALS_PER_UNIT_TIME * tf))
        dt = tf / tau
        kept = 0
        i = 0
        while kept < N_PER_TF and i < ATTEMPTS_PER_KEPT * N_PER_TF:
            r = grape(H0, [X1, X2], CNOT, tf, tau, seed=SEED0 + i)
            i += 1
            if r.error > DEFAULT_MAX_ERROR:
                continue
            u = r.u
            H_list = [H0 + u[0, k] * X1 + u[1, k] * X2 for k in range(tau)]
            F0 = gate_fidelity(propagator(H_list, dt), CNOT)
            if F0 <= ft:
                continue
            kept += 1
            dH = [u[0, k] * X1 for k in range(tau)]  # structure X1
            C = structure_constant("control", X1, dt, tau, u[0])
            L = lipschitz_constant(ft, DIM, C)

            def fid_fn(mu, dH=dH, H_list=H_list, dt=dt, tau=tau):
                return gate_fidelity(
                    propagator([H_list[k] + mu * dH[k] for k in range(tau)], dt), CNOT
                )

            M = float(
                iterative_margin(
                    fid_fn, L, ft, mu0=0.0, eta=ETA, margin_tol=MARGIN_TOL
                ).M
            )
            rates = kosut.uncertainty_rates(H_list, dH, dt)
            rows.append(
                {
                    "tf": tf,
                    "tau": tau,
                    "seed": r.seed,
                    "fid": F0,
                    "M": M,
                    "r0": uniform_margin(L, F0, ft),
                    "rfs": fs_margin(dH, dt, F0, ft).r_fs,
                    "KM": kosut.margin(rates, ft, nominal_error=1.0 - F0),
                    "KMtv": kosut.margin(
                        rates, ft, nominal_error=1.0 - F0, uncertainty="trajectory"
                    ),
                }
            )
            print(
                f"tf={tf:5.1f} ctrl {kept}/{N_PER_TF}: M={M:.3e} "
                f"rfs={rows[-1]['rfs']:.3e} KMtv={rows[-1]['KMtv']:.3e}",
                flush=True,
            )

        if kept == 0:
            # Said out loud rather than left as a gap in the table: at short
            # durations the synthesis may never reach the error threshold
            # within the attempt budget, and a silently missing duration
            # reads as an oversight.
            print(
                f"tf={tf:5.1f}: no controller reached max_error within "
                f"{ATTEMPTS_PER_KEPT * N_PER_TF} attempts; no rows",
                flush=True,
            )

    out = args.out
    write_rows(out / f"duration_sweep_{ft:g}.csv", rows)

    for tf in DURATIONS:
        sel = [r for r in rows if r["tf"] == tf]
        if not sel:
            continue
        med = {
            k: np.median([r[k] for r in sel]) for k in ("M", "r0", "rfs", "KM", "KMtv")
        }
        print(f"tf={tf:5.1f}: " + "  ".join(f"{k}={med[k]:.3e}" for k in med))


if __name__ == "__main__":
    main()

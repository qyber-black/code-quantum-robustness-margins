#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single-qubit pi-pulse: certified margins against analytic truth.

A resonant pi-pulse (X gate) over T = 1 with constant Rabi rate
Omega = pi, tau = 8 intervals, under two structures:

* amplitude error, Hhat = Omega sigma_x / 2 (multiplicative; commutes
  with the nominal evolution): the fidelity is F(delta) =
  cos(delta pi / 2) exactly, so the true crossing is
  delta* = (2/pi) arccos F_T, for constant AND time-varying
  perturbations alike (a sign-varying delta only helps).  The
  Fubini-Study certificate is exact here: the perturbed path is a
  geodesic, r_FS = delta*.
* detuning error, Hhat = sigma_z / 2 (additive; transverse): the
  generalised Rabi formula gives F(delta) =
  (Omega/Omega_g) |sin(Omega_g T / 2)|, Omega_g = sqrt(Omega^2 +
  delta^2), flat to first order at delta = 0, so certificates built
  from worst-case slopes are conservative and iteration recovers.

Prints and writes results/single-qubit-python/single_qubit_<FT>.csv
with, per structure: certified M (with bracket), analytic constant
crossing delta*, r_0, r_FS, adversarial upper bound m_adv on the
time-varying margin, and the Kosut margins (both classes).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from qrobustness import iterative_margin, lipschitz_constant, structure_constant
from qrobustness.core import gate_fidelity, propagator
from qrobustness import kosut
from qrobustness.timevarying import (
    adversarial_upper_bound,
    fs_margin,
    uniform_margin,
)

from _drivers import DEFAULT_ETA, PAULI_X, PAULI_Z, base_parser

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/single-qubit-python"

T = 1.0
TAU = 8
OMEGA = np.pi
DT = T / TAU
DIM = 2
ETA = DEFAULT_ETA
#: Bracket refinement for the certified margin, matched to the other drivers.
MARGIN_TOL = 1e-8

#: The script's claim is that the numerics equal the analytic fidelity, so
#: the agreement it demands is exactness to rounding, not a tolerance.
EXACT_TOL = 1e-12
#: Fractions of the analytic crossing at which that agreement is checked.
CHECK_FRACTIONS = (0.25, 0.5, 1.0)

#: Adversarial bracket: search up to a little past the certified radius
#: when the iterated margin does not already exceed it, from a fixed seed.
ADVERSARY_SPAN = 1.05
ADVERSARY_SEED = 1


def main() -> None:
    """Certify both structures and check them against analytic truth."""
    ap = base_parser(OUT_DIR, description=__doc__)
    args = ap.parse_args()
    ft = args.FT

    H = 0.5 * OMEGA * PAULI_X
    H_list = [H] * TAU
    Uf = expm(-1j * T * H)  # the exact pi-pulse gate (= -i sigma_x)
    F0 = gate_fidelity(propagator(H_list, DT), Uf)
    # Explicit, not assert: python -O strips assertions, and this script's
    # whole claim is that the numerics match analytic truth.
    if abs(F0 - 1.0) >= EXACT_TOL:
        raise SystemExit(f"ERROR: nominal fidelity {F0!r} is not 1 to {EXACT_TOL:g}")

    structures = {
        "amplitude": [0.5 * OMEGA * PAULI_X] * TAU,
        "detuning": [0.5 * PAULI_Z] * TAU,
    }

    def analytic_F(tag, delta):
        """Closed-form fidelity for either structure at offset ``delta``."""
        if tag == "amplitude":
            return abs(np.cos(0.5 * np.pi * delta))
        og = np.sqrt(OMEGA**2 + delta**2)
        return abs(np.sin(0.5 * og * T)) * OMEGA / og

    rows = []
    for tag, dH in structures.items():
        # Certified constant margin via the generic API: the structure is
        # constant in time, which is the 'control with unit amplitude'
        # special case of the structure constant.
        C = structure_constant("control", dH[0], DT, TAU, np.ones(TAU))
        L = lipschitz_constant(ft, DIM, C)

        def fid_fn(mu, dH=dH):
            return gate_fidelity(
                propagator([H_list[k] + mu * dH[k] for k in range(TAU)], DT), Uf
            )

        res = iterative_margin(fid_fn, L, ft, mu0=0.0, eta=ETA, margin_tol=MARGIN_TOL)
        M = float(res.M)

        # Analytic constant crossing.
        d_star = brentq(lambda d, t=tag: analytic_F(t, d) - ft, 0.0, 1.0)

        # Uniform time-varying certificates and adversarial bracket.
        r0 = uniform_margin(L, F0, ft)
        fs = fs_margin(dH, DT, F0, ft, r0=r0)
        br = adversarial_upper_bound(
            H_list,
            dH,
            DT,
            Uf,
            ft,
            fs.r,
            max(M, ADVERSARY_SPAN * fs.r),
            seed=ADVERSARY_SEED,
        )

        # Kosut margins, both classes (eps_0 = 0 here).
        rates = kosut.uncertainty_rates(H_list, dH, DT)
        KM = kosut.margin(rates, ft)
        KMtv = kosut.margin(rates, ft, uncertainty="trajectory")

        # Consistency of the analytic formula with the numerics. This is
        # the point of the script, so it must survive python -O.
        for d in (f * d_star for f in CHECK_FRACTIONS):
            gap = abs(fid_fn(d) - analytic_F(tag, d))
            if gap >= EXACT_TOL:
                raise SystemExit(
                    f"ERROR: {tag} numerics disagree with the analytic "
                    f"fidelity at delta={d!r} by {gap:.3e}"
                )

        rows.append(
            {
                "structure": tag,
                "M": M,
                "M_upper": float(res.M_upper),
                "delta_star": float(d_star),
                "r0": r0,
                "r_fs": fs.r_fs,
                "m_adv": br.m_adv,
                "KM": KM,
                "KM_tv": KMtv,
            }
        )
        print(
            f"{tag:>10}: M={M:.6f} (<= delta*={d_star:.6f})  "
            f"r0={r0:.6f}  r_fs={fs.r_fs:.6f}  m_adv={br.m_adv:.6f}  "
            f"KM={KM:.6f}  KM_tv={KMtv:.6f}",
            flush=True,
        )

    amp = rows[0]
    print(
        f"\nGeodesic tightness (amplitude): r_fs = {amp['r_fs']:.10f}, "
        f"delta* = {amp['delta_star']:.10f}, "
        f"rel diff = {abs(amp['r_fs'] - amp['delta_star']) / amp['delta_star']:.2e}"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"single_qubit_{ft:g}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

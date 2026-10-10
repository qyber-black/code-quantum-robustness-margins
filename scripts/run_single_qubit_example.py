#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single-qubit pi-pulse: certified margins checked against the analytic result.

A resonant pi-pulse (X gate), T = 1, Omega = pi, tau = 8, under two
structures: amplitude error Omega sigma_x / 2 (analytic F = cos(delta pi/2))
and detuning sigma_z / 2 (generalised Rabi formula). For each, computes M
with M_upper, the analytic constant crossing delta*, r_0, r_FS, the
adversarial upper witness m_adv on the time-varying margin, and M^K and
M^{K,tri}_tv (the xQRM paper, Numerical evaluation, a single qubit against
analytic truth). Exits with an error if the nominal fidelity or the
numerical fidelity at fractions of delta* differs from the analytic value by
EXACT_TOL or more. Options: --FT, --out.

Writes results/single-qubit-python/single_qubit_<FT>.csv, one row per
structure:
    structure: amplitude or detuning.
    M, M_upper, delta_star: iterated bracket and analytic crossing.
    r0, r_fs, m_adv: r_0, r_FS and m_adv.
    adv_violated: 1 if the search found a violation (m_adv is a witness),
        0 if m_adv is only the search ceiling.
    KM, KM_tv: M^K and M^{K,tri}_tv.
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
#: Relative bracket tolerance, as in the other drivers.
MARGIN_TOL = 1e-8

#: Required agreement of numerical and analytic fidelity (rounding level).
EXACT_TOL = 1e-12
#: Fractions of the analytic crossing at which that agreement is tested.
CHECK_FRACTIONS = (0.25, 0.5, 1.0)

#: Adversarial search up to ADVERSARY_SPAN max(M_upper, r_FS), fixed seed.
ADVERSARY_SPAN = 1.05
ADVERSARY_SEED = 1


def main() -> None:
    """Certify both structures and test them against analytic truth."""
    ap = base_parser(OUT_DIR, description=__doc__)
    args = ap.parse_args()
    ft = args.FT

    H = 0.5 * OMEGA * PAULI_X
    H_list = [H] * TAU
    Uf = expm(-1j * T * H)  # the exact pi-pulse gate (= -i sigma_x)
    F0 = gate_fidelity(propagator(H_list, DT), Uf)
    # Explicit check, not assert, so it survives python -O.
    if abs(F0 - 1.0) >= EXACT_TOL:
        raise SystemExit(f"ERROR: nominal fidelity {F0!r} is not 1 to {EXACT_TOL:g}")

    structures = {
        "amplitude": [0.5 * OMEGA * PAULI_X] * TAU,
        "detuning": [0.5 * PAULI_Z] * TAU,
    }

    def analytic_F(tag, delta):
        """Closed-form fidelity for either structure at the offset ``delta``."""
        if tag == "amplitude":
            return abs(np.cos(0.5 * np.pi * delta))
        og = np.sqrt(OMEGA**2 + delta**2)
        return abs(np.sin(0.5 * og * T)) * OMEGA / og

    rows = []
    for tag, dH in structures.items():
        # Time-constant structure: 'control' with unit amplitude.
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

        # Uniform time-varying certificates and the adversarial witness.
        r0 = uniform_margin(L, F0, ft)
        fs = fs_margin(dH, DT, F0, ft, r0=r0)
        br = adversarial_upper_bound(
            H_list,
            dH,
            DT,
            Uf,
            ft,
            fs.r,
            # Above the constant crossing a constant perturbation violates, so
            # the search has a witness to find.
            ADVERSARY_SPAN * max(float(res.M_upper), fs.r),
            seed=ADVERSARY_SEED,
            n_starts=4,
            starts="legacy",
            maxiter=200,
        )

        # Kosut margins, both classes (eps_0 = 0 here).
        rates = kosut.uncertainty_rates(H_list, dH, DT)
        KM = kosut.margin(rates, ft)
        KMtv = kosut.margin(rates, ft, uncertainty="trajectory")

        # Numerics against the analytic fidelity (explicit, survives python -O).

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
                "adv_violated": int(br.delta_adv is not None),
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

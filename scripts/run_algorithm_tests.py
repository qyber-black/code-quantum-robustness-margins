#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Checks of the continuation algorithm against known answers.

Computes, for the xQRM paper (Numerical evaluation, targeted checks of the
continuation), joint-gauge and angular radii for nearly dependent structures
and directional margins M, M_upper on one-parameter rays whose first
crossing is known. Options: --FT, --out. Writes
results/algorithm-tests-python/:

crosstalk_<FT>.csv
    One row per controller and kappa, for the structures u1 H1 and
    u1 (H1 + kappa H2) on the main ensemble, along the diagonal (1, -1)/sqrt 2.
    controller, kappa, gram_corr_max: instance; largest interval Gram
        correlation.
    r_poly_diag, r_gauge_diag, r_ang_diag: cross-polytope, C_joint gauge and
        angular (C^stat_FS) radii along the diagonal.
    gain_gauge, gain_ang: the last two divided by r_poly_diag.
    minF_gauge_boundary, minF_ang_boundary: least fidelity on the gauge and
        angular region boundaries over N_SPHERE directions.
    M_diag, Mupper_diag, nev_diag: directional margin M, M_upper and
        evaluation count along the diagonal.
rays_<FT>.csv
    One row per case: the single-qubit pi-pulse detuning ray at FT = 0.25
    (non-monotone, Lipschitz and angular steps), a curve that touches FT
    before crossing it, and the detuning ray with a bounded evaluation error,
    with and without the matching evaluation band.
    case, FT: case name and threshold.
    M, M_upper, reason, n_unresolved, n_evals: continuation result.
    first_unsafe, prefix_ok, witness_ok: true first crossing; whether M lies
        below it and M_upper above it.
    island_lo, island_hi / touch_point / noise: case-specific extras.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from qrobustness import iterative_margin, lipschitz_constant, structure_constant
from qrobustness import multiparam as mp
from qrobustness.core import gate_fidelity, propagator

from _drivers import (
    DEFAULT_ETA,
    PAULI_X,
    PAULI_Z,
    ROOT,
    base_parser,
    load_ensemble,
    write_rows,
)

OUT_DIR = ROOT / "results/algorithm-tests-python"
MARGIN_TOL = 1e-8
KAPPAS = (1.0, 0.3, 0.1, 0.03, 0.01)
N_SPHERE = 64
SPHERE_SEED = 0
DIAG = np.array([1.0, -1.0]) / np.sqrt(2.0)

# Single-qubit pi-pulse of run_single_qubit_example.py.
T, TAU, OMEGA = 1.0, 8, np.pi
DT = T / TAU
SQ_FT_LOW = 0.25  # below the first revival maximum (about 1/3): an island
NOISE_AMP = 1e-9  # bounded evaluation error added to the detuning fidelity
NOISE_FREQ = 1.0e5
TOUCH_A = 0.01  # amplitude of the threshold-touching curve
TOUCH_FT = 0.9
#: |g'| <= TOUCH_L_OVER_A * A on this interval, which is the Lipschitz
#: constant passed to the margin on the touching curve.
TOUCH_OMEGA = (0.0, 3.0)
TOUCH_L_OVER_A = 4


def crosstalk(fT):
    problem, controllers = load_ensemble()
    dim = problem["dim"]
    H1, H2 = problem["H1"], problem["H2"]
    rows = []
    sphere = mp.sphere_directions(2, N_SPHERE, seed=SPHERE_SEED)
    for idx, c in enumerate(controllers, start=1):
        dt = c["tf"] / c["tau"]
        u1 = c["u1"]
        Hs = [problem["H0"] + a * H1 + b * H2 for a, b in zip(u1, c["u2"], strict=True)]
        F0 = gate_fidelity(propagator(Hs, dt), problem["Uf"])
        for kappa in KAPPAS:
            struct = [
                [a * H1 for a in u1],
                [a * (H1 + kappa * H2) for a in u1],
            ]

            def fid(x, struct=struct, Hs=Hs, dt=dt):
                return gate_fidelity(
                    propagator(
                        [
                            h + x[0] * s0 + x[1] * s1
                            for h, s0, s1 in zip(Hs, *struct, strict=True)
                        ],
                        dt,
                    ),
                    problem["Uf"],
                )

            C = np.array(
                [
                    structure_constant("control", H1, dt, len(u1), u1),
                    structure_constant("control", H1 + kappa * H2, dt, len(u1), u1),
                ]
            )
            L = np.array([lipschitz_constant(fT, dim, x) for x in C])
            G = mp.joint_gauge(struct, dt)
            A = mp.angular_gauge(struct, dt)
            surplus = F0 - fT
            r_poly = surplus / float(L @ np.abs(DIAG))
            r_gauge = G.boundary_radius(DIAG, surplus, fT, dim)
            r_ang = A.boundary_radius(DIAG, F0, fT)
            min_gauge = min(
                fid(G.boundary_radius(d, surplus, fT, dim) * d) for d in sphere
            )
            min_ang = min(fid(A.boundary_radius(d, F0, fT) * d) for d in sphere)
            res = mp.directional_margin(
                fid,
                L,
                fT,
                DIAG,
                L_dir=G.L_dir(DIAG, fT, dim),
                angular_gauge=A,
                eta=DEFAULT_ETA,
                margin_tol=MARGIN_TOL,
                return_diagnostics=True,
            )
            corr = max(
                abs(P[0, 1]) / np.sqrt(P[0, 0] * P[1, 1])
                for P in G.grams
                if P[0, 0] > 0 and P[1, 1] > 0
            )
            rows.append(
                {
                    "controller": idx,
                    "kappa": kappa,
                    "gram_corr_max": corr,
                    "r_poly_diag": r_poly,
                    "r_gauge_diag": r_gauge,
                    "r_ang_diag": r_ang,
                    "gain_gauge": r_gauge / r_poly,
                    "gain_ang": r_ang / r_poly,
                    "minF_gauge_boundary": min_gauge,
                    "minF_ang_boundary": min_ang,
                    "M_diag": res.M_plus,
                    "Mupper_diag": res.M_upper_plus,
                    "nev_diag": res.n_evals,
                }
            )
    return rows


def _detuning_problem():
    H_list = [OMEGA * PAULI_X / 2] * TAU
    dH = [PAULI_Z / 2] * TAU

    def F(delta):
        return gate_fidelity(
            propagator([h + delta * d for h, d in zip(H_list, dH, strict=True)], DT),
            PAULI_X,
        )

    def F_exact(delta):
        g = np.hypot(OMEGA, delta)
        return OMEGA / g * abs(np.sin(g * T / 2))

    C = structure_constant("drift", PAULI_Z / 2, DT, TAU)
    return F, F_exact, C


def _row(name, fT, res, first, extra=None):
    row = {
        "case": name,
        "FT": fT,
        "M": res.M_plus,
        "M_upper": res.M_upper_plus,
        "reason": res.reason_plus,
        "n_unresolved": res.n_unresolved,
        "n_evals": res.n_evals,
        "first_unsafe": first,
        "prefix_ok": int(res.M_plus <= first * (1 + 1e-12)),
        "witness_ok": int(
            not np.isfinite(res.M_upper_plus) or res.M_upper_plus >= first * (1 - 1e-9)
        ),
    }
    row.update(extra or {})
    return row


def rays(fT):
    rows = []
    F, F_exact, C = _detuning_problem()
    # (a) non-monotone: the first crossing below the first zero of the
    # Rabi curve (Omega_g = 2 pi), then a revival island above SQ_FT_LOW.
    z1 = np.sqrt((2 * np.pi) ** 2 - OMEGA**2)
    first = brentq(lambda d: F_exact(d) - SQ_FT_LOW, 1e-6, z1)
    z2 = np.sqrt((4 * np.pi) ** 2 - OMEGA**2)
    peak = minimize_scalar(
        lambda d: -F_exact(d),
        bounds=(z1, z2),
        method="bounded",
        options={"xatol": 1e-12},
    ).x
    island = (
        brentq(lambda d: F_exact(d) - SQ_FT_LOW, z1, peak),
        brentq(
            lambda d: F_exact(d) - SQ_FT_LOW,
            peak,
            z2,
        ),
    )
    L = lipschitz_constant(SQ_FT_LOW, 2, C)
    for rule in ("lipschitz", "angular"):
        radius = None
        if rule == "angular":
            s = C / np.sqrt(2)
            th = np.arccos(SQ_FT_LOW)

            def radius(Fv, s=s, th=th):
                return max(0.0, (th - np.arccos(min(Fv, 1.0))) / s)

        res = iterative_margin(
            F,
            L,
            SQ_FT_LOW,
            eta=DEFAULT_ETA,
            margin_tol=MARGIN_TOL,
            safe_radius_fn=radius,
            return_diagnostics=True,
        )
        rows.append(
            _row(
                f"revival_{rule}",
                SQ_FT_LOW,
                res,
                first,
                {"island_lo": island[0], "island_hi": island[1]},
            )
        )

    # (b) a curve that touches the threshold at mu = 1 and crosses it at
    # mu = 2: g = FT + A (1 - mu)^2 (2 - mu) / 2, |g'| <= TOUCH_L_OVER_A * A
    # on TOUCH_OMEGA.
    def touch(mu):
        return TOUCH_FT + TOUCH_A * (1 - mu) ** 2 * (2 - mu) / 2

    res = iterative_margin(
        touch,
        TOUCH_L_OVER_A * TOUCH_A,
        TOUCH_FT,
        eta=DEFAULT_ETA,
        omega=TOUCH_OMEGA,
        margin_tol=MARGIN_TOL,
        return_diagnostics=True,
    )
    rows.append(_row("touch", TOUCH_FT, res, 2.0, {"touch_point": 1.0}))

    # (c) bounded evaluation error on the detuning ray at the paper
    # threshold, with and without the matching band.
    first_c = brentq(lambda d: F_exact(d) - fT, 1e-6, z1)

    def noisy(d):
        return F(d) + NOISE_AMP * np.sin(NOISE_FREQ * d)

    Lc = lipschitz_constant(fT, 2, C)
    for band in (0.0, NOISE_AMP):
        res = iterative_margin(
            noisy,
            Lc,
            fT,
            eta=DEFAULT_ETA,
            margin_tol=MARGIN_TOL,
            eval_tol=band,
            return_diagnostics=True,
        )
        rows.append(
            _row(
                f"noisy_band{band:g}",
                fT,
                res,
                first_c,
                {"noise": NOISE_AMP},
            )
        )
    return rows


def main() -> None:
    ap = base_parser(OUT_DIR, __doc__.splitlines()[0])
    args = ap.parse_args()
    fT, out = args.FT, args.out
    ray_rows = rays(fT)
    fields = list(dict.fromkeys(k for r in ray_rows for k in r))
    write_rows(out / f"rays_{fT}.csv", ray_rows, fields)
    write_rows(out / f"crosstalk_{fT}.csv", crosstalk(fT))


if __name__ == "__main__":
    main()

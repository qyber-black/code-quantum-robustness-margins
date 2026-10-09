#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The mixed coherent-dissipative certified region in the (mu_1, gamma_z) plane.

For one controller of the main ensemble, with the multiplicative H1 control
error mu_1 and the common local dephasing rate gamma_z >= 0, computes
F^pro on a grid (the numerically resolved contour F^pro = FT^2), the
diamond-norm constants L_mu^op and L_gamma of the free mixed simplex
L_mu^op |mu_1| + L_gamma gamma_z <= F^pro_0 - FT^2, and the iterated margin
along five mixed rays (the xQRM paper, Numerical evaluation, mixed
coherent-dissipative uncertainty). Options: --FT, --out, --controller
(1-based), --n (grid points per axis).

Writes results/lindblad-margin-python/mixed_ctrl<i>.npz:
    mu, gamma, F: grid axes and F^pro on the grid.
    F0, FT, L_mu, L_gamma, surplus: nominal F^pro, threshold, constants and
        F^pro_0 - FT^2.
    ray_angle, ray_dmu, ray_dgamma: ray angle (degrees) and unit direction.
    ray_free, ray_M: one-step (free simplex) radius and iterated margin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qrobustness import lindblad as lb

from _drivers import PAULI_Z, base_parser, load_ensemble

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/lindblad-margin-python"
#: Relative bracket tolerance for the open-system margin, coarser than the
#: closed-system 1e-8 because each probe is a Liouville-space propagation.
MARGIN_TOL = 1e-6

#: Grid extent for the contour: the mu axis is scaled from L_mu^op and widened
#: by MU_SPAN_OVERSHOOT, since that constant is loose for coherent
#: directions; the gamma axis is scaled from L_gamma.
MU_SPAN_FACTOR = 3.0
MU_SPAN_OVERSHOOT = 50
GAMMA_SPAN_FACTOR = 2.5

#: Ray angles from the positive mu axis to the gamma axis.
RAY_ANGLES_DEG = (0.0, 22.5, 45.0, 67.5, 90.0)


def main() -> None:
    """Map the certified mixed region of one controller."""
    ap = base_parser(OUT_DIR, description=__doc__)
    ap.add_argument("--controller", type=int, default=1)
    ap.add_argument("--n", type=int, default=41)
    args = ap.parse_args()
    ft_pro = args.FT**2

    problem, controllers = load_ensemble()
    c = controllers[args.controller - 1]
    dt = c["tf"] / c["tau"]
    tau = c["tau"]
    nq = problem["n_qubits"]

    G_z = sum(lb.dissipator(V) for V in lb.local_ops(PAULI_Z, nq))
    L_gamma = lb.rate_lipschitz(lb.diamond_norm(G_z).value_certified, c["tf"])

    H_list = [
        problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
        for k in range(tau)
    ]
    GH = [lb.hamiltonian_superop(H) for H in H_list]
    # Coherent structure evaluated through the open functional: multiplicative H1.
    G_mu = [lb.hamiltonian_superop(c["u1"][k] * problem["H1"]) for k in range(tau)]
    # One verified SDP suffices: G_mu^(k) = u_1k (-i[Hhat_1, .]) is
    # homogeneous in u_1k, so dnorm scales with |u_1k|.
    dn_H1 = lb.diamond_norm(lb.hamiltonian_superop(problem["H1"])).value_certified
    L_mu = 0.5 * dt * float(np.sum(np.abs(c["u1"]))) * dn_H1
    print(f"L_gamma = {L_gamma:.4f}  L_mu^op = {L_mu:.4f}", flush=True)

    def F_pro(mu: float, g: float) -> float:
        G_list = [GH[k] + mu * G_mu[k] + g * G_z for k in range(tau)]
        return lb.process_fidelity(lb.channel(G_list, dt), problem["Uf"])

    F0 = F_pro(0.0, 0.0)
    surplus = F0 - ft_pro
    print(f"F_pro(0) = {F0:.8f}  surplus = {surplus:.2e}", flush=True)

    # Contour grid, resolved numerically.
    mu_max = MU_SPAN_FACTOR * surplus / L_mu * MU_SPAN_OVERSHOOT
    g_max = GAMMA_SPAN_FACTOR * surplus / L_gamma
    mus = np.linspace(-mu_max, mu_max, args.n)
    gs = np.linspace(0.0, g_max, args.n)
    F = np.empty((args.n, args.n))
    for i, m in enumerate(mus):
        for j, g in enumerate(gs):
            F[i, j] = F_pro(float(m), float(g))
        print(f"row {i + 1}/{args.n}", flush=True)

    # Iterated margins along mixed rays (mu, gamma), gamma >= 0.

    angles = np.array(RAY_ANGLES_DEG) * np.pi / 180.0
    rays = []
    for a in angles:
        d = np.array([np.cos(a), np.sin(a)])
        L_d = L_mu * abs(d[0]) + L_gamma * d[1]

        def F_ray(s: float, d=d) -> float:
            return F_pro(s * d[0], s * d[1])

        om = lb.open_margin(F_ray, L_d, ft_pro, margin_tol=MARGIN_TOL)
        rays.append(
            {
                "angle_deg": float(np.degrees(a)),
                "d_mu": float(d[0]),
                "d_gamma": float(d[1]),
                "free": surplus / L_d,
                "M": float(om.M_plus),
            }
        )
        print(
            f"ray {np.degrees(a):5.1f} deg: free={surplus / L_d:.3e} "
            f"iterated={om.M_plus:.3e}",
            flush=True,
        )

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"mixed_ctrl{args.controller}.npz"
    np.savez(
        path,
        mu=mus,
        gamma=gs,
        F=F,
        F0=F0,
        FT=args.FT,
        L_mu=L_mu,
        L_gamma=L_gamma,
        surplus=surplus,
        ray_angle=[r["angle_deg"] for r in rays],
        ray_free=[r["free"] for r in rays],
        ray_M=[r["M"] for r in rays],
        ray_dmu=[r["d_mu"] for r in rays],
        ray_dgamma=[r["d_gamma"] for r in rays],
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

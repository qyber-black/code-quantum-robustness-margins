#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Open-system margins for the paper-2 case study.

Three-qubit chain controllers under local dephasing D[sz_q] on each qubit
q with a common uncertain rate gamma.  Per controller:

  - certified margin on gamma (open_margin with the diamond-norm constant),
    with margin_tol brackets;
  - the true threshold crossing gamma* from a bisection on the exact
    process fidelity (the conservatism reference);
  - conservatism factor gamma* / M_gamma.

The process-fidelity threshold is FT_pro = FT^2, matching the closed-system
threshold on |Tr(Uf' U)|/N.

Writes results/lindblad-margin-python/open_margins_<FT>.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from qrobustness import (
    dH_structure,
    structure_constant,
)
from qrobustness import lindblad as lb

from _drivers import base_parser, load_ensemble, true_crossing, write_rows

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/lindblad-margin-python"
#: Bracket refinement for the open-system margins: each probe is a
#: Liouville-space propagation, so this is looser than the closed-system
#: 1e-8 the coherent drivers use.
MARGIN_TOL = 1e-6

#: Stopping rules for the two conservatism references below.
GAMMA_STAR_STEPS = 60
GAMMA_STAR_RTOL = 1e-10
MU_STAR_STEPS = 80
MU_STAR_RTOL = 1e-12


def main() -> None:
    """Certified dephasing-rate margins and their true crossings."""
    ap = base_parser(OUT_DIR, description=__doc__)
    ap.add_argument(
        "--controllers",
        type=int,
        default=0,
        help="limit to the first n controllers (0 = all)",
    )
    ap.add_argument(
        "--coherent-controllers",
        type=int,
        default=3,
        help="controllers for the coherent-through-open-functional "
        "comparison, which needs a diamond norm per interval "
        "and a bisection per controller (0 = skip)",
    )
    args = ap.parse_args()
    ft_pro = args.FT**2

    problem, controllers = load_ensemble()
    if args.controllers:
        controllers = controllers[: args.controllers]

    Vs = lb.local_dephasing_ops(problem["n_qubits"])
    # The dissipative structure generator for the common rate gamma is the
    # sum of the per-qubit dissipators; it is interval-independent.
    G_gamma = sum(lb.dissipator(V) for V in Vs)

    rows = []
    # The common-rate local dephasing family has the exactly proved
    # diamond norm 2n (paper Lemma "common-rate diamond norms"), so no
    # SDP is solved for it here. run_dnorm_certificates.py keeps the
    # solver cross-check of that closed form.
    dn = lb.common_rate_local_dnorm(problem["n_qubits"])
    print(f"dnorm(sum_q D[sz_q]) = {dn.value:.6f}  (status {dn.status})", flush=True)

    for ci, c in enumerate(controllers):
        dt = c["tf"] / c["tau"]
        tau = c["tau"]
        t_f = c["tf"]
        H_list = [
            problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
            for k in range(tau)
        ]
        GH_list = [lb.hamiltonian_superop(H) for H in H_list]
        L_gamma = lb.rate_lipschitz(dn.value, t_f)

        def F_pro(gamma: float, GH_list=GH_list, dt=dt) -> float:
            G_list = [GH + float(gamma) * G_gamma for GH in GH_list]
            return lb.process_fidelity(lb.channel(G_list, dt), problem["Uf"])

        F0 = F_pro(0.0)
        if F0 <= ft_pro:
            print(f"controller {ci + 1}: nominal below threshold, skipped", flush=True)
            continue

        res = lb.open_margin(
            F_pro, L_gamma, ft_pro, margin_tol=MARGIN_TOL, return_diagnostics=True
        )
        M_gamma = res.M_plus  # gamma >= 0 side
        r0_gamma = (F0 - ft_pro) / L_gamma  # one-step certified radius

        # Conservatism reference: true crossing by bisection on gamma.
        gamma_star = true_crossing(
            F_pro,
            ft_pro,
            M_gamma,
            steps=GAMMA_STAR_STEPS,
            rel_tol=GAMMA_STAR_RTOL,
        )

        rows.append(
            {
                "controller": ci + 1,
                "fid": c["fid"],
                "err": c["error"],
                "F_pro_0": F0,
                "L_gamma": L_gamma,
                "M_gamma": M_gamma,
                "M_gamma_upper": res.M_upper_plus,
                "r0_gamma": r0_gamma,
                "gamma_star": gamma_star,
                "conservatism_M": gamma_star / M_gamma if M_gamma > 0 else np.inf,
                "conservatism_r0": gamma_star / r0_gamma,
                "n_evals": res.n_evals,
            }
        )
        print(
            f"controller {ci + 1}/{len(controllers)}  "
            f"M_gamma={M_gamma:.3e}  gamma*={gamma_star:.3e}  "
            f"r0={r0_gamma:.3e}  cons_r0={rows[-1]['conservatism_r0']:.1f}x  "
            f"evals={res.n_evals}",
            flush=True,
        )

    out = args.out
    write_rows(out / f"open_margins_{args.FT:g}.csv", rows)

    if args.coherent_controllers:
        write_coherent(args, problem, controllers, ft_pro, out)


def write_coherent(args, problem, controllers, ft_pro, out) -> None:
    """Coherent perturbation measured through the open-system functional.

    The open-system constant covers dissipation; applied to a coherent
    structure it is far looser than the closed-system one, and the paper
    quotes the size of that gap. Generating it here keeps those figures out
    of the prose.
    """
    N = problem["H0"].shape[0]
    B_T = float(np.sqrt((1.0 - args.FT**2) / N))
    # For a Hamiltonian superoperator the diamond norm is the spectral
    # spread of the generator; reported once for the unit structure.
    dn_unit = lb.diamond_norm(lb.hamiltonian_superop(problem["H1"])).value
    rows = []
    for ci, c in enumerate(controllers[: args.coherent_controllers]):
        tau = c["tau"]
        dt = c["tf"] / tau
        dH = dH_structure(
            problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], "H1"
        )
        GH = [
            lb.hamiltonian_superop(
                problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
            )
            for k in range(tau)
        ]
        Gd = [lb.hamiltonian_superop(d) for d in dH]
        # A multiplicative control error scales one fixed structure per
        # interval, Hhat^(k) = u1(k) H1, and the diamond norm is absolutely
        # homogeneous, so sum_k dnorm(-i[Hhat^(k),.]) = dnorm(-i[H1,.])
        # sum_k |u1(k)|. That is one SDP instead of tau of them, and exact
        # rather than an approximation.
        #
        # The SDP stays because the certified value is what a robustness
        # constant may use. Measured against the solver optimum, that
        # optimum matches the spectral spread of the generator to six
        # decimals for every Hermitian tried, so dnorm(-i[A,.]) = spread(A)
        # looks like an identity and would give a tighter constant here --
        # the certified bound is up to 9% above it at N=4 through the
        # Gershgorin step. Taking that shortcut needs the identity proved,
        # not observed, so it is left alone.
        L_op = 0.5 * dt * dn_unit * float(np.abs(np.asarray(c["u1"])).sum())
        L_closed = B_T * structure_constant("control", problem["H1"], dt, tau, c["u1"])

        def F_pro(mu: float, GH=GH, Gd=Gd, dt=dt) -> float:
            G = [g + float(mu) * d for g, d in zip(GH, Gd, strict=True)]
            return lb.process_fidelity(lb.channel(G, dt), problem["Uf"])

        F0 = F_pro(0.0)
        r0 = (F0 - ft_pro) / L_op
        res = lb.open_margin(
            F_pro, L_op, ft_pro, margin_tol=MARGIN_TOL, return_diagnostics=True
        )
        M = res.M_plus
        mu_star = true_crossing(
            F_pro,
            ft_pro,
            r0,
            lo=0.0,
            steps=MU_STAR_STEPS,
            rel_tol=MU_STAR_RTOL,
        )
        rows.append(
            {
                "controller": ci + 1,
                "dnorm_unit": dn_unit,
                "L_op": L_op,
                "L_closed": L_closed,
                "ratio_L": L_op / L_closed,
                "F_pro_0": F0,
                "r0": r0,
                "M": M,
                "mu_star": mu_star,
                "ratio_r0": mu_star / r0,
                "ratio_M": mu_star / M if M > 0 else float("inf"),
                "n_evals": res.n_evals,
            }
        )
        print(
            f"coherent {ci + 1}: L_op/L_closed={rows[-1]['ratio_L']:.2f}x  "
            f"mu*/r0={rows[-1]['ratio_r0']:.1f}x  "
            f"mu*/M={rows[-1]['ratio_M']:.3f}",
            flush=True,
        )

    path = out / f"open_coherent_{args.FT:g}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

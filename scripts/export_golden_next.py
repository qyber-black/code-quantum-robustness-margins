#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export Python-internal regression fixtures for the multiparameter,
time-varying and open-system layers.

Peer of export_golden.py for the layers that have no MATLAB counterpart
yet; the fixture pins the Python results so refactoring cannot silently
change published numbers. Verified by tests/test_regression_next.py.
The open-system record requires cvxpy and is skipped (with a notice)
when it is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qrobustness import multiparam as mp
from qrobustness import timevarying as tv

from _drivers import load_ensemble, three_structure_specs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/reference/paper_next_subset.json"

FT = 0.999
MARGIN_TOL = 1e-8
# Same anchor controllers as export_golden.py (0-based, filtered list).
INDICES = [0, 58, 60]
STRUCTURES = ("H0", "H1", "H2")
# One axis and one diagonal direction keep the fixture fast to recompute.
DIRECTIONS = {
    "+e1": np.array([0.0, 1.0, 0.0]),
    "diagppp": np.ones(3) / np.sqrt(3.0),
}


def _record(problem, controllers, idx: int) -> dict:
    c = controllers[idx]
    dt = c["tf"] / c["tau"]
    specs = three_structure_specs(problem, c)
    C, L = mp.structure_constants(specs, dt, c["tau"], FT, problem["dim"])
    fn = mp.make_multiparam_fidelity_fn(
        problem["H0"],
        problem["H1"],
        problem["H2"],
        c["u1"],
        c["u2"],
        problem["Uf"],
        dt,
        STRUCTURES,
    )
    F0 = fn(np.zeros(3))
    P = mp.safe_polytope(np.zeros(3), L, F0, FT)
    rec = {
        "controller_index": idx,
        "fid": F0,
        "L": list(L),
        "inradius_linf": P.inradius_linf,
        "inradius_l2": P.inradius_l2,
        "r0_joint": tv.uniform_margin(L, F0, FT),
    }
    for name, d in DIRECTIONS.items():
        res = mp.directional_margin(fn, L, FT, d, margin_tol=MARGIN_TOL)
        rec[f"M_{name}"] = res.M
        rec[f"Mub_{name}"] = res.M_upper
    return rec


def _open_record(problem, controllers, idx: int) -> dict | None:
    try:
        from qrobustness import lindblad as lb
    except ImportError:
        return None
    try:
        import cvxpy  # noqa: F401
    except ImportError:
        return None
    c = controllers[idx]
    dt = c["tf"] / c["tau"]
    tau = c["tau"]
    ft_pro = FT**2
    Vs = lb.local_dephasing_ops(problem["n_qubits"])
    G_gamma = sum(lb.dissipator(V) for V in Vs)
    dn = lb.diamond_norm(G_gamma)
    H_list = [
        problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
        for k in range(tau)
    ]
    GH_list = [lb.hamiltonian_superop(H) for H in H_list]
    L_gamma = 0.5 * c["tf"] * dn.value

    def F_pro(gamma: float) -> float:
        G_list = [GH + float(gamma) * G_gamma for GH in GH_list]
        return lb.process_fidelity(lb.channel(G_list, dt), problem["Uf"])

    F0 = F_pro(0.0)
    res = lb.open_margin(F_pro, L_gamma, ft_pro, margin_tol=1e-6)
    return {
        "controller_index": idx,
        "dnorm_dephasing": dn.value,
        "L_gamma": L_gamma,
        "F_pro_0": F0,
        "r0_gamma": (F0 - ft_pro) / L_gamma,
        "M_gamma": res.M_plus,
    }


def main() -> None:
    problem, controllers = load_ensemble()
    out = {
        "FT": FT,
        "margin_tol": MARGIN_TOL,
        "records": [_record(problem, controllers, i) for i in INDICES],
    }
    open_rec = _open_record(problem, controllers, INDICES[0])
    if open_rec is not None:
        out["open_records"] = [open_rec]
    else:
        print("cvxpy unavailable: open-system record omitted")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(
        f"Wrote {OUT} ({len(out['records'])} records"
        f"{', 1 open record' if open_rec else ''})"
    )


if __name__ == "__main__":
    main()

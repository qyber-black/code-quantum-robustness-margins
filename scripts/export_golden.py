#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export golden numeric fixtures for MATLAB/Python consistency tests."""

from __future__ import annotations

import json
from pathlib import Path

from qrobustness import (
    dH_structure,
    differential_sensitivity,
    iterative_margin,
    lipschitz_constant,
    load_controllers,
    load_problem,
    make_fidelity_fn,
    perturbed_hamiltonians,
    structure_constant,
    uncertainty_rates,
)
from qrobustness.kosut import margin as kosut_margin

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
OUT = ROOT / "data/reference/case_study_subset.json"

FT = 0.999
ETA = 1e-6
# 0-based indices into the filtered (eps0 <= 1e-4) controller list:
# 0 = lowest \varepsilon_0, 58 = most asymmetric M+/- (H0), 60 = highest \varepsilon_0
INDICES = [0, 58, 60]
KOSUT_N_QUAD = 24
KOSUT_N_DEV = 17
STRUCTURES = ("H0", "H1", "H2")


def _record(problem, controllers, idx: int, tag: str) -> dict:
    c = controllers[idx]
    dt = c["tf"] / c["tau"]
    if tag == "H0":
        C = structure_constant("drift", problem["H0"], dt, c["tau"])
    elif tag == "H1":
        C = structure_constant("control", problem["H1"], dt, c["tau"], c["u1"])
    else:
        C = structure_constant("control", problem["H2"], dt, c["tau"], c["u2"])
    L = lipschitz_constant(FT, problem["dim"], C)
    fid_fn = make_fidelity_fn(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], problem["Uf"], dt, tag
    )
    F0 = fid_fn(0.0)
    margin = iterative_margin(fid_fn, L, FT, mu0=0.0, eta=ETA)
    H_list = perturbed_hamiltonians(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag, 0.0
    )
    dH = dH_structure(problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag)
    zeta = differential_sensitivity(H_list, dH, dt, problem["Uf"], n_quad=32)
    # Supplementary Kosut et al. bound (arXiv:2507.01215); see qrobustness/kosut.py.
    rates = uncertainty_rates(H_list, dH, dt, n_quad=KOSUT_N_QUAD, n_dev=KOSUT_N_DEV)
    return {
        "controller_index": idx,
        "structure": tag,
        "fid": float(F0),
        "L": float(L),
        "C": float(C),
        "zeta": float(zeta),
        "M": float(margin.M),
        "M_minus": float(margin.M_minus),
        "M_plus": float(margin.M_plus),
        "converged_minus": bool(margin.converged_minus),
        "converged_plus": bool(margin.converged_plus),
        "k_w_unc": float(rates.w_unc),
        "k_w_avg": float(rates.w_avg),
        "k_w_dev": float(rates.w_dev),
        "k_T": float(rates.T),
        "k_M": float(kosut_margin(rates, FT, nominal_error=c["error"])),
    }


def main() -> None:
    problem = load_problem(CTRL / "problem9.mat")
    controllers = load_controllers(CTRL / "controllers.csv", 1e-4)
    records = []
    for idx in INDICES:
        for tag in STRUCTURES:
            records.append(_record(problem, controllers, idx, tag))
    payload = {
        "meta": {
            "FT": FT,
            "eta": ETA,
            "n_quad": 32,
            "indices": INDICES,
            "kosut_n_quad": KOSUT_N_QUAD,
            "kosut_n_dev": KOSUT_N_DEV,
            # How each approximated quantity is evaluated.  n_quad and
            # kosut_n_quad are recorded for provenance; zeta and <Htil> are
            # closed form and do not use them.
            "zeta_method": "exact",
            "kosut_w_avg": "exact",
            "kosut_w_dev": "grid+brent, adaptive",
            "note": "0=min eps0, 58=asymmetric M+/- H0, 60=max eps0",
        },
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUT} ({len(records)} records)")


if __name__ == "__main__":
    main()

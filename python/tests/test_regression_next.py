# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression fixtures for the multiparameter/time-varying/open layers.

Python-internal peer of test_consistency.py for the layers without a
MATLAB counterpart; run scripts/export_golden_next.py to (re)generate
the fixture after an intentional numeric change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from qrobustness import load_controllers, load_problem
from qrobustness import multiparam as mp
from qrobustness import timevarying as tv

ROOT = Path(__file__).resolve().parents[2]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
GOLDEN = ROOT / "data/reference/paper_next_subset.json"

ATOL = 1e-10
RTOL = 1e-8
STRUCTURES = ("H0", "H1", "H2")


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip(f"Missing fixture {GOLDEN}; run scripts/export_golden_next.py")
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="module")
def case():
    problem = load_problem(CTRL / "problem9.mat")
    controllers = load_controllers(CTRL / "controllers.csv", 1e-4)
    return problem, controllers


def _close(a, b, name=""):
    if not np.isclose(a, b, rtol=RTOL, atol=ATOL):
        raise AssertionError(f"{name}: {a} vs {b} (rtol={RTOL}, atol={ATOL})")


def test_multiparam_tv_match_fixture(golden, case):
    problem, controllers = case
    ft = golden["FT"]
    for rec in golden["records"]:
        c = controllers[rec["controller_index"]]
        dt = c["tf"] / c["tau"]
        specs = [
            ("drift", problem["H0"]),
            ("control", problem["H1"], c["u1"]),
            ("control", problem["H2"], c["u2"]),
        ]
        _, L = mp.structure_constants(specs, dt, c["tau"], ft, problem["dim"])
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
        P = mp.safe_polytope(np.zeros(3), np.asarray(L), F0, ft)
        tag = f"ctrl[{rec['controller_index']}]"
        _close(F0, rec["fid"], f"{tag}.fid")
        for j in range(3):
            _close(L[j], rec["L"][j], f"{tag}.L[{j}]")
        _close(P.inradius_linf, rec["inradius_linf"], f"{tag}.inradius_linf")
        _close(P.inradius_l2, rec["inradius_l2"], f"{tag}.inradius_l2")
        _close(tv.uniform_margin(L, F0, ft), rec["r0_joint"], f"{tag}.r0_joint")
        for name, d in (
            ("+e1", np.array([0.0, 1.0, 0.0])),
            ("diagppp", np.ones(3) / np.sqrt(3.0)),
        ):
            res = mp.directional_margin(fn, L, ft, d, margin_tol=golden["margin_tol"])
            _close(res.M, rec[f"M_{name}"], f"{tag}.M_{name}")
            _close(res.M_upper, rec[f"Mub_{name}"], f"{tag}.Mub_{name}")


def test_open_matches_fixture(golden, case):
    if "open_records" not in golden:
        pytest.skip("Fixture has no open-system record (cvxpy unavailable at export)")
    cvxpy = pytest.importorskip("cvxpy")  # noqa: F841
    from qrobustness import lindblad as lb

    problem, controllers = case
    ft_pro = golden["FT"] ** 2
    for rec in golden["open_records"]:
        c = controllers[rec["controller_index"]]
        dt = c["tf"] / c["tau"]
        Vs = lb.local_dephasing_ops(problem["n_qubits"])
        G_gamma = sum(lb.dissipator(V) for V in Vs)
        dn = lb.diamond_norm(G_gamma)
        # The SDP solver terminates on a tolerance, not to roundoff.
        assert dn.value == pytest.approx(rec["dnorm_dephasing"], rel=1e-6)
        H_list = [
            problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
            for k in range(c["tau"])
        ]
        GH_list = [lb.hamiltonian_superop(H) for H in H_list]
        L_gamma = 0.5 * c["tf"] * dn.value
        assert L_gamma == pytest.approx(rec["L_gamma"], rel=1e-6)

        def F_pro(gamma: float) -> float:
            G_list = [GH + float(gamma) * G_gamma for GH in GH_list]
            return lb.process_fidelity(lb.channel(G_list, dt), problem["Uf"])

        _close(F_pro(0.0), rec["F_pro_0"], "F_pro_0")
        res = lb.open_margin(F_pro, L_gamma, ft_pro, margin_tol=1e-6)
        # M depends on L_gamma (solver-tolerance limited) only through
        # step lengths; bracket at the solver tolerance.
        assert res.M_plus == pytest.approx(rec["M_gamma"], rel=1e-5)

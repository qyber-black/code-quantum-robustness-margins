# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-check Python against golden fixtures (and MATLAB export if present)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
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

ROOT = Path(__file__).resolve().parents[2]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
GOLDEN = ROOT / "data/reference/case_study_subset.json"
MATLAB_GOLDEN = ROOT / "data/reference/case_study_subset_matlab.json"

ATOL = 1e-10
RTOL = 1e-8

FIELDS = ("fid", "L", "C", "zeta", "M", "M_minus", "M_plus")
# Supplementary Kosut et al. bound fields; absent from pre-existing goldens.
KOSUT_FIELDS = ("k_w_unc", "k_w_avg", "k_w_dev", "k_T", "k_M")
KOSUT_N_QUAD = 24
KOSUT_N_DEV = 17


def _close(a, b, name=""):
    if not np.isclose(a, b, rtol=RTOL, atol=ATOL):
        raise AssertionError(f"{name}: {a} vs {b} (rtol={RTOL}, atol={ATOL})")


def _compute_python_record(problem, controllers, idx, tag, FT=0.999, eta=1e-6):
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
    margin = iterative_margin(fid_fn, L, FT, mu0=0.0, eta=eta)
    H_list = perturbed_hamiltonians(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag, 0.0
    )
    dH = dH_structure(problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag)
    zeta = differential_sensitivity(H_list, dH, dt, problem["Uf"], n_quad=32)
    rates = uncertainty_rates(H_list, dH, dt, n_quad=KOSUT_N_QUAD, n_dev=KOSUT_N_DEV)
    return {
        "fid": F0,
        "L": L,
        "C": C,
        "zeta": zeta,
        "M": margin.M,
        "M_minus": margin.M_minus,
        "M_plus": margin.M_plus,
        "converged_minus": bool(margin.converged_minus),
        "converged_plus": bool(margin.converged_plus),
        "k_w_unc": rates.w_unc,
        "k_w_avg": rates.w_avg,
        "k_w_dev": rates.w_dev,
        "k_T": rates.T,
        "k_M": kosut_margin(rates, FT, nominal_error=c["error"]),
    }


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip(f"Missing golden file {GOLDEN}; run scripts/export_golden.py")
    return json.loads(GOLDEN.read_text())


def test_python_matches_golden(golden):
    problem = load_problem(CTRL / "problem9.mat")
    controllers = load_controllers(CTRL / "controllers.csv", 1e-4)
    assert len(golden["records"]) >= 9  # >=3 controllers x 3 structures
    for rec in golden["records"]:
        got = _compute_python_record(
            problem, controllers, rec["controller_index"], rec["structure"]
        )
        keys = FIELDS + tuple(f for f in KOSUT_FIELDS if f in rec)
        for key in keys:
            _close(got[key], rec[key], f"{rec['structure']}[{rec['controller_index']}].{key}")
        assert got["converged_minus"] is True
        assert got["converged_plus"] is True
        assert rec["converged_minus"] is True
        assert rec["converged_plus"] is True


def test_matlab_python_consistency_if_present(golden):
    if not MATLAB_GOLDEN.exists():
        pytest.skip("MATLAB golden export not present; run make export-golden")
    matlab = json.loads(MATLAB_GOLDEN.read_text())
    by_key = {(r["controller_index"], r["structure"]): r for r in golden["records"]}
    for rec in matlab["records"]:
        key = (rec["controller_index"], rec["structure"])
        assert key in by_key
        ref = by_key[key]
        fields = FIELDS + tuple(f for f in KOSUT_FIELDS if f in rec and f in ref)
        for field in fields:
            _close(rec[field], ref[field], f"matlab vs py {key} {field}")
        assert bool(rec["converged_minus"]) is True
        assert bool(rec["converged_plus"]) is True
        assert bool(ref["converged_minus"]) is True
        assert bool(ref["converged_plus"]) is True


def test_golden_has_kosut_fields(golden):
    """The golden fixture must carry the supplementary Kosut et al. values."""
    if not all(f in golden["records"][0] for f in KOSUT_FIELDS):
        pytest.skip("Golden predates the Kosut comparison; run make export-golden")
    for rec in golden["records"]:
        assert rec["k_w_unc"] > 0
        assert rec["k_M"] > 0
        assert rec["k_M"] <= rec["M"], "structured margin should not be the smaller one"

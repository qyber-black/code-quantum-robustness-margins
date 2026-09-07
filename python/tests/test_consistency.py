# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-check the Python implementation against its MATLAB peer.

The reference is MATLAB's committed results table, not a recorded fixture.
A fixture asserts that some previous answer was correct, which it never
established; two independent implementations agreeing is evidence, and if
they move together the committed tree shows it in git. This is the subset
that runs without MATLAB installed; make test-parity does the full
61-controller comparison in both directions.
"""

from __future__ import annotations

import csv
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
#: The committed peer tree. Comparing against MATLAB's published table
#: rather than a recorded fixture keeps the check cross-language -- two
#: independent implementations agreeing is evidence -- without a third copy
#: of the numbers to re-record. The full 61-controller comparison is
#: make test-parity; this is the subset that runs without MATLAB installed.
MATLAB_TABLE = ROOT / "results/lipschitz-margin-matlab/margins_table_0.999.csv"
#: Controllers to check live, chosen as in the drivers: smallest nominal
#: error, an asymmetric-margin case, and the largest nominal error.
INDICES = (0, 58, 60)

ATOL = 1e-10
RTOL = 1e-8

FIELDS = ("fid", "L", "C", "zeta", "M", "M_minus", "M_plus")
# Supplementary Kosut et al. bound fields, checked as a property below.
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
        problem["H0"],
        problem["H1"],
        problem["H2"],
        c["u1"],
        c["u2"],
        problem["Uf"],
        dt,
        tag,
    )
    F0 = fid_fn(0.0)
    # margin_tol matches the driver that wrote the peer table.
    margin = iterative_margin(fid_fn, L, FT, mu0=0.0, eta=eta, margin_tol=1e-8)
    H_list = perturbed_hamiltonians(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag, 0.0
    )
    dH = dH_structure(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag
    )
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
        # Both engines implement the angular absorption of 1.0.1, so the
        # default is what they must agree on.
        "k_M": kosut_margin(rates, FT, nominal_error=c["error"]),
    }


@pytest.fixture(scope="module")
def matlab_table():
    # A skip, not a failure. The table is committed, so a fresh checkout
    # has it; but make maintainer-clean deliberately removes every
    # generated result, and running the tests before rebuilding them is a
    # legitimate order. Skipping is right for a missing input; a wrong
    # value is what this test is for.
    if not MATLAB_TABLE.is_file():
        pytest.skip(
            f"{MATLAB_TABLE.relative_to(ROOT)} not present; "
            "run make paper-QRM-margins ENGINE=matlab"
        )
    with MATLAB_TABLE.open(newline="") as f:
        return {int(r["controller"]) - 1: r for r in csv.DictReader(f)}


def test_python_matches_matlab(matlab_table):
    """Live Python against MATLAB's committed table, per controller and
    structure: the cross-language check the peer implementations exist to
    pass. Nothing here records a previous Python answer -- if the two
    implementations drift apart, one of them is wrong, and if they move
    together the committed tree shows it in git."""
    problem = load_problem(CTRL / "problem9.mat")
    controllers = load_controllers(CTRL / "controllers.csv", 1e-4)
    for idx in INDICES:
        assert idx in matlab_table, f"controller {idx + 1} absent from the peer table"
        ref = matlab_table[idx]
        for structure in ("H0", "H1", "H2"):
            got = _compute_python_record(problem, controllers, idx, structure)
            _close(got["fid"], float(ref["fid"]), f"{structure}[{idx}].fid")
            for key, col in (
                ("M", f"M_{structure}"),
                ("M_minus", f"Mm_{structure}"),
                ("M_plus", f"Mp_{structure}"),
                ("zeta", f"zeta_{structure}"),
            ):
                _close(got[key], float(ref[col]), f"{structure}[{idx}].{key}")
            assert got["converged_minus"] is True
            assert got["converged_plus"] is True


def test_kosut_margin_does_not_exceed_the_structured_one(matlab_table):
    """A property, not a recording: the universal specialisation is a lower
    certificate, so it must not exceed the structured margin it is compared
    against."""
    problem = load_problem(CTRL / "problem9.mat")
    controllers = load_controllers(CTRL / "controllers.csv", 1e-4)
    for idx in INDICES:
        for structure in ("H0", "H1", "H2"):
            got = _compute_python_record(problem, controllers, idx, structure)
            assert got["k_w_unc"] > 0
            assert got["k_M"] > 0
            assert got["k_M"] <= got["M"], (
                f"{structure}[{idx}]: universal margin {got['k_M']} exceeds "
                f"the structured margin {got['M']}"
            )

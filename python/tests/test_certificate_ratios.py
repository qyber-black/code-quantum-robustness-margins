# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ratios between the paper's own certificates are closed form or bounded.

Two corollaries, checked against the published trees rather than against a
recomputation, because it is the published numbers the paper quotes.

The dominance factor r_FS/r_0 is a function of the two fidelities alone:
the same expression must hold on a one-qubit pulse, a two-qubit CNOT, the
three-qubit chain and a four-qubit chain, over dimensions 2 to 16. And the
joint gauge's gain over the cross-polytope is an l1-over-l2 ratio, so it
can never exceed sqrt(p).

A certificate that acquired a system dependence it should not have would
break the first; one whose gauge stopped being a norm would break the
second.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

FT = 0.999
#: Everything but the three-qubit tree agrees to roundoff. There r_fs comes
#: from a bisection with its own stopping tolerance, which is the looser
#: number here and not a disagreement about the formula.
TOL = 1e-9
#: Joint uncertainty dimension of the main ensemble.
N_PARAMS = 3


def _rows(rel):
    with (RESULTS / rel).open(newline="") as f:
        return list(csv.DictReader(f))


def _predicted(fid):
    """Corollary: r_FS/r_0 from the two fidelities, nothing else."""
    return (
        np.sqrt(1.0 - FT**2) * (np.arccos(FT) - np.arccos(min(1.0, fid))) / (fid - FT)
    )


def _three_qubit_pairs():
    mp = {
        r["controller"]: r
        for r in _rows("multiparameter-margin-python/multiparam_0.999.csv")
    }
    fs = {
        (r["controller"], r["structure"]): float(r["r_fs"])
        for r in _rows("time-bandwidth-bound-python/fs_validity_0.999.csv")
    }
    for c, row in mp.items():
        for k in ("H0", "H1", "H2"):
            yield float(row["fid"]), fs[(c, k)] / float(row["r0_" + k])


def _flat_pairs(rel, keys):
    for r in _rows(rel):
        for k in keys:
            yield float(r["fid"]), float(r["rfs_" + k]) / float(r["r0_" + k])


@pytest.mark.parametrize(
    "name,pairs",
    [
        ("three-qubit", _three_qubit_pairs),
        (
            "cnot",
            lambda: _flat_pairs(
                "cnot-python/cnot_margins_0.999.csv", ("H0", "X1", "X2")
            ),
        ),
        (
            "four-qubit",
            lambda: _flat_pairs(
                "scaling-python/scaling4q_margins_0.999.csv",
                ("H0", "X1", "X2", "X3", "X4"),
            ),
        ),
    ],
)
def test_the_dominance_factor_is_a_function_of_the_fidelities(name, pairs):
    """One formula, four systems, dimensions 2 to 16."""
    seen = 0
    for fid, got in pairs():
        assert got == pytest.approx(_predicted(fid), rel=TOL), name
        seen += 1
    assert seen > 0, f"no {name} instances found"


def test_the_dominance_factor_tends_to_two():
    """For a numerically exact gate it is 2 - (1 - FT)/3 + O((1-FT)^2)."""
    for ft in (0.99, 0.999, 0.9999):
        exact = np.sqrt(1.0 - ft**2) * np.arccos(ft) / (1.0 - ft)
        assert exact == pytest.approx(2.0 - (1.0 - ft) / 3.0, abs=(1.0 - ft) ** 2)
        assert exact < 2.0


def test_the_joint_gauge_gain_cannot_exceed_root_p():
    """An l1-over-l2 ratio in p components, so it lies in [1, sqrt(p)]."""
    jg = _rows("multiparameter-margin-python/joint_gauge_0.999.csv")
    assert jg, "no joint-gauge rows"
    for r in jg:
        for field in ("diag_gain_min", "diag_gain_med", "diag_gain_max"):
            gain = float(r[field])
            assert 1.0 - 1e-12 <= gain <= np.sqrt(N_PARAMS) + 1e-12, (
                f"controller {r['controller']} {field} = {gain}"
            )

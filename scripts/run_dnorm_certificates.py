#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Diagnostics for every diamond norm the certificates rest on.

A certified diamond norm is the solver's optimum repaired to exact
feasibility and re-evaluated upward, so three quantities describe it: the
solver optimum, the verified positive-semidefinite shift applied to the
primal iterate, and the inflation the upward evaluation adds. The paper
quotes bounds on the last of these and on the deviation of the certified
value from the closed forms of Lemma 2n, and neither was recorded
anywhere -- the appendix said they were.

The norm depends only on the generator, not on the controller or the
threshold, so the distinct generators are few and this driver is cheap.
It covers the two dissipative families the paper certifies at the sizes
it proves them for, and the coherent superoperators of the three-qubit
case study.

Writes results/lindblad-margin-python/dnorm_certificates.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _drivers import PAULI_Z, load_ensemble, write_rows
from qrobustness import lindblad as lb

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/lindblad-margin-python"

SM = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)  # sigma_-

#: Qubit counts for the exact families. The closed form 2n is proved for
#: all n; these are the sizes the appendix reports a check at.
FAMILY_SIZES = (1, 2, 3)


def _row(name, n_qubits, S, closed_form=None):
    """One certified norm and the three numbers that describe it."""
    d = lb.diamond_norm(S)
    return {
        "generator": name,
        "n_qubits": n_qubits,
        "dim": S.shape[0],
        "raw": d.raw,
        "certified": d.value_certified,
        "gap": d.gap,
        # The inflation the paper bounds: what the upward evaluation adds,
        # relative to the optimum it is inflating.
        "rel_inflation": d.gap / d.raw,
        "feas_shift": d.feas_shift,
        "status": d.status,
        # Which solver produced it. The value depends on the choice, and
        # leaving it to cvxpy is how the published numbers came to differ
        # across a virtualenv rebuild with identical package versions.
        "solver": d.solver,
        "closed_form": "" if closed_form is None else closed_form,
        "dev_closed_form": (
            "" if closed_form is None else abs(d.value_certified - closed_form)
        ),
    }


def main() -> None:
    """Certify each generator and record what the certification cost."""
    # No --FT and no --controllers: a diamond norm is a property of the
    # generator alone, so neither would change a single number here.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    rows = []
    for n in FAMILY_SIZES:
        for name, op in (("dephasing", PAULI_Z), ("amp_damping", SM)):
            G = sum(lb.dissipator(V) for V in lb.local_ops(op, n))
            rows.append(_row(name, n, G, closed_form=2 * n))

    problem, _ = load_ensemble()
    nq = problem["n_qubits"]
    for k in ("H0", "H1", "H2"):
        rows.append(_row(f"coherent_{k}", nq, lb.hamiltonian_superop(problem[k])))

    for r in rows:
        dev = r["dev_closed_form"]
        print(
            f"{r['generator']:>13s} n={r['n_qubits']}  "
            f"certified={r['certified']:.9f}  "
            f"rel_inflation={r['rel_inflation']:.3e}  "
            f"shift={r['feas_shift']:.3e}  "
            + (f"dev_closed_form={dev:.3e}  " if dev != "" else "")
            + f"({r['status']})",
            flush=True,
        )

    write_rows(args.out / "dnorm_certificates.csv", rows)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Diagnostics for every diamond norm used by the certificates.

Computes the certified diamond norm, with its solver optimum, verified
feasibility shift and upward-evaluation inflation, for the local dephasing
and amplitude-damping dissipators on 1, 2 and 3 qubits (closed form 2n) and
the coherent superoperators of H0, H1, H2 of the main ensemble (the xQRM
paper, Verified diamond-norm upper bounds). The norm depends only on the
generator, so there is no --FT. Options: --out.

Writes results/lindblad-margin-python/dnorm_certificates.csv:
    generator, n_qubits, dim: the generator and its size.
    raw, certified, gap, rel_inflation: solver optimum, certified upper bound,
        their difference, and gap/raw.
    feas_shift, status, solver: PSD shift applied to the primal iterate,
        solver status and solver name.
    closed_form, dev_closed_form: 2n and |certified - 2n| (dissipators only).
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

#: Qubit counts at which the dissipative families are checked against 2n.
FAMILY_SIZES = (1, 2, 3)


def _row(name, n_qubits, S, closed_form=None):
    """One certified norm and the three numbers that describe that norm."""
    d = lb.diamond_norm(S)
    return {
        "generator": name,
        "n_qubits": n_qubits,
        "dim": S.shape[0],
        "raw": d.raw,
        "certified": d.value_certified,
        "gap": d.gap,
        # Upward-evaluation inflation relative to the solver optimum.
        "rel_inflation": d.gap / d.raw,
        "feas_shift": d.feas_shift,
        "status": d.status,
        # The value depends on the solver, so it is recorded.
        "solver": d.solver,
        "closed_form": "" if closed_form is None else closed_form,
        "dev_closed_form": (
            "" if closed_form is None else abs(d.value_certified - closed_form)
        ),
    }


def main() -> None:
    """Certify each generator and record what that certification cost."""
    # No --FT: a diamond norm depends on the generator alone.

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

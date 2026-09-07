#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Input handling shared by the three paper-artefact generators.

The tables, figures and macros generators each carried their own copy of
this, and the copies had drifted: tables raised a message naming the
driver that produces the missing file, macros raised a plainer one, and
figures had lost the existence check altogether, so a missing input
surfaced as a bare FileNotFoundError that --allow-missing could not
cover. One copy is how that stops recurring.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

#: Which driver produces each results file. Keyed by filename, not by
#: results tree: several drivers write into the same tree, so a tree-keyed
#: map named the wrong driver for at least six inputs -- it sent you to
#: run_time_bandwidth_bound_comparison.py for fs_validity_0.999.csv, which
#: run_fs_validity.py writes. Following that advice leaves the file still
#: missing. test_paper_driver_map keeps this in step with the Makefile.
DRIVERS = {
    "berberich_comparison_0.999.csv": "run_berberich_comparison.py",
    "budget_sweep_ctrl16_H1.csv": "run_budget_sweep.py",
    "cnot_margins_0.999.csv": "run_cnot_case_study.py",
    "duration_sweep_0.999.csv": "run_duration_sweep.py",
    "focal_tests_0.999.csv": "run_lipschitz_margin_case_study.py",
    "fs_validity_0.999.csv": "run_fs_validity.py",
    "joint_gauge_0.999.csv": "run_joint_gauge.py",
    "kosut_comparison_0.999.csv": "run_time_bandwidth_bound_comparison.py",
    "kosut_comparison_0.999_angular.csv": "run_time_bandwidth_bound_comparison.py",
    "kosut_comparison_0.999_angular_tv.csv": "run_time_bandwidth_bound_comparison.py",
    "margins_table_0.999.csv": "run_lipschitz_margin_case_study.py",
    "mixed_ctrl1.npz": "run_mixed_example.py",
    "multiparam_0.999.csv": "run_multiparameter_case_study.py",
    "multiparam_0.999_angular.csv": "run_multiparameter_case_study.py",
    "open_amp_0.999.csv": "run_open_amplitude_damping.py",
    "open_coherent_0.999.csv": "run_open_system_case_study.py",
    "open_margins_0.999.csv": "run_open_system_case_study.py",
    "dnorm_certificates.csv": "run_dnorm_certificates.py",
    "open_threshold_sweep.csv": "run_open_threshold_sweep.py",
    # Cohort membership per threshold, written beside the sweep by the
    # same driver: only controllers nominally above a threshold are
    # eligible for it, and a summary must say how many it covers.
    "open_threshold_cohort.csv": "run_open_threshold_sweep.py",
    "robust_vs_nominal_0.999.csv": "run_robust_vs_nominal.py",
    "scaling4q_margins_0.999.csv": "run_scaling_example.py",
    "single_qubit_0.999.csv": "run_single_qubit_example.py",
    "slice_ctrl1_0.999.npz": "run_slice_scan.py",
    "tv_bracket_0.999.csv": "run_multiparameter_case_study.py",
    "validity_0.999.csv": "run_kosut_validity.py",
    "validity_0.999_tv.csv": "run_kosut_validity.py",
    "validity_witness_0.999.csv": "run_kosut_validity.py",
    "verification_0.999.csv": "run_theorem_verification.py",
}

_ALLOW_MISSING = False
_RES: Path | None = None


def configure(results_root: Path, allow_missing: bool) -> None:
    """Set the results root and the --allow-missing policy for this run."""
    global _RES, _ALLOW_MISSING
    _RES, _ALLOW_MISSING = results_root, allow_missing


def describe(path: Path) -> str:
    """Say what is missing and which driver produces it."""
    try:
        rel = path.relative_to(_RES) if _RES else path
    except ValueError:
        # describe() is the error path; it must not raise one of its own.
        rel = path
    driver = DRIVERS.get(path.name, "the corresponding driver")
    return f"missing input results/{rel}; produce it with scripts/{driver}"


def have(path: Path, target: Path | None = None) -> bool:
    """True if an input is present.

    Absent input is fatal unless --allow-missing was passed. Under
    --allow-missing any stale target is deleted rather than left behind,
    so a skipped artefact cannot masquerade as a fresh one.
    """
    if path.exists():
        return True
    msg = describe(path)
    if not _ALLOW_MISSING:
        raise SystemExit(
            f"ERROR: {msg}\n(pass --allow-missing to skip it during development)"
        )
    print(f"WARNING: {msg} -- skipping", file=sys.stderr)
    if target is not None and target.exists():
        target.unlink()
        print(f"WARNING: removed stale {target.name}", file=sys.stderr)
    return False


def read(path: Path, required: bool = True) -> list:
    """Rows of a results CSV.

    A missing input aborts, even under --allow-missing, unless the caller
    passes ``required=False`` to say it can cope with no rows. That
    default is deliberate: returning [] unconditionally turned a one-line
    diagnostic into an IndexError three lines later at every one of the
    thirty-odd unguarded call sites, because almost all of them index
    rows[0] or take a median. --allow-missing is honoured by have(),
    which skips a whole artefact and removes its stale target; that is
    the designed path, and a bare read() is not on it.
    """
    if not path.exists():
        if _ALLOW_MISSING and not required:
            print(f"WARNING: {describe(path)} -- skipping", file=sys.stderr)
            return []
        raise SystemExit(f"ERROR: {describe(path)}")
    with path.open() as fh:
        return list(csv.DictReader(fh))


def col(rows, key) -> np.ndarray:
    """One numeric column of ``rows`` as a float array."""
    return np.array([float(r[key]) for r in rows])

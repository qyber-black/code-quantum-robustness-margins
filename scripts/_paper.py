#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Input handling for the gen_paper_xqrm_* generators.

Locates results files, reads CSVs, and on a missing input names the driver
that produces it; --allow-missing skips an artefact instead of failing."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

#: Results file name -> the driver that writes it (several drivers share a
#: results tree). test_paper_driver_map keeps this aligned with the Makefile.
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
    # Controllers nominally above each threshold of the sweep.
    "open_threshold_cohort.csv": "run_open_threshold_sweep.py",
    "robust_vs_nominal_0.999.csv": "run_robust_vs_nominal.py",
    "scaling4q_margins_0.999.csv": "run_scaling_example.py",
    "single_qubit_0.999.csv": "run_single_qubit_example.py",
    "ghz_detuning_0.999.csv": "run_state_examples.py",
    "tfim_preparation_0.999.csv": "run_state_examples.py",
    "ghz_dephasing_0.999.csv": "run_state_examples.py",
    "state_variance_0.999.csv": "run_state_examples.py",
    "closed_limit.csv": "run_state_examples.py",
    "crosstalk_0.999.csv": "run_algorithm_tests.py",
    "rays_0.999.csv": "run_algorithm_tests.py",
    "brackets_0.999.csv": "run_bracket_audit.py",
    "timing_0.999.csv": "run_bracket_audit.py",
    "environment.json": "run_bracket_audit.py",
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
    """Assign the results root and the --allow-missing policy for this run."""
    global _RES, _ALLOW_MISSING
    _RES, _ALLOW_MISSING = results_root, allow_missing


def describe(path: Path) -> str:
    """State what is missing and which driver produces it."""
    try:
        rel = path.relative_to(_RES) if _RES else path
    except ValueError:
        # describe() is the error path; it must not raise an error of its own.
        rel = path
    driver = DRIVERS.get(path.name, "the corresponding driver")
    return f"missing input results/{rel}; produce it with scripts/{driver}"


def have(path: Path, target: Path | None = None) -> bool:
    """True when an input is present.

    Missing input is fatal unless --allow-missing was passed. Under
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

    A missing input fails, even under --allow-missing, unless
    ``required=False``, in which case [] is returned under --allow-missing.
    Use have() to skip a whole artefact.
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

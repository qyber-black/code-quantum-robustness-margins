#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check that a paper's results can be reproduced.

Reruns the paper's drivers into a separate scratch tree (--scratch, default
build/repro/<paper>) with the production flags from _invocations and
SERIAL_BLAS_ENV, then:

1. compares every committed CSV in the paper's results/ trees with its
   recomputed copy (numeric columns to --rtol/--atol, text exactly, the
   VOLATILE_COLUMNS skipped; a file not recomputed is reported missing);
2. for xqrm, regenerates the tables and macros from results/ and checks
   that every generated macro, table and figure is used by the paper and
   every one the paper uses exists;
3. for xqrm, compares the generated tables, figures and macros.tex with
   the copies in the paper repository.

Prints each mismatch and missing file; exits 1 if there are any.

Usage: python3 scripts/check_reproducible.py --paper xqrm [--rtol 1e-9]
                                             [--skip-recompute]"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from _invocations import DRIVER_RUNS, SERIAL_BLAS_ENV

ROOT = Path(__file__).resolve().parents[1]

# Default workers for the adversarial sweeps (capped at 32); --jobs, which
# the Makefile sets from JOBS, overrides it. Seeds depend on the job, not
# on execution order, so the setting does not change the results.
DEFAULT_JOBS = str(min(32, os.cpu_count() or 4))


def _with_jobs(table, jobs):
    """Copy of the DRIVER_RUNS table with --jobs added for the drivers that
    accept it."""
    out = {k: [list(run) for run in runs] for k, runs in table.items()}
    for script in (
        "run_kosut_validity.py",
        "run_fs_validity.py",
        "run_bracket_audit.py",
    ):
        runs = out.setdefault(script, [[]])
        for run in runs:
            run += ["--jobs", str(jobs)]
    return out


# Per-paper manifest: the drivers to rerun and the result tree each writes,
# the trees to compare, and the artefacts the paper carries.
PAPERS = {
    "xqrm": {
        "paper_root": ROOT / ".." / "paper-xQRM",
        "artefact_script": "gen_paper_xqrm_{}.py",
        "artefact_dir": "paper-xqrm",
        # Argument lists per driver, one run per list (from _invocations,
        # with --jobs added).
        "driver_runs": _with_jobs(DRIVER_RUNS, DEFAULT_JOBS),
        "trees": [
            "multiparameter-margin-python",
            "time-bandwidth-bound-python",
            "single-qubit-python",
            "cnot-python",
            "scaling-python",
            "lindblad-margin-python",
            "verification-python",
            "state-examples-python",
            "algorithm-tests-python",
            "bracket-audit-python",
        ],
        "drivers": [
            ("run_multiparameter_case_study.py", "multiparameter-margin-python"),
            ("run_joint_gauge.py", "multiparameter-margin-python"),
            ("run_slice_scan.py", "multiparameter-margin-python"),
            ("run_time_bandwidth_bound_comparison.py", "time-bandwidth-bound-python"),
            ("run_kosut_validity.py", "time-bandwidth-bound-python"),
            ("run_fs_validity.py", "time-bandwidth-bound-python"),
            ("run_budget_sweep.py", "time-bandwidth-bound-python"),
            ("run_berberich_comparison.py", "time-bandwidth-bound-python"),
            ("run_single_qubit_example.py", "single-qubit-python"),
            ("run_cnot_case_study.py", "cnot-python"),
            ("run_robust_vs_nominal.py", "cnot-python"),
            ("run_duration_sweep.py", "cnot-python"),
            ("run_scaling_example.py", "scaling-python"),
            ("run_open_system_case_study.py", "lindblad-margin-python"),
            ("run_open_amplitude_damping.py", "lindblad-margin-python"),
            ("run_open_threshold_sweep.py", "lindblad-margin-python"),
            ("run_mixed_example.py", "lindblad-margin-python"),
            ("run_dnorm_certificates.py", "lindblad-margin-python"),
            ("run_theorem_verification.py", "verification-python"),
            ("run_state_examples.py", "state-examples-python"),
            ("run_algorithm_tests.py", "algorithm-tests-python"),
            ("run_bracket_audit.py", "bracket-audit-python"),
        ],
    },
    "qrm": {
        "paper_root": ROOT / ".." / "paper-QRM",
        "artefact_script": None,  # paper 1 carries figures only
        "artefact_dir": None,
        "driver_runs": _with_jobs(DRIVER_RUNS, DEFAULT_JOBS),
        "trees": ["lipschitz-margin-python", "time-bandwidth-bound-python"],
        # A tree is compared whole, so every driver writing into a listed
        # tree must be listed, including those only the xQRM paper uses.
        "drivers": [
            ("run_lipschitz_margin_case_study.py", "lipschitz-margin-python"),
            ("run_time_bandwidth_bound_comparison.py", "time-bandwidth-bound-python"),
            ("run_berberich_comparison.py", "time-bandwidth-bound-python"),
            ("run_budget_sweep.py", "time-bandwidth-bound-python"),
            ("run_fs_validity.py", "time-bandwidth-bound-python"),
            ("run_kosut_validity.py", "time-bandwidth-bound-python"),
        ],
    },
}


class Report:
    def __init__(self):
        self.checked = 0
        self.failures = []
        self.missing = []

    def fail(self, what, detail):
        self.failures.append((what, detail))

    def miss(self, what):
        self.missing.append(what)

    def ok(self):
        return not self.failures and not self.missing


#: The timing columns of run_bracket_audit.py, the only columns exempt from
#: comparison.
VOLATILE_COLUMNS = frozenset(
    {
        "t_preproc_med",
        "t_preproc_iqr",
        "t_eval_med",
        "t_eval_iqr",
        "t_dir_med",
        "t_dir_iqr",
    }
)


def compare_csv(a: Path, b: Path, rtol: float, atol: float, rep: Report):
    """Numeric comparison of two CSVs with identical headers.

    Columns named in ``VOLATILE_COLUMNS`` are omitted; everything else must
    agree to the given tolerance.
    """
    ra = list(csv.reader(a.open()))
    rb = list(csv.reader(b.open()))
    skip = {j for j, name in enumerate(ra[0] if ra else []) if name in VOLATILE_COLUMNS}
    if len(ra) != len(rb):
        rep.fail(a.name, f"row count {len(ra)} vs {len(rb)}")
        return
    for i, (rowa, rowb) in enumerate(zip(ra, rb, strict=True)):
        if len(rowa) != len(rowb):
            rep.fail(a.name, f"row {i}: column count differs")
            return
        for j, (x, y) in enumerate(zip(rowa, rowb, strict=True)):
            if j in skip or x == y:
                continue
            try:
                fx, fy = float(x), float(y)
            except ValueError:
                rep.fail(a.name, f"row {i} col {j}: {x!r} vs {y!r}")
                return
            if math.isnan(fx) and math.isnan(fy):
                continue
            if abs(fx - fy) > atol + rtol * max(abs(fx), abs(fy)):
                rep.fail(
                    a.name,
                    f"row {i} col {j}: {fx!r} vs {fy!r} "
                    f"(rel {abs(fx - fy) / max(abs(fx), abs(fy), 1e-300):.2e})",
                )
                return


def compare_trees(new: Path, old: Path, rtol: float, atol: float, rep: Report):
    """Compare every CSV under ``old`` with the same path under ``new``."""
    for f in sorted(old.rglob("*")):
        if not f.is_file() or f.suffix not in (".csv",):
            continue
        rel = f.relative_to(old)
        g = new / rel
        if not g.exists():
            rep.miss(f"recomputed {rel} (driver did not produce it)")
            continue
        rep.checked += 1
        compare_csv(g, f, rtol, atol, rep)


def compare_text(a: Path, b: Path, rep: Report, what: str):
    if not a.exists():
        rep.miss(f"{what}: {a}")
        return
    if not b.exists():
        rep.miss(f"{what}: {b}")
        return
    rep.checked += 1
    if a.read_text() != b.read_text():
        rep.fail(what, f"{a.name} differs from {b.name}")


def check_paper_contract(paper: Path, art: Path, rep: Report):
    """Check generated artefacts against the paper source, both ways.

    Fails on a macro, table or figure the paper uses that was not
    generated, and on one generated that the paper (main.tex and tables/)
    never uses.
    """
    source = (paper / "main.tex").read_text()
    # Macros used inside the paper's tables count as used.
    for tex in sorted((paper / "tables").glob("*.tex")):
        source += tex.read_text()
    macros = (art / "macros.tex").read_text()
    defined = set(re.findall(r"\\xqdef\{(xq\w+)\}", macros))
    used = set(re.findall(r"\\(xq\w+)", source)) - {"xqdef"}
    for name in sorted(used - defined):
        rep.fail("undefined generated macro", name)
    for name in sorted(defined - used):
        rep.fail("generated macro cited nowhere", name)
    tables_used = set(re.findall(r"\\input\{tables/([^}]+)\}", source))
    for name in sorted(tables_used):
        if not (art / "tables" / f"{name}.tex").exists():
            rep.fail("missing generated table", name)
    for f in sorted((art / "tables").glob("*.tex")):
        if f.stem not in tables_used:
            rep.fail("generated table input nowhere", f.stem)
    figs_used = set(re.findall(r"\\includegraphics\{figures/([^}]+)\}", source))
    for name in sorted(figs_used):
        if not (art / "figures" / f"{name}.pdf").exists():
            rep.fail("missing generated figure", name)
    for f in sorted((art / "figures").glob("*.pdf")):
        if f.stem not in figs_used:
            rep.fail("generated figure included nowhere", f.stem)


def run(cmd, cwd=None):
    """Run ``cmd`` with SERIAL_BLAS_ENV, as the Makefile does; return its code."""
    env = {**os.environ, **SERIAL_BLAS_ENV}
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=env)
    return r.returncode


def supports(script: Path, flag: str) -> bool:
    """Whether a driver accepts an option, queried of the driver itself."""
    try:
        out = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return flag in (out.stdout + out.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", choices=sorted(PAPERS), required=True)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--scratch", type=Path, default=ROOT / "build" / "repro")
    ap.add_argument(
        "--skip-recompute",
        action="store_true",
        help="compare an existing scratch tree (debugging)",
    )
    ap.add_argument(
        "--first", type=int, default=None, help="limit ensembles, for a fast smoke run"
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=int(DEFAULT_JOBS),
        help="workers for the adversarial sweeps (default: %(default)s). "
        "Wall clock only: the seeds depend on the job, not on the order, "
        "so any setting gives byte-identical CSVs.",
    )
    args = ap.parse_args()

    spec = dict(PAPERS[args.paper], driver_runs=_with_jobs(DRIVER_RUNS, args.jobs))
    scratch = args.scratch / args.paper
    rep = Report()
    env_py = sys.executable

    if not args.skip_recompute:
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True)
        for script, tree in spec["drivers"]:
            path = ROOT / "scripts" / script
            # A driver without --out would overwrite results/, the
            # reference, so it is reported missing instead of run.
            if not supports(path, "--out"):
                rep.miss(
                    f"{script} has no --out; cannot recompute without "
                    f"overwriting results/ (add --out to that driver)"
                )
                continue
            out = scratch / tree
            out.mkdir(parents=True, exist_ok=True)
            for extra in spec.get("driver_runs", {}).get(script, [[]]):
                cmd = [env_py, str(path), "--out", str(out)] + list(extra)
                if args.first is not None and supports(path, "--first"):
                    cmd += ["--first", str(args.first)]
                if run(cmd) != 0:
                    rep.fail(script, f"driver exited non-zero ({' '.join(extra)})")

    # 1. recomputed results set against the committed ones
    for tree in spec["trees"]:
        compare_trees(
            scratch / tree, ROOT / "results" / tree, args.rtol, args.atol, rep
        )

    # 2 and 3. artefacts, for papers that include generated ones
    if spec["artefact_script"]:
        art = ROOT / "results" / spec["artefact_dir"]
        for kind in ("tables", "macros"):
            script = ROOT / "scripts" / spec["artefact_script"].format(kind)
            if not script.exists():
                continue
            if run([env_py, str(script)]) != 0:
                rep.fail(script.name, "artefact generator exited non-zero")
        paper = spec["paper_root"]
        check_paper_contract(paper, art, rep)
        if (art / "tables").exists() and (paper / "tables").exists():
            for t in sorted((art / "tables").glob("*.tex")):
                compare_text(
                    t, paper / "tables" / t.name, rep, "paper table out of date"
                )
        # Figures are byte-reproducible (no CreationDate), so compared exactly.
        if (art / "figures").exists() and (paper / "figures").exists():
            for f in sorted((art / "figures").glob("*.pdf")):
                g = paper / "figures" / f.name
                if not g.exists():
                    rep.miss(f"paper figure missing: {f.name}")
                    continue
                rep.checked += 1
                if f.read_bytes() != g.read_bytes():
                    rep.fail("paper figure out of date", f.name)
        if (art / "macros.tex").exists():
            compare_text(
                art / "macros.tex",
                paper / "macros.tex",
                rep,
                "paper macros out of date",
            )

    print()
    print(f"reproduction check: paper={args.paper}  comparisons={rep.checked}")
    for w in rep.missing:
        print(f"  MISSING  {w}")
    for w, d in rep.failures:
        print(f"  MISMATCH {w}: {d}")
    if rep.ok():
        print("OK: recomputed results and paper artefacts agree")
        return
    print(f"FAILED: {len(rep.failures)} mismatches, {len(rep.missing)} missing")
    sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent reproduction check for a paper's results.

Recomputing into the same tree proves nothing: a driver that silently
did not run leaves the previous answer in place and everything agrees
with itself. This recomputes into a SEPARATE tree and compares, so a
disagreement is visible and a driver that fails to run is a missing
file rather than a stale pass.

Three comparisons, each of which can fail independently:

1. recomputed result CSVs against the committed ones in results/ --
   the numbers the drivers produce are stable;
2. artefacts (tables, macros) regenerated from the recomputed tree
   against those regenerated from the committed tree -- the paper
   layer is a pure function of the results;
3. those artefacts against the copies actually sitting in the paper
   repository -- the paper is not carrying something older.

CSVs are compared numerically with a relative tolerance, not byte for
byte: BLAS reassociation and library versions move the last bits, and
a check that fails on that is a check people learn to ignore. Text
columns must match exactly.

Usage: python3 scripts/check_reproducible.py --paper xqrm [--rtol 1e-9]
                                             [--skip-recompute]
"""

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

# Workers for the adversarial sweeps. Each job's seeds depend on the job,
# not on execution order, so this changes the wall clock and nothing else:
# serial and parallel runs produce byte-identical CSVs. Capped so the
# machine stays usable and so the number does not balloon on a big host.
#
# --jobs overrides it and the Makefile passes JOBS through, because
# computing it here meant `make reproduce JOBS=16` throttled the drivers
# the paper target reran and not the ones this script reran, which is the
# expensive half.
DEFAULT_JOBS = str(min(32, os.cpu_count() or 4))


# Per-paper manifest: which drivers to rerun, which result trees they
# write, and which artefacts the paper carries. A third paper is a new
# entry here, not a code change.
def _with_jobs(table, jobs):
    """The shared table, plus --jobs for the drivers that accept it.

    --jobs is deliberately absent from _invocations: it sets worker count,
    not experiment, and the caller supplies it.
    """
    out = {k: [list(run) for run in runs] for k, runs in table.items()}
    for script in ("run_kosut_validity.py", "run_fs_validity.py"):
        runs = out.setdefault(script, [[]])
        for run in runs:
            run += ["--jobs", str(jobs)]
    return out


PAPERS = {
    "xqrm": {
        "paper_root": ROOT / ".." / "paper-xQRM",
        "artefact_script": "gen_paper_xqrm_{}.py",
        "artefact_dir": "paper-xqrm",
        # Extra arguments per driver. A driver may need SEVERAL runs to
        # reproduce everything it contributes: the multiparameter case
        # study writes multiparam_<FT>.csv under --step lipschitz and
        # multiparam_<FT>_angular.csv under --step angular, and the tree
        # holds both -- the angular/Lipschitz evaluation-count macros
        # compare them against each other. One run per list.
        # From scripts/_invocations.py, the single source these flags now
        # share with the Makefile recipes. --jobs is added below, since it
        # is a property of the machine rather than of the experiment.
        "driver_runs": _with_jobs(DRIVER_RUNS, DEFAULT_JOBS),
        "trees": [
            "multiparameter-margin-python",
            "time-bandwidth-bound-python",
            "single-qubit-python",
            "cnot-python",
            "scaling-python",
            "lindblad-margin-python",
            "verification-python",
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
        ],
    },
    "qrm": {
        "paper_root": ROOT / ".." / "paper-QRM",
        "artefact_script": None,  # paper 1 carries figures only
        "artefact_dir": None,
        "driver_runs": _with_jobs(DRIVER_RUNS, DEFAULT_JOBS),
        "trees": ["lipschitz-margin-python", "time-bandwidth-bound-python"],
        # A tree is reproduced whole. Result trees are named after the
        # method, not the paper, so time-bandwidth-bound-python holds the
        # universal-bound comparison this paper publishes alongside the
        # validity sweeps, budget sweep and Berberich comparison the second
        # paper publishes. Every driver that writes into a listed tree is
        # listed here, or compare_trees demands a file nothing produced.
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


#: No column is exempt from comparison. Nothing recorded measures the
#: machine rather than the science: cost is reported as an evaluation
#: count, which is deterministic, and no wall-clock timing is written.
#: A column that could not reproduce would turn every run into a failure
#: and hide the columns that matter.
VOLATILE_COLUMNS = frozenset()


def compare_csv(a: Path, b: Path, rtol: float, atol: float, rep: Report):
    """Numeric comparison of two CSVs with identical headers.

    Columns named in ``VOLATILE_COLUMNS`` are skipped; everything else must
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
    """Generated inputs and source references must match, both ways.

    Undefined is the direction LaTeX would catch anyway. The value here is
    the other one: a macro that is generated but cited nowhere is a number
    the paper computes and does not report, which is what a dropped claim
    looks like. Only numbers the paper uses exist as macros, so a spare one
    is a failure, not housekeeping.
    """
    source = (paper / "main.tex").read_text()
    # Tables carry no macro today, but a macro moving into one must not read
    # as unused, so the citation set spans every generated input the paper has.
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
    """Run a driver under the same environment the Makefile gives it.

    SERIAL_BLAS_ENV is not optional. Without it these drivers spread 8x8
    to 64x64 work across every core as spin-wait: a reproduction run spent
    over half an hour inside run_open_amplitude_damping, which takes 4m53s
    pinned. That the Makefile pinned and this did not was the fourth time
    the two diverged, which is why the environment now lives beside the
    flags in _invocations.
    """
    env = {**os.environ, **SERIAL_BLAS_ENV}
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=env)
    return r.returncode


def supports(script: Path, flag: str) -> bool:
    """Whether a driver accepts an option, asked of the driver itself."""
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
            # A driver with no --out writes into results/, which is the
            # tree being compared against. Running it would overwrite the
            # reference and turn this check into a comparison of a tree
            # with itself, so it is refused rather than run.
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

    # 1. recomputed results against the committed ones
    for tree in spec["trees"]:
        compare_trees(
            scratch / tree, ROOT / "results" / tree, args.rtol, args.atol, rep
        )

    # 2 and 3. artefacts, for papers that carry generated ones
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
        # Figures are byte-reproducible (CreationDate is suppressed), so
        # they can be compared exactly rather than skipped.
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

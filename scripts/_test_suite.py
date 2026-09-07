#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run every test stage for one engine and report a single tally.

Each stage is a Make target, invoked here rather than reimplemented, so
there is still one definition of what a stage does. They all run, whatever
the ones before them did: stopping at the first failure hides how many
other stages would also have failed, which is the number that tells one
broken thing from everything being broken.

Output is streamed as it arrives, so a long stage still shows progress,
and the summary comes at the end with the counts each stage reported.
The exit code is non-zero if any stage failed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The stages, in order. A stage that is not meaningful for an engine is
#: still run: the target itself says so and exits cleanly, which is more
#: honest than silently skipping it here.
STAGES = ("test-lint", "test-unit", "test-synth", "test-parity")

#: pytest's summary line, e.g. "2 failed, 241 passed, 1 skipped in 12.3s".
PYTEST_COUNT = re.compile(r"(\d+) (passed|failed|skipped|error|xfailed|xpassed)")

#: The MATLAB/Octave suite's tally from run_all_tests.
MATLAB_COUNT = re.compile(r"(\d+) passed, (\d+) failed of (\d+) tests")

#: One cross-engine table comparison.
COMPARISON = re.compile(r"^overall=(PASS|FAIL)$", re.M)


def counts_from(text: str) -> dict:
    """Test counts a stage reported, whichever runner produced them.

    Returns a dict of outcome -> count. Empty when the stage reports no
    counts at all, which is not the same as reporting zero.
    """
    out: dict[str, int] = {}
    # The MATLAB tally is read first and then removed: its wording ("17
    # passed, 2 failed of 19 tests") also matches the pytest pattern, and
    # scanning both over the same line counted every test twice.
    for m in MATLAB_COUNT.finditer(text):
        out["passed"] = out.get("passed", 0) + int(m.group(1))
        out["failed"] = out.get("failed", 0) + int(m.group(2))
    text = MATLAB_COUNT.sub("", text)
    for n, kind in PYTEST_COUNT.findall(text):
        out[kind] = out.get(kind, 0) + int(n)
    comparisons = COMPARISON.findall(text)
    if comparisons:
        out["passed"] = out.get("passed", 0) + comparisons.count("PASS")
        out["failed"] = out.get("failed", 0) + comparisons.count("FAIL")
    return out


def run_stage(name: str, engine: str) -> tuple[int, dict, float]:
    """Run one stage, streaming its output; return code, counts and seconds."""
    print(f"\n########## make {name} ENGINE={engine}", flush=True)
    started = time.perf_counter()
    chunks = []
    proc = subprocess.Popen(
        ["make", name, f"ENGINE={engine}"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        chunks.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()
    code = proc.wait()
    return code, counts_from("".join(chunks)), time.perf_counter() - started


def format_counts(counts: dict) -> str:
    """One-line rendering of a stage's counts, in a fixed order."""
    if not counts:
        return "-"
    order = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
    return ", ".join(f"{counts[k]} {k}" for k in order if counts.get(k))


def main() -> int:
    """Run the stages and print the tally."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--engine", default="python", choices=("python", "matlab", "octave")
    )
    ap.add_argument(
        "--stage",
        action="append",
        choices=STAGES,
        help="run only this stage (repeatable); default is all of them",
    )
    args = ap.parse_args()
    stages = tuple(args.stage) if args.stage else STAGES

    results = []
    for name in stages:
        code, counts, secs = run_stage(name, args.engine)
        results.append((name, code, counts, secs))

    total: dict[str, int] = {}
    for _, _, counts, _ in results:
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v

    width = max(len(n) for n in stages)
    print(f"\n{'=' * (width + 46)}")
    print(f"Test summary  ENGINE={args.engine}")
    print(f"{'=' * (width + 46)}")
    for name, code, counts, secs in results:
        status = "ok" if code == 0 else f"FAILED ({code})"
        print(f"  {name:<{width}}  {status:<14} {secs:7.1f}s  {format_counts(counts)}")
    print(f"{'-' * (width + 46)}")
    failed_stages = [n for n, c, _, _ in results if c != 0]
    print(
        f"  {'TOTAL':<{width}}  {'':<14} {sum(r[3] for r in results):7.1f}s  "
        f"{format_counts(total) if total else '-'}"
    )
    if failed_stages:
        print(
            f"\n{len(failed_stages)} of {len(results)} stages FAILED: "
            f"{', '.join(failed_stages)}"
        )
        return 1
    print(f"\nall {len(results)} stages ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

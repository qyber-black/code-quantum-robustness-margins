# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The test summary must count what the runners actually printed.

A tally that silently reads zero from output it does not recognise is worse
than no tally: it turns a stage nobody ran into a stage that passed. These
pin the parsing against the real output of each runner in the tree.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _runner():
    """Load scripts/_test_suite.py as a module, by path."""
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(
        "_suite", ROOT / "scripts/_test_suite.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_suite"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_reads_a_pytest_summary_line():
    """pytest's own wording, including the mixed-outcome form."""
    counts = _runner().counts_from("2 failed, 241 passed, 1 skipped in 12.34s")
    assert counts == {"failed": 2, "passed": 241, "skipped": 1}


def test_reads_the_matlab_tally():
    """The line run_all_tests prints at the end."""
    counts = _runner().counts_from("\n17 passed, 2 failed of 19 tests\n")
    assert counts == {"passed": 17, "failed": 2}


def test_counts_each_cross_engine_comparison():
    """Each compare_* report ends in one overall= line."""
    text = "fid: fail=0\noverall=PASS\nM_H0: fail=3\noverall=FAIL\noverall=PASS\n"
    counts = _runner().counts_from(text)
    assert counts == {"passed": 2, "failed": 1}


def test_reports_nothing_rather_than_zero_when_it_recognises_nothing():
    """The distinction that matters: no counts is not the same as no tests.

    A stage whose output the parser does not recognise must come out empty,
    so the summary shows a dash, rather than contributing a confident zero.
    """
    assert _runner().counts_from("make: Nothing to be done for 'test-lint'.") == {}


def test_formats_counts_in_a_fixed_order():
    """Failures are read at a glance, so they sit next to the passes."""
    fmt = _runner().format_counts
    assert (
        fmt({"skipped": 1, "failed": 2, "passed": 3}) == "3 passed, 2 failed, 1 skipped"
    )
    assert fmt({}) == "-"

# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A reproduction reruns every driver that writes into the trees it compares.

`compare_trees` walks every CSV in a committed tree and demands the
recomputation produced it, so a tree is reproduced whole. Result trees are
named after the method rather than the paper, so one tree can hold outputs
both papers publish; a driver missing from a paper's list makes that
paper's reproduction fail on a file nothing it ran could have written.

The failure is invisible until someone runs the full reproduction, which
takes hours, so the agreement is asserted here instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from _paper import DRIVERS  # noqa: E402
from check_reproducible import PAPERS  # noqa: E402


def _tree_csvs(tree: str):
    """Committed CSV names in one result tree."""
    return {f.name for f in (ROOT / "results" / tree).glob("*.csv")}


def test_papers_are_declared():
    """Guard against the specification table being read as empty."""
    assert set(PAPERS) == {"qrm", "xqrm"}


def test_every_tree_csv_has_a_driver_the_paper_reruns():
    """Every file a reproduction will demand is one it can produce."""
    gaps = {}
    for paper, spec in PAPERS.items():
        listed = {script for script, _tree in spec["drivers"]}
        for tree in spec["trees"]:
            for name in sorted(_tree_csvs(tree)):
                driver = DRIVERS.get(name)
                if driver is None or driver not in listed:
                    gaps.setdefault(paper, []).append((name, driver or "unmapped"))
    assert not gaps, (
        "a reproduction would demand files no driver it reruns produces: "
        f"{gaps}. Add the driver to that paper's list in check_reproducible, "
        "or record the file in _paper.DRIVERS."
    )


def test_multi_invocation_drivers_run_every_invocation():
    """A driver with several recorded invocations must get them all.

    Without `driver_runs` the recomputation runs a driver once with no
    flags, so a driver whose outputs differ by flag silently produces one
    of them.
    """
    for paper, spec in PAPERS.items():
        assert "driver_runs" in spec, (
            f"{paper} has no driver_runs, so a multi-invocation driver would "
            "run once and its other outputs would be reported missing"
        )

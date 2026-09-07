# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A paper stage asks for files; it does not run a driver.

`paper-PAPER-EXPNAME` is a phony target, so its recipe runs every time it
is named. A recipe that invokes a driver there therefore recomputes on
every `make paper`, however current the results are -- which is how two
stages of paper 1 came to rerun a 61-controller case study and three
time-bandwidth comparisons on a tree that needed neither.

The fix is structural: the python branch delegates to a file set, and the
file rules decide. This asserts that shape, because timing the build would
not: a stale-looking recomputation is indistinguishable from a slow one.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: A paper stage: paper-QRM-margins, paper-xQRM-open. `paper-QRM` and
#: `paper-xQRM` are aggregates that only name other targets.
STAGE = re.compile(r"^(paper-[A-Za-z]+-[a-z0-9-]+):[^\n]*\n((?:\t[^\n]*\n)*)", re.M)

#: A Python driver invocation. The peers are invoked through MRUN/ORUN and
#: have no file rules, so they are not covered by this.
DRIVER = re.compile(r"\$\(XRUN\)/run_|scripts/run_[a-z0-9_]+\.py")

#: Where the python branch of a stage recipe starts.
PYTHON_BRANCH = 'test "$(ENGINE)" = python'


def _stages():
    text = (ROOT / "Makefile").read_text()
    found = STAGE.findall(text)
    assert found, "no paper-PAPER-EXPNAME targets found; did the naming change?"
    return found


def test_no_stage_runs_a_python_driver_in_its_own_recipe():
    """Every stage delegates, so an up-to-date tree costs nothing."""
    offenders = []
    for name, recipe in _stages():
        for line in recipe.splitlines():
            if "$(MRUN)" in line or "$(ORUN)" in line:
                continue  # a peer, which has no file rules
            if DRIVER.search(line):
                offenders.append(f"{name}: {line.strip()}")
    assert not offenders, (
        "these paper stages invoke a driver directly, so they recompute on "
        "every make paper:\n  " + "\n  ".join(offenders)
    )


def test_every_stage_delegates_to_a_file_set():
    """The delegation names a FILES_ variable, which is what make can date."""
    missing = []
    for name, recipe in _stages():
        if "$(MAKE) $(FILES_" not in recipe:
            missing.append(name)
    assert not missing, (
        "these paper stages never ask make for a file set: " + ", ".join(missing)
    )

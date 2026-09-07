# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every MATLAB package documents exactly the functions it contains.

``Contents.m`` is what ``help qrobustness`` prints, so it is the first
thing a reader of the peer meets. It is also hand-written and therefore
drifts: a function added without a line here is invisible, and a line left
behind after a rename points at nothing. Neither shows up in any other
check -- the package still loads, the tests still pass -- so the agreement
is asserted here.

The test lives on the Python side because it is the only suite CI runs.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "matlab" / "+qrobustness"

#: A listing line is two spaces, the name, then a dash and the summary.
ENTRY = re.compile(r"^%\s{4,}([a-zA-Z_]\w*)\s+-\s+\S", re.M)


def _packages():
    """The top-level package and every subpackage, as directories."""
    return [PACKAGE] + sorted(p for p in PACKAGE.iterdir() if p.name.startswith("+"))


def _functions(pkg: Path) -> set:
    """Function file names in one package directory."""
    return {f.stem for f in pkg.glob("*.m") if f.stem != "Contents"}


def _listed(pkg: Path) -> set:
    """Names in that package's Contents.m listing."""
    text = (pkg / "Contents.m").read_text()
    return {m.group(1) for m in ENTRY.finditer(text)}


def test_every_package_has_a_contents_file():
    """A package without Contents.m shows nothing for "help qrobustness"."""
    missing = [p.name for p in _packages() if not (p / "Contents.m").is_file()]
    assert not missing, f"packages without a Contents.m: {missing}"


def test_contents_lists_every_function():
    """A function absent from Contents.m is invisible to the reader."""
    gaps = {}
    for pkg in _packages():
        undocumented = _functions(pkg) - _listed(pkg)
        if undocumented:
            gaps[pkg.name] = sorted(undocumented)
    assert not gaps, f"functions missing from Contents.m: {gaps}"


def test_contents_lists_nothing_that_is_gone():
    """A line left behind after a rename points at a function that is gone."""
    stale = {}
    for pkg in _packages():
        gone = _listed(pkg) - _functions(pkg)
        if gone:
            stale[pkg.name] = sorted(gone)
    assert not stale, f"Contents.m names functions that do not exist: {stale}"

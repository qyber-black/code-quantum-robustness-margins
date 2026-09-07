# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every driver must at least parse, and the paper map must match the Makefile.

No test imported any of the 33 scripts -- 7500 lines including all 21
experiment drivers. A multi-line `assert` rewritten by a line-oriented
regex left one of them unparseable, 204 tests still passed, and the fault
surfaced only when a six-hour regeneration died on it. Compiling every
script costs milliseconds and closes that gap.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))

sys.path.insert(0, str(ROOT / "scripts"))


def test_there_are_scripts_to_check():
    """Guard against the glob silently matching nothing."""
    assert len(SCRIPTS) > 20, f"expected the driver set, found {len(SCRIPTS)}"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_parses(path: Path):
    """A syntax error must fail here, not in a six-hour run."""
    ast.parse(path.read_text(), filename=str(path))


def _makefile_owners() -> dict:
    """Which driver each results file is built by, per the Makefile rules."""
    mk = (ROOT / "Makefile").read_text()
    trees = dict(re.findall(r"^([A-Z_]+) := \$\(R\)/([a-z0-9-]+)$", mk, re.M))
    owners = {}
    rule = re.compile(
        r"^((?:\$\([A-Z_]+\)/[^\s:]+[ \\\n\t]*)+)&?:[^\n]*\n"
        r"(?:[^\n]*\n)*?\t[^\n]*?(?:XRUN\)/|scripts/)(run_[a-z_]+\.py)",
        re.M,
    )
    for m in rule.finditer(mk):
        for var, name in re.findall(r"\$\(([A-Z_]+)\)/([^\s\\]+)", m.group(1)):
            if var in trees:
                owners[name] = m.group(2)
    return owners


def test_paper_driver_map_matches_the_makefile():
    """_paper.DRIVERS must name the driver that actually writes each file.

    Keyed by output file, not by results tree: several trees are written
    by more than one driver, so the advice must name the right one.
    """
    from _paper import DRIVERS

    owners = _makefile_owners()
    assert owners, "could not read any output-to-driver mapping from the Makefile"
    wrong = {
        name: (DRIVERS.get(name), driver)
        for name, driver in owners.items()
        if DRIVERS.get(name) != driver
    }
    assert not wrong, (
        "scripts/_paper.py DRIVERS disagrees with the Makefile "
        f"(file: mapped -> actual): {wrong}"
    )

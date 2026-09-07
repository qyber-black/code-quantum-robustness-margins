# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Documentation names targets that exist.

`make <name>` in the README, in docs/, or in the advice a test prints when
it fails is an instruction a reader will follow. A renamed target leaves
that instruction pointing at nothing, and nothing else notices: the
documentation still builds, the tests still pass, and the reader finds out.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: A target definition at column 0.
DEFINITION = re.compile(r"^([a-z][a-zA-Z0-9_-]*):", re.M)
REFERENCE = re.compile(r"\bmake ([A-Za-z][A-Za-z0-9-]*)\b")

#: Code formatting in Markdown: fenced blocks and inline spans. A command a
#: reader is meant to type is written as code, so looking only there keeps
#: ordinary prose ("make the tolerance smaller") out of the result.
CODE_SPAN = re.compile(r"```.*?```|`[^`\n]+`", re.S)

#: Files whose `make` advice a reader acts on.
DOCUMENTED = ("README.md", "docs/*.md", "python/tests/*.py", "matlab/tests/*.m")

#: Metavariables that stand for a family rather than naming a target:
#: `make paper-PAPER`, `make test-TESTNAME`. A name containing one is a
#: pattern, and the reader substitutes before running it.
METAVARIABLES = ("PAPER", "ENGINE", "TESTNAME", "NAME", "ID")


def _targets() -> set:
    """Every target the Makefile defines."""
    return set(DEFINITION.findall((ROOT / "Makefile").read_text()))


def _referenced() -> dict:
    """Target name -> the files that tell a reader to run it.

    Markdown is read inside code formatting only. Source files are read
    whole, since their advice sits in string literals, but only hyphenated
    names are taken there: a bare word after "make" in a sentence is
    English, not a target.
    """
    out: dict[str, set] = {}
    for pattern in DOCUMENTED:
        for f in sorted(ROOT.glob(pattern)):
            if f.name == Path(__file__).name:
                continue
            text = f.read_text()
            if f.suffix == ".md":
                names = [
                    n
                    for span in CODE_SPAN.findall(text)
                    for n in REFERENCE.findall(span)
                ]
            else:
                names = [n for n in REFERENCE.findall(text) if "-" in n]
            names = [n.rstrip("-") for n in names]
            for name in names:
                if any(v in name for v in METAVARIABLES):
                    continue
                out.setdefault(name, set()).add(str(f.relative_to(ROOT)))
    return out


def test_the_makefile_defines_targets():
    """Guard against the definition pattern silently matching nothing."""
    assert len(_targets()) > 20, "could not read the target list from the Makefile"


def test_every_documented_target_exists():
    """A documented `make` command must be one a reader can actually run."""
    targets = _targets()
    missing = {
        name: sorted(files)
        for name, files in _referenced().items()
        if name not in targets
    }
    assert not missing, f"documented targets that do not exist: {missing}"

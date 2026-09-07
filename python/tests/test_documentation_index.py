# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every document is reachable, and every link resolves.

A file in `docs/` that nothing links to is documentation nobody finds, and a
link to a file that does not exist is worse: the reader concludes the answer
was never written.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: A relative Markdown link target, without an anchor or a URL scheme.
LINK = re.compile(r"\]\((?!https?:|mailto:)([^)#\s]+)(?:#[^)\s]*)?\)")


def _markdown():
    """The documentation set: the README and everything in docs/."""
    return [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def test_every_document_is_linked():
    """A document nothing links to cannot be found from the entry point."""
    text = "".join(f.read_text() for f in _markdown())
    linked = set(re.findall(r"docs/([a-z0-9-]+\.md)", text))
    orphans = sorted({f.name for f in (ROOT / "docs").glob("*.md")} - linked)
    assert not orphans, f"documents linked from nowhere: {orphans}"


def test_every_relative_link_resolves():
    """A link to a path that does not exist reads as a missing answer."""
    broken = {}
    for f in _markdown():
        for target in LINK.findall(f.read_text()):
            if not (f.parent / target).exists():
                broken.setdefault(str(f.relative_to(ROOT)), []).append(target)
    assert not broken, f"links that do not resolve: {broken}"

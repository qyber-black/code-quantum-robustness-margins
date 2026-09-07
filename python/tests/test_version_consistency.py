# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Repository metadata that nothing derives, so nothing checks: the version
declarations, and the licence header on every source file.

The version is declared in six places; they must all agree.

There is no single source for it: packaging metadata, the citation file and
the Zenodo record each need the number in their own format, and the README
quotes it twice in prose and BibTeX. Nothing derives one from another, so a
bump that misses a file leaves the repository claiming two versions at once
-- which is what a released artefact must never do.

Each file is also required to exist. A missing one is a failure, not a skip:
the citation and Zenodo records are what a DOI is minted from, and a test
that quietly passes when they are absent would defeat the purpose.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def _pyproject() -> str:
    text = (ROOT / "python/pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert m, "python/pyproject.toml declares no version"
    return m.group(1)


def _dunder() -> str:
    text = (ROOT / "python/src/qrobustness/__init__.py").read_text()
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M)
    assert m, "__init__.py declares no __version__"
    return m.group(1)


def _zenodo() -> str:
    return json.loads((ROOT / ".zenodo.json").read_text())["version"]


def _citation() -> str:
    text = (ROOT / "CITATION.cff").read_text()
    m = re.search(r"^version:\s*(\S+)", text, re.M)
    assert m, "CITATION.cff declares no version"
    return m.group(1)


def _readme() -> list:
    text = (ROOT / "README.md").read_text()
    prose = re.search(r"Version (\d+\.\d+\.\d+)\. Software", text)
    bibtex = re.search(r"version = \{(\d+\.\d+\.\d+)\}", text)
    assert prose, "README.md quotes no version in prose"
    assert bibtex, "README.md quotes no version in BibTeX"
    return [prose.group(1), bibtex.group(1)]


SOURCES = {
    "python/pyproject.toml": _pyproject,
    "python/src/qrobustness/__init__.py": _dunder,
    ".zenodo.json": _zenodo,
    "CITATION.cff": _citation,
}


def test_every_version_bearing_file_is_present():
    """A missing citation or Zenodo record is a failure, never a skip."""
    for name in list(SOURCES) + ["README.md", "CHANGELOG.md"]:
        assert (ROOT / name).is_file(), f"{name} is missing from the repository"


def test_all_version_declarations_agree():
    """Every file that states a version states the same one, and it is a
    semantic version. Nothing derives these from each other, so agreement
    has to be asserted."""
    found = {name: fn() for name, fn in SOURCES.items()}
    for i, v in enumerate(_readme()):
        found[f"README.md[{i}]"] = v
    distinct = set(found.values())
    assert len(distinct) == 1, (
        f"version declarations disagree: {found}. Bump every one of them; "
        "nothing derives these from each other."
    )
    assert SEMVER.match(distinct.pop()), "version is not MAJOR.MINOR.PATCH"


def test_changelog_documents_the_declared_version():
    """A bump with no changelog entry is a release nobody can read."""
    version = _dunder()
    text = (ROOT / "CHANGELOG.md").read_text()
    assert f"## [{version}]" in text, (
        f"CHANGELOG.md has no '## [{version}]' section; the version was bumped "
        "without recording what changed."
    )


#: Every source file must carry the SPDX header REUSE compliance needs.
SOURCE_GLOBS = (
    ("python", "**/*.py"),
    ("scripts", "*.py"),
    ("matlab", "**/*.m"),
)
SPDX_TAG = "SPDX-License-Identifier"


def test_every_source_file_carries_the_licence_header():
    """A file without the SPDX tag breaks REUSE compliance silently.

    Nothing else notices: the file lints, imports and passes its own tests,
    and the omission surfaces only when someone runs the compliance tool or
    asks what licence a file is under.
    """
    missing = []
    for subdir, pattern in SOURCE_GLOBS:
        for f in sorted((ROOT / subdir).glob(pattern)):
            if "__pycache__" in f.parts or ".venv" in f.parts:
                continue
            if SPDX_TAG not in f.read_text():
                missing.append(str(f.relative_to(ROOT)))
    assert not missing, f"source files without an {SPDX_TAG} header: {missing}"

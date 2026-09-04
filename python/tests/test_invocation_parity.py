# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Makefile and check_reproducible must invoke drivers identically.

They diverged four times -- --adversary twice, BLAS pinning, and three
missing driver runs -- and each time a reproduction failed on a file that
was in fact correct. scripts/_invocations.py is the single source; these
tests fail if a recipe stops using it.

The first version of this file tested the wrong thing. It asserted that no
*literal* flag appeared in a recipe, which is only half the failure: the
divergence that actually happened twice is a recipe that simply stops
passing its flags. Deleting `$(FLAGS_run_robust_vs_nominal)` left all of
those tests green while make ran the driver bare. Every test below is
therefore checked against a deliberately reintroduced fault in
``test_the_guards_catch_a_reintroduced_divergence``; a guard that has not
been shown to fail is not a guard.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from _invocations import DRIVER_RUNS, SERIAL_BLAS_ENV  # noqa: E402

MAKEFILE = (ROOT / "Makefile").read_text()

#: Supplied per call site, not per experiment. --jobs and --out say where
#: to run and where to write; --controller-dir and --max-error say which
#: ensemble, and check-xQRM-synth uses them to point the certificate
#: harness at a freshly synthesised one. None of them selects an
#: experiment, so none belongs in the shared table.
CONTEXT_FLAGS = {"--jobs", "--out", "--controller-dir", "--max-error"}

#: Recipes that are not part of the paper pipeline: synthesis smoke and the
#: paper-1 sweep, whose arguments are properties of those targets alone.
EXEMPT = {"run_synthesize_controllers.py", "run_lipschitz_margin_case_study.py"}

#: A flag is a double dash then a letter, then letters, digits or dashes.
#: The earlier `--[a-z-]+` missed --FT and truncated --n2-starts to --n.
FLAG = re.compile(r"--[A-Za-z][A-Za-z0-9-]*")


def _invocations(text: str) -> list[tuple[str, str]]:
    """(driver, argument text) for every driver invocation in a recipe.

    Covers both spellings: the $(XRUN)/ shorthand and the expanded
    $(PYTHON) $(ROOT)/scripts/ form, which the earlier version missed
    entirely, so rewriting a recipe that way was a silent bypass.
    """
    pat = re.compile(
        r"(?:\$\(XRUN\)/|\$\(ROOT\)/scripts/)(run_[a-z_]+\.py)((?:[^\n\\]|\\\n)*)"
    )
    return [(m.group(1), m.group(2)) for m in pat.finditer(text)]


def _flags_vars(args: str) -> list[str]:
    """FLAGS_ variables referenced in one invocation's argument text."""
    return re.findall(r"\$\((FLAGS_[A-Za-z0-9_]+)\)", args)


def test_no_literal_production_flags_in_recipes():
    """Every production flag must come from the shared table."""
    bad = {}
    for script, args in _invocations(MAKEFILE):
        if script in EXEMPT:
            continue
        lits = set(FLAG.findall(args)) - CONTEXT_FLAGS
        if lits:
            bad[script] = sorted(lits)
    assert bad == {}, (
        f"literal flags in Makefile recipes: {bad}; "
        "add them to scripts/_invocations.py instead"
    )


def test_each_driver_recipe_references_its_flags_variable():
    """The failure that actually happened: a recipe stops passing its flags.

    For a driver with N argument lists the Makefile must reference
    FLAGS_<stem> and FLAGS_<stem>_1 .. _<N-1>, each at least once.
    """
    referenced = {}
    for script, args in _invocations(MAKEFILE):
        referenced.setdefault(script, set()).update(_flags_vars(args))
    for script, runs in DRIVER_RUNS.items():
        stem = script[:-3]
        want = {f"FLAGS_{stem}"} | {f"FLAGS_{stem}_{i}" for i in range(1, len(runs))}
        got = referenced.get(script, set())
        assert want <= got, (
            f"{script}: recipes reference {sorted(got) or 'nothing'}, need "
            f"{sorted(want)}. A missing reference runs the driver without its "
            "flags, which is how --adversary was lost twice."
        )


def test_makefile_never_overrides_the_blas_environment():
    """Any assignment anywhere wins if it comes after the include."""
    for var in SERIAL_BLAS_ENV:
        for m in re.finditer(rf"^\s*{var}\s*[:?+]?=", MAKEFILE, re.M):
            line = MAKEFILE[: m.start()].count("\n") + 1
            raise AssertionError(
                f"{var} assigned at Makefile:{line}; the thread environment "
                "must come from SERIAL_BLAS_ENV via the generated fragment"
            )


def test_every_table_driver_is_invoked_by_a_recipe():
    """A table entry nothing runs is a silent no-op."""
    invoked = {s for s, _ in _invocations(MAKEFILE)}
    missing = set(DRIVER_RUNS) - invoked
    assert not missing, f"in DRIVER_RUNS but no recipe runs them: {sorted(missing)}"


def test_the_guards_catch_a_reintroduced_divergence():
    """Each guard above must fail on the fault it exists for.

    Without this the guards are untested assertions about untested code,
    which is precisely how the first version of this file came to pass on
    the bug it was written to prevent.
    """
    import pytest

    global MAKEFILE
    original = MAKEFILE
    faults = {
        # The historical --adversary loss: the reference simply goes.
        "dropped flags reference": (
            "$(XRUN)/run_robust_vs_nominal.py $(FLAGS_run_robust_vs_nominal)",
            "$(XRUN)/run_robust_vs_nominal.py",
            test_each_driver_recipe_references_its_flags_variable,
        ),
        # A literal flag creeping back in.
        "literal flag": (
            "$(XRUN)/run_budget_sweep.py",
            "$(XRUN)/run_budget_sweep.py --absorption additive",
            test_no_literal_production_flags_in_recipes,
        ),
        # An override placed after the include, which is the one that wins.
        "late BLAS override": (
            "# --- Tests ---",
            "OMP_NUM_THREADS := 8\n\n# --- Tests ---",
            test_makefile_never_overrides_the_blas_environment,
        ),
        # Pointing a later run at index 0, so its outputs never regenerate.
        "collapsed run index": (
            "$(FLAGS_run_time_bandwidth_bound_comparison_2)",
            "$(FLAGS_run_time_bandwidth_bound_comparison)",
            test_each_driver_recipe_references_its_flags_variable,
        ),
    }
    try:
        for name, (old, new, guard) in faults.items():
            assert original.count(old) >= 1, f"{name}: anchor not found"
            MAKEFILE = original.replace(old, new, 1)
            with pytest.raises(AssertionError):
                guard()
    finally:
        MAKEFILE = original

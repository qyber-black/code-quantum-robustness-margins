#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The production flags for every driver, defined once.

Two places need them: the Makefile recipe that produces the committed
results, and check_reproducible, which recomputes into a scratch tree and
compares. Written twice they drift, and the symptom is a reproduction that
cannot pass because the recomputed tree lacks a file the committed tree
has.

The table therefore lives here and both read it. The Makefile pulls it in
as a generated fragment (``--make``); check_reproducible imports
DRIVER_RUNS.
Adding an invocation in one place adds it in both.

A driver absent from this table takes no production flags. --out and
--jobs are deliberately excluded: the caller supplies those, the Makefile
from JOBS and check_reproducible from its scratch directory and its own
worker count.
"""

from __future__ import annotations

import sys

#: The BLAS thread environment every driver runs under. Not a
#: micro-optimisation: these drivers work on 8x8 to 64x64 matrices, far
#: below the size at which threading pays, so an unpinned pool only adds
#: spin-wait. Measured on the 61-controller sweeps, one process, no other
#: load:
#:
#:   run_open_system_case_study   3h30m unpinned  ->  1m17s pinned
#:   run_open_amplitude_damping   3h52m unpinned  ->  4m53s pinned
#:
#: It lives here, with the flags, because the Makefile had it and
#: check_reproducible did not -- the fourth time those two diverged. The
#: reproduction consequently spent over half an hour on a driver that
#: takes five minutes.
#:
#: Pinning changes BLAS reduction order, so results move in their last
#: bits: max relative difference 5.6e-14, six orders below the 1e-8 the
#: comparison allows.
SERIAL_BLAS_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

#: driver -> the argument lists it must be run with, in order. More than
#: one list means the driver runs more than once, each writing different
#: outputs.
DRIVER_RUNS: dict[str, list[list[str]]] = {
    "run_lipschitz_margin_case_study.py": [
        # --sweep adds the fidelity-vs-delta computation and the H*_all.png
        # plots, which sync-QRM publishes into the paper. The CSVs come out
        # either way, so a reproduction run without it compared clean while
        # the figures the paper shows were never regenerated.
        #
        # Python only. The peers call the same option do_sweep and default
        # it to true, so their recipes take no flag and the generated
        # MFLAGS for this driver is deliberately unused; wiring it into a
        # peer recipe would pass a name they do not have.
        ["--sweep"],
    ],
    "run_multiparameter_case_study.py": [
        # tv_bracket_<FT>.csv is written only for the first --adversary
        # controllers; five is what the paper probes.
        ["--step", "lipschitz", "--adversary", "5"],
        ["--step", "angular"],
    ],
    "run_kosut_validity.py": [
        # The documented violation is expected, so it must not fail the run.
        ["--n-starts", "12", "--starts", "mixed", "--allow-violations"],
        # The trajectory class writes validity_<FT>_tv.csv, which the
        # Makefile never mentioned at all.
        [
            "--n-starts",
            "12",
            "--starts",
            "mixed",
            "--allow-violations",
            "--uncertainty",
            "trajectory",
        ],
    ],
    "run_time_bandwidth_bound_comparison.py": [
        # Three files, three runs. The defaults give the angular constant
        # comparison; the additive one feeds the absorption macros, and the
        # trajectory one feeds tab_tv. The grouped Make target named all
        # three but the recipe ran the driver once, so two were never
        # regenerated and sat on whatever values they last had.
        [],
        ["--absorption", "additive"],
        ["--uncertainty", "trajectory"],
    ],
    "run_robust_vs_nominal.py": [
        # Without this the adversarial witness columns are absent and
        # tab_robust silently loses two rows.
        ["--adversary"],
    ],
}


def flags(script: str, index: int = 0) -> list[str]:
    """Production flags for one invocation of ``script``.

    Raises rather than returning [] for an out-of-range index: silently
    treating "run 7 of a 2-run driver" as "this driver takes no flags"
    would drop arguments in a call site that then succeeds with the wrong
    ones, which is the failure this table exists to prevent.
    """
    runs = DRIVER_RUNS.get(script, [[]])
    if not 0 <= index < len(runs):
        raise IndexError(f"{script} has {len(runs)} invocation(s); no index {index}")
    return runs[index]


def _emit_make() -> None:
    """Emit a Makefile fragment: SERIAL_BLAS, FLAGS_<stem>, FLAGS_<stem>_<n>."""
    print("# Generated by scripts/_invocations.py -- do not edit.")
    print("# Regenerated whenever that file changes; see the Makefile rule.")
    env = " ".join(f"{k}={v}" for k, v in SERIAL_BLAS_ENV.items())
    print(f"SERIAL_BLAS := {env}")
    for script, runs in sorted(DRIVER_RUNS.items()):
        if not script.endswith(".py"):
            raise ValueError(f"table key {script!r} is not a .py filename")
        stem = script[:-3]
        for i, run in enumerate(runs):
            name = f"FLAGS_{stem}" if i == 0 else f"FLAGS_{stem}_{i}"
            for tok in run:
                # The fragment is interpolated unquoted into recipes, so a
                # '#' would truncate the variable, a '$' would be expanded
                # by make, and a space would split into two shell words --
                # each giving a successful run with the wrong arguments.
                if any(c in tok for c in "#$ \t\n"):
                    raise ValueError(
                        f"{script}: argument {tok!r} needs quoting that this "
                        "fragment cannot express; keep flags simple"
                    )
            print(f"{name} := {' '.join(run)}")
            # The MATLAB/Octave peers take the same production settings as
            # name/value pairs rather than CLI flags. Emitting them from this
            # one table keeps the peer recipes from drifting the way the
            # Python ones did; without it the peers were the only invocations
            # with hand-written flags and nothing checked them.
            print(f"M{name} := {_matlab_args(run)}")


def _matlab_args(run: list[str]) -> str:
    """Render a Python flag list as MATLAB name/value pairs.

    ``["--absorption", "additive"]`` becomes ``,'absorption','additive'``;
    a valueless flag becomes ``,'flag',true``. The leading comma lets a
    recipe append the result directly after its fixed arguments, and an
    empty run renders as the empty string.
    """
    out: list[str] = []
    i = 0
    while i < len(run):
        tok = run[i]
        if not tok.startswith("--"):
            raise ValueError(f"expected a --flag, got {tok!r}")
        name = tok[2:].replace("-", "_")
        if i + 1 < len(run) and not run[i + 1].startswith("--"):
            out.append(f"'{name}','{run[i + 1]}'")
            i += 2
        else:
            out.append(f"'{name}',true")
            i += 1
    return ("," + ",".join(out)) if out else ""


if __name__ == "__main__":
    if "--make" in sys.argv:
        _emit_make()
    else:
        script = sys.argv[1]
        index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        print(" ".join(flags(script, index)))

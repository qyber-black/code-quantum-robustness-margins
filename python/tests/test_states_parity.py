# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The committed parity fixture of the state and open-system state
certificates (matlab/tests/fixtures/states_parity.json, read by the MATLAB and
Octave peers) is what scripts/gen_states_parity.py produces now: a
change of the Python reference that is not regenerated would otherwise leave
the peers compared against stale numbers."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "matlab" / "tests" / "fixtures" / "states_parity.json"


def _close(a, b, path=""):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _close(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, str):
        assert a == b, path
    else:
        assert np.isclose(a, b, rtol=1e-9, atol=1e-12), f"{path}: {a} != {b}"


def test_fixture_is_current(tmp_path):
    out = tmp_path / "states_parity.json"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_states_parity.py"), str(out)],
        check=True,
        capture_output=True,
    )
    _close(json.loads(FIXTURE.read_text()), json.loads(out.read_text()))

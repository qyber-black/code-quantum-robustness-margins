#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write the parity fixture of the state and open-system state certificates.

The JSON holds fixed random inputs (d, dt, H, A, B, psi0) and the Python
reference outputs: under "states" the state speed C, C_joint, the
Fubini-Study angle, F_T, the state margin, and the preparation quantities;
under "openstates" the open speed D, the trace distance and the Hamiltonian
diamond norms. The MATLAB/Octave tests test_states and test_openstates
recompute and compare them; python/tests/test_states_parity.py checks the
committed fixture against this script.

Usage: python scripts/gen_states_parity.py [output.json]
"""

import json
import sys
from pathlib import Path

import numpy as np

from qrobustness import lindblad as lb
from qrobustness import openstates as os_
from qrobustness import states as st

OUT = (
    Path(__file__).resolve().parents[1]
    / "matlab"
    / "tests"
    / "fixtures"
    / "states_parity.json"
)


def _herm(rng, d, scale):
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    return scale * (a + a.conj().T) / 2


def _c(x):
    """Complex array as {re, im} nested lists (row-major)."""
    x = np.asarray(x)
    return {"re": np.real(x).tolist(), "im": np.imag(x).tolist()}


def _mlist(ms):
    return [_c(m) for m in ms]


def build():
    rng = np.random.default_rng(20261003)
    d, K, dt = 3, 3, 0.4
    H = [_herm(rng, d, 1.0) for _ in range(K)]
    A = [_herm(rng, d, 0.6) for _ in range(K)]
    B = [_herm(rng, d, 0.5) for _ in range(K)]
    psi0 = rng.normal(size=d) + 1j * rng.normal(size=d)
    psi0 /= np.linalg.norm(psi0)

    def state(s, mu):
        return st.propagate_state(
            [h + s * a + mu * b for h, a, b in zip(H, A, B, strict=True)], dt, psi0
        )

    C = st.state_speed(B, dt)
    target = state(0.0, 0.05)

    def fid(mu):
        return st.state_fidelity(target, state(0.0, mu))

    FT = 0.6 * fid(0.0)
    sm = st.state_angular_margin(fid, C, FT, k_max=20000)
    psi_p, E_p, gap_p = st.nondegenerate_eigenvector(H[0] + 0.2 * A[0], 0)
    states = {
        "C": C,
        "C_joint": st.state_speed_joint([B, A], dt, [0.7, -0.4]),
        "fs_angle": st.fs_angle(state(0.0, 0.0), state(0.3, 0.2)),
        "FT": FT,
        "state_margin": [sm.M_minus, sm.M_plus],
        "prep_E": E_p,
        "prep_gap": gap_p,
        "prep_sigma": st.preparation_speed(A[0], psi_p, gap_p),
        "prep_spread": st.preparation_speed(A[0], psi_p, gap_p, bound="spread"),
        "prep_radius": st.gapped_preparation_radius(
            0.1, gap_p, st.half_spread(A[0]), C
        ),
    }

    SZ = np.diag([1.0, -1.0, 0.0]).astype(complex)
    LOW = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=complex)
    rho0 = np.outer(psi0, psi0.conj())
    # Fixed constant, so parity does not depend on the diamond-norm iteration.
    dnorm = 3.0

    def rho(s, mu):
        G = [
            lb.generator(h + s * a, [SZ, LOW], [0.3 + mu, 0.1])
            for h, a in zip(H, A, strict=True)
        ]
        return os_.evolve_density(G, dt, rho0)

    openstates = {
        "dnorm": dnorm,
        "D": os_.open_speed([lb.dissipator(SZ)] * K, dt, norm=lambda G: dnorm),
        "trace_distance": os_.trace_distance(rho(0.0, 0.0), rho(0.5, 0.2)),
        "ham_dnorm": lb.hamiltonian_dnorm(lb.hamiltonian_superop(B[0])).value_certified,
        "D_exact": os_.open_speed(
            [lb.hamiltonian_superop(b) for b in B], dt, exact_hamiltonian=True
        ),
    }

    return {
        "inputs": {
            "d": d,
            "dt": dt,
            "H": _mlist(H),
            "A": _mlist(A),
            "B": _mlist(B),
            "psi0": _c(psi0),
        },
        "states": states,
        "openstates": openstates,
    }


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""State-target examples from the xQRM paper (State targets, examples).

Computes state-target margins on small examples with an analytic or
independently computed reference. Options: --FT, --out. Writes
results/state-examples-python/:

ghz_detuning_<FT>.csv
    GHZ preparation on n = 1..6 qubits (R_y(pi/2) and a CNOT ladder, unit
    intervals) and a hold of HOLD intervals, collective detuning sum_q Z_q / 2.
    n: qubits.
    C_state_hold, C_choi_hold: state speed and Choi speed of the hold.
    r_state_hold, crossing_state_hold, f_at_r_state: one-step state radius,
        analytic crossing 2 arccos(f_T) / (n HOLD), fidelity at the radius.
    r_gate_hold, crossing_gate_hold: gate (Choi) radius and analytic gate
        crossing of the hold.
    C_state_all, r_state_all, M_all, M_upper_all, n_evals_all: detuning over
        preparation and hold: state speed, one-step radius, iterated bracket,
        evaluations.
tfim_preparation_<FT>.csv
    Ground state of the open transverse-field Ising chain, L = 2..4 sites,
    fields h0 in TFIM_FIELDS, relative field error h = h0 (1 + mu).
    L, h0, gap: chain, field and spectral gap.
    C_prep_fd, C_prep_sigma, C_prep_spread: C_prep by finite differences, and
        its sigma/gap and half-spread/gap bounds.
    r_one_step: one-step gapped-preparation radius.
    M_<dir>, evals_<dir>, crossing_lo_<dir>, crossing_hi_<dir>: re-centred
        continuation, its evaluations, and the resolved crossing bracket, for
        dir in minus, plus.
ghz_dephasing_<FT>.csv
    GHZ hold under common-rate local dephasing, state fidelity
    (1 + exp(-2 n gamma HOLD)) / 2 (checked against propagation for n <= 4).
    n, D_half: qubits and D/2 = n HOLD.
    r_one_step, M, M_upper, n_evals: one-step radius, iterated bracket,
        evaluations.
    crossing: analytic crossing.
closed_limit.csv
    Structures H0, H1, H2 of the main ensemble as Hamiltonian superoperators.
    structure, spread: structure and its eigenvalue spread.
    closed_form_certified, sdp_raw, sdp_certified, sdp_solver,
        free_certified: verified closed form, SDP optimum, certified SDP
        value, solver, and the solver-free bound.
state_variance_<FT>.csv
    Main ensemble, per controller, structure and initial state (|100>, |+>).
    controller, structure, state: instance.
    C_state, C_choi, int_sigma: state-speed constant (half spread), Choi
        speed, and int sigma_t dt along the nominal evolution.
"""

from __future__ import annotations

from functools import reduce

import numpy as np

from qrobustness import lindblad as lb
from qrobustness import openstates as os_
from qrobustness import states as st

from _drivers import (
    DEFAULT_ETA,
    PAULI_X,
    PAULI_Z,
    ROOT,
    base_parser,
    load_ensemble,
    write_rows,
)

OUT_DIR = ROOT / "results/state-examples-python"
PAULI_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
EYE2 = np.eye(2, dtype=complex)
HOLD = 4  # unit-length hold intervals after the preparation
N_MAX = 6
TFIM_SITES = (2, 3, 4)
TFIM_FIELDS = (1.0, 0.5, 0.25)
SUBSTEPS = 64  # quadrature points per interval for int sigma_t dt
MARGIN_TOL = 1e-8


def _choi_speed(B_list, dt):
    """Choi speed dt sum_k ||traceless(B_k)||_F / sqrt(N)."""

    N = B_list[0].shape[0]
    return float(
        dt
        * sum(np.linalg.norm(B - np.trace(B) / N * np.eye(N)) for B in B_list)
        / np.sqrt(N)
    )


def _op(single, q, n):
    return reduce(np.kron, [single if i == q else EYE2 for i in range(n)])


def _ghz_problem(n):
    """Nominal interval Hamiltonians (unit intervals) and the detuning."""
    Hs = [np.pi / 4 * _op(PAULI_Y, 0, n)]
    eye = np.eye(2**n)
    for q in range(n - 1):
        Hs.append(
            np.pi / 4 * (eye - _op(PAULI_Z, q, n)) @ (eye - _op(PAULI_X, q + 1, n))
        )
    hold = [np.zeros((2**n, 2**n), dtype=complex)] * HOLD
    det = sum(_op(PAULI_Z, q, n) for q in range(n)) / 2
    ghz = np.zeros(2**n, dtype=complex)
    ghz[0] = ghz[-1] = 1 / np.sqrt(2)
    return Hs, hold, det, ghz


def ghz_detuning(fT):
    rows = []
    for n in range(1, N_MAX + 1):
        prep, hold, det, ghz = _ghz_problem(n)
        zero = np.zeros(2**n, dtype=complex)
        zero[0] = 1.0
        assert st.state_fidelity(ghz, st.propagate_state(prep, 1.0, zero)) > 1 - 1e-12
        # hold only: the structure commutes, the bound is attained
        C_hold = st.state_speed([det] * HOLD, 1.0)
        r_state = (np.arccos(fT)) / C_hold
        exact_state = 2 * np.arccos(fT) / (n * HOLD)
        s_gate = _choi_speed([det] * HOLD, 1.0)
        r_gate = np.arccos(fT) / s_gate
        exact_gate = 2 * np.arccos(fT ** (1.0 / n)) / HOLD

        def f_hold(mu, det=det, ghz=ghz):
            return st.state_fidelity(
                ghz, st.propagate_state([mu * det] * HOLD, 1.0, ghz)
            )

        # preparation and hold together, detuning throughout
        allH = prep + hold
        C_all = st.state_speed([det] * len(allH), 1.0)

        def f_all(mu, allH=allH, det=det, ghz=ghz, zero=zero):
            return st.state_fidelity(
                ghz, st.propagate_state([h + mu * det for h in allH], 1.0, zero)
            )

        m = st.state_angular_margin(
            f_all,
            C_all,
            fT,
            margin_tol=MARGIN_TOL,
            return_diagnostics=True,
            eta=DEFAULT_ETA,
        )
        rows.append(
            {
                "n": n,
                "C_state_hold": C_hold,
                "C_choi_hold": s_gate,
                "r_state_hold": r_state,
                "crossing_state_hold": exact_state,
                "f_at_r_state": f_hold(r_state),
                "r_gate_hold": r_gate,
                "crossing_gate_hold": exact_gate,
                "C_state_all": C_all,
                "r_state_all": np.arccos(fT) / C_all,
                "M_all": m.M,
                "M_upper_all": m.M_upper,
                "n_evals_all": m.n_evals,
            }
        )
    return rows


def _tfim(L, h):
    zz = sum(_op(PAULI_Z, i, L) @ _op(PAULI_Z, i + 1, L) for i in range(L - 1))
    xs = sum(_op(PAULI_X, i, L) for i in range(L))
    return -zz - h * xs, -xs


def _prep_continuation(f, gap_at, s, fT, sign, k_max=100000):
    """Re-centred continuation with the gapped-preparation radius; returns
    the certified prefix and the number of fidelity evaluations."""
    mu, evals = 0.0, 0
    thT = np.arccos(fT)
    for _ in range(k_max):
        F = f(mu)
        evals += 1
        if F - fT < DEFAULT_ETA:
            break
        r = st.gapped_preparation_radius(thT - np.arccos(min(F, 1.0)), gap_at(mu), s)
        if r <= 0:
            break
        mu += sign * r
    return abs(mu), evals


def _crossing(f, fT, start, sign, tol=1e-10):
    """Resolved crossing beyond a safe start: outward search, then bisection.
    Returns the bracket (lo safe, hi unsafe); hi is inf when none is found."""
    lo = start
    step = max(start, 1e-6)
    hi = lo + step
    while f(sign * hi) >= fT:
        lo, step = hi, 2 * step
        hi = lo + step
        if hi > 1e3:
            return lo, np.inf
    while (hi - lo) > tol * hi:
        mid = 0.5 * (lo + hi)
        if f(sign * mid) >= fT:
            lo = mid
        else:
            hi = mid
    return lo, hi


def tfim_preparation(fT):
    rows = []
    for L in TFIM_SITES:
        for h0 in TFIM_FIELDS:
            H0, dH_unit = _tfim(L, h0)
            dH = h0 * dH_unit  # relative field error: d H / d mu
            chi, _, gap0 = st.nondegenerate_eigenvector(H0, 0)
            s = st.half_spread(dH)

            def ground(mu, H0=H0, dH=dH):
                return st.nondegenerate_eigenvector(H0 + mu * dH, 0)

            def f(mu, chi=chi, ground=ground):
                return st.state_fidelity(chi, ground(mu)[0])

            def gap_at(mu, ground=ground):
                return ground(mu)[2]

            hstep = 1e-6
            p_plus, p_minus = ground(hstep)[0], ground(-hstep)[0]
            dpsi = (p_plus * np.vdot(p_plus, chi) - p_minus * np.vdot(p_minus, chi)) / (
                2 * hstep
            )
            c_fd = float(np.linalg.norm(dpsi - np.vdot(chi, dpsi) * chi))
            r1 = st.gapped_preparation_radius(np.arccos(fT), gap0, s)
            row = {
                "L": L,
                "h0": h0,
                "gap": gap0,
                "C_prep_fd": c_fd,
                "C_prep_sigma": st.preparation_speed(dH, chi, gap0),
                "C_prep_spread": st.preparation_speed(dH, chi, gap0, bound="spread"),
                "r_one_step": r1,
            }
            for name, sign in (("minus", -1.0), ("plus", 1.0)):
                M, ev = _prep_continuation(f, gap_at, s, fT, sign)
                lo, hi = _crossing(f, fT, M, sign)
                row[f"M_{name}"] = M
                row[f"evals_{name}"] = ev
                row[f"crossing_lo_{name}"] = lo
                row[f"crossing_hi_{name}"] = hi
            rows.append(row)
    return rows


def _integrated_sigma(Hs, dt, B_list, psi0):
    """int_0^T sigma_t dt of the structure in the nominal state (Simpson)."""
    total = 0.0
    psi = np.asarray(psi0, dtype=complex)
    ts = np.linspace(0.0, dt, SUBSTEPS + 1)
    for H, B in zip(Hs, B_list, strict=True):
        w, V = np.linalg.eigh(H)
        coeffs = V.conj().T @ psi
        vals = []
        for t in ts:
            p = V @ (np.exp(-1j * w * t) * coeffs)
            v = B @ p
            m = np.vdot(p, v)
            vals.append(float(np.linalg.norm(v - m * p)))
        vals = np.array(vals)
        h = ts[1] - ts[0]
        total += (
            h
            / 3
            * (vals[0] + vals[-1] + 4 * vals[1:-1:2].sum() + 2 * vals[2:-1:2].sum())
        )
        psi = V @ (np.exp(-1j * w * dt) * coeffs)
    return total


def state_variance(fT):
    problem, controllers = load_ensemble()
    dim = problem["dim"]
    e100 = np.zeros(dim, dtype=complex)
    e100[4] = 1.0  # |100>
    plus = np.ones(dim, dtype=complex) / np.sqrt(dim)
    rows = []
    for idx, c in enumerate(controllers, start=1):
        dt = c["tf"] / c["tau"]
        Hs = [
            problem["H0"] + a * problem["H1"] + b * problem["H2"]
            for a, b in zip(c["u1"], c["u2"], strict=True)
        ]
        structures = {
            "H0": [problem["H0"]] * len(Hs),
            "H1": [a * problem["H1"] for a in c["u1"]],
            "H2": [b * problem["H2"] for b in c["u2"]],
        }
        for name, B_list in structures.items():
            C_state = st.state_speed(B_list, dt)
            C_choi = _choi_speed(B_list, dt)
            for label, psi0 in (("e100", e100), ("plus", plus)):
                rows.append(
                    {
                        "controller": idx,
                        "structure": name,
                        "state": label,
                        "C_state": C_state,
                        "C_choi": C_choi,
                        "int_sigma": _integrated_sigma(Hs, dt, B_list, psi0),
                    }
                )
    return rows


def ghz_dephasing(fT):
    rows = []
    qT = fT**2
    for n in range(1, N_MAX + 1):
        D = HOLD * lb.common_rate_local_dnorm(n).value_certified
        L = D / 2

        def q(g, n=n):
            return 0.5 * (1 + np.exp(-2 * n * g * HOLD))

        if n <= 4:
            ghz = np.zeros(2**n, dtype=complex)
            ghz[0] = ghz[-1] = 1 / np.sqrt(2)
            Gz = sum(lb.dissipator(_op(PAULI_Z, k, n)) for k in range(n))
            for g in (1e-3, 1e-2, 0.1):
                rho = os_.evolve_density(
                    [g * Gz] * HOLD, 1.0, np.outer(ghz, ghz.conj())
                )
                assert abs(np.real(ghz.conj() @ rho @ ghz) - q(g)) < 1e-12
        r0 = (1 - qT) / L
        exact = -np.log(2 * qT - 1) / (2 * n * HOLD)
        m = os_.open_state_fidelity_margin(
            q,
            D,
            qT,
            omega=(0.0, np.inf),
            margin_tol=MARGIN_TOL,
            return_diagnostics=True,
            eta=DEFAULT_ETA,
        )
        rows.append(
            {
                "n": n,
                "D_half": L,
                "r_one_step": r0,
                "M": m.M_plus,
                "M_upper": m.M_upper_plus,
                "n_evals": m.n_evals,
                "crossing": exact,
            }
        )
    return rows


def closed_limit():
    problem, _ = load_ensemble()
    rows = []
    for name in ("H0", "H1", "H2"):
        B = problem[name]
        S = lb.hamiltonian_superop(B)
        w = np.linalg.eigvalsh(B)
        exact = lb.hamiltonian_dnorm(S)
        sdp = lb.diamond_norm(S)
        free = lb.diamond_norm_free(S)
        rows.append(
            {
                "structure": name,
                "spread": float(w[-1] - w[0]),
                "closed_form_certified": exact.value_certified,
                "sdp_raw": sdp.raw,
                "sdp_certified": sdp.value_certified,
                "sdp_solver": sdp.solver,
                "free_certified": free.value_certified,
            }
        )
    return rows


def main() -> None:
    ap = base_parser(OUT_DIR, __doc__.splitlines()[0])
    args = ap.parse_args()
    fT, out = args.FT, args.out
    write_rows(out / f"ghz_detuning_{fT}.csv", ghz_detuning(fT))
    write_rows(out / f"tfim_preparation_{fT}.csv", tfim_preparation(fT))
    write_rows(out / f"ghz_dephasing_{fT}.csv", ghz_dephasing(fT))
    write_rows(out / "closed_limit.csv", closed_limit())
    write_rows(out / f"state_variance_{fT}.csv", state_variance(fT))


if __name__ == "__main__":
    main()

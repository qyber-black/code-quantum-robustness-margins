# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the specialised Kosut-Lidar-Rabitz bound (arXiv:2507.01215)."""

import re
from pathlib import Path

import numpy as np
import pytest
from qrobustness import gate_fidelity, propagator
from qrobustness.kosut import (
    T_OMEGA_MAX,
    fidelity_bound,
    fidelity_bound_at,
    margin,
    threshold_time_bandwidth,
    time_bandwidth,
    uncertainty_rates,
)
from scipy.linalg import expm

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def _random_pwc(seed=0, tau=5, dt=0.3):
    rng = np.random.default_rng(seed)
    H_list = []
    for _ in range(tau):
        f = rng.normal(size=2)
        H_list.append(0.5 * SZ + f[0] * SX + f[1] * SY)
    return H_list, dt


def test_fidelity_bound_monotone_and_endpoints():
    assert fidelity_bound(0.0) == pytest.approx(1.0)
    ys = np.linspace(0.0, T_OMEGA_MAX, 50)
    F = np.array([fidelity_bound(y) for y in ys])
    assert np.all(np.diff(F) <= 1e-15)
    assert fidelity_bound(T_OMEGA_MAX) == 0.0
    assert fidelity_bound(2.0 * T_OMEGA_MAX) == 0.0


def test_threshold_inversion_is_exact():
    for FT in (0.9, 0.99, 0.999, 0.9999):
        y = threshold_time_bandwidth(FT)
        assert fidelity_bound(y) == pytest.approx(FT, abs=1e-12)
    # Nominal error tightens the effective threshold.
    assert threshold_time_bandwidth(0.999, 1e-4) < threshold_time_bandwidth(0.999)
    # No headroom left -> nothing certifiable.
    assert threshold_time_bandwidth(0.999, 2e-3) == 0.0
    H_list, dt = _random_pwc()
    rates = uncertainty_rates(H_list, [SX for _ in H_list], dt)
    assert margin(rates, 0.999, nominal_error=2e-3) == 0.0


def test_rates_scale_and_margin_inverts_bound():
    H_list, dt = _random_pwc()
    dH_list = [0.3 * SX for _ in H_list]
    rates = uncertainty_rates(H_list, dH_list, dt)

    assert rates.T == pytest.approx(len(H_list) * dt)
    assert rates.w_unc == pytest.approx(0.3 * np.linalg.norm(SX, 2))
    assert rates.w_avg >= 0 and rates.w_dev >= 0
    # T*Omega_bnd is symmetric in delta and strictly increasing in |delta|.
    assert time_bandwidth(rates, 0.0) == pytest.approx(0.0)
    assert time_bandwidth(rates, -0.02) == pytest.approx(time_bandwidth(rates, 0.02))
    assert time_bandwidth(rates, 0.01) < time_bandwidth(rates, 0.02)

    FT = 0.999
    M = margin(rates, FT)
    assert M > 0
    assert fidelity_bound_at(rates, M) == pytest.approx(FT, rel=1e-9)
    assert fidelity_bound_at(rates, 1.01 * M) < FT


def test_unc_rate_matches_direct_definition():
    """w_unc, w_avg, w_dev agree with a brute-force interaction-picture sample."""
    H_list, dt = _random_pwc(seed=3, tau=4)
    dH_list = [0.5 * SY for _ in H_list]
    rates = uncertainty_rates(H_list, dH_list, dt, n_quad=40, n_dev=201)

    # Brute-force <Htil> via dense sampling with explicit expm.
    n = 400
    samples = []
    P = np.eye(2, dtype=complex)
    for k, H in enumerate(H_list):
        for s in np.linspace(0.0, dt, n, endpoint=False):
            US = expm(-1j * H * s) @ P
            samples.append(US.conj().T @ dH_list[k] @ US)
        P = expm(-1j * H * dt) @ P
    Havg = sum(samples) / len(samples)

    assert np.linalg.norm(Havg, 2) == pytest.approx(rates.w_avg, rel=1e-3)
    dev = max(np.linalg.norm(S - Havg, 2) for S in samples)
    assert dev == pytest.approx(rates.w_dev, rel=1e-2)


def test_bound_holds_against_true_fidelity():
    """The specialised bound must lower-bound the actual perturbed fidelity."""
    H_list, dt = _random_pwc(seed=7, tau=6)
    Hhat = SZ
    dH_list = [Hhat for _ in H_list]
    rates = uncertainty_rates(H_list, dH_list, dt)
    Uf = propagator(H_list, dt)  # nominal is exact: F_nom = 1

    for delta in (1e-4, 1e-3, 5e-3, 1e-2, 3e-2):
        Fp = gate_fidelity(propagator([H + delta * Hhat for H in H_list], dt), Uf)
        Flb = fidelity_bound_at(rates, delta)
        assert Fp >= Flb - 1e-12


def test_margin_is_conservative_versus_true_threshold_crossing():
    """Their implied margin must not exceed the true fidelity-threshold radius."""
    H_list, dt = _random_pwc(seed=11, tau=6)
    Hhat = SZ
    dH_list = [Hhat for _ in H_list]
    rates = uncertainty_rates(H_list, dH_list, dt)
    Uf = propagator(H_list, dt)
    FT = 0.999

    M = margin(rates, FT)
    Fp = gate_fidelity(propagator([H + M * Hhat for H in H_list], dt), Uf)
    assert Fp >= FT


def test_zero_perturbation_gives_infinite_margin():
    H_list, dt = _random_pwc(seed=5, tau=3)
    dH_list = [np.zeros((2, 2), dtype=complex) for _ in H_list]
    rates = uncertainty_rates(H_list, dH_list, dt)
    assert rates.w_unc == 0.0
    assert margin(rates, 0.999) == float("inf")


def test_input_validation():
    H_list, dt = _random_pwc(tau=2)
    with pytest.raises(ValueError):
        uncertainty_rates([], [], dt)
    with pytest.raises(ValueError):
        uncertainty_rates(H_list, [SX], dt)
    with pytest.raises(ValueError):
        uncertainty_rates(H_list, [SX, SX], 0.0)
    with pytest.raises(ValueError):
        threshold_time_bandwidth(1.0)
    with pytest.raises(ValueError):
        fidelity_bound(-0.1)


def test_csv_headers_match_matlab_peer():
    """The two drivers must agree on CSV column names and order."""
    root = Path(__file__).resolve().parents[2]
    m = (root / "matlab/+qrobustness/+compat/kosut_csv_headers.m").read_text()
    tags = re.search(r"tags = \{([^}]*)\}", m).group(1)
    per = re.search(r"per = \{([^}]*)\}", m).group(1)
    tags = re.findall(r"'([^']+)'", tags)
    per = re.findall(r"'([^']+)'", per)
    matlab_headers = ["controller", "fid", "err"] + [f"{f}_{t}" for t in tags for f in per]

    src = (root / "scripts/run_time_bandwidth_bound_comparison.py").read_text()
    ns: dict = {}
    for line in src.splitlines():
        if line.startswith(("STRUCTURES =", "PER_STRUCTURE =")):
            exec(line, ns)  # noqa: S102 - trusted repo source
    py_headers = ["controller", "fid", "err"] + [
        f"{f}_{t}" for t in ns["STRUCTURES"] for f in ns["PER_STRUCTURE"]
    ]
    assert py_headers == matlab_headers

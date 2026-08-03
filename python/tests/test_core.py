# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path

import numpy as np
import pytest
from qrobustness import (
    dH_structure,
    differential_sensitivity,
    gate_fidelity,
    iterative_margin,
    lipschitz_constant,
    load_controllers,
    load_problem,
    make_fidelity_fn,
    perturbed_hamiltonians,
    propagator,
    structure_constant,
)

ROOT = Path(__file__).resolve().parents[2]


def test_propagator_fidelity():
    N = 2
    U = propagator([np.zeros((N, N)), np.zeros((N, N))], 0.1)
    assert np.linalg.norm(U - np.eye(N)) < 1e-12
    assert abs(gate_fidelity(U, np.eye(N)) - 1.0) < 1e-12

    Hz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    dt = 0.3
    U = propagator([Hz], dt)
    Uref = expm_ref(-1j * dt * Hz)
    assert np.linalg.norm(U - Uref) < 1e-12


def expm_ref(A):
    from scipy.linalg import expm

    return expm(A)


def test_lipschitz_structure():
    H0 = np.array([[0, 1], [1, 0]], dtype=complex)
    H1 = np.array([[1, 0], [0, -1]], dtype=complex)
    dt = 0.25
    tau = 4
    tf = tau * dt
    controls = np.array([0.5, -0.2, 0.1, 0.3])

    C0 = structure_constant("drift", H0, dt, tau)
    assert abs(C0 - tf * np.linalg.norm(H0, "fro")) < 1e-14

    C1 = structure_constant("control", H1, dt, tau, controls)
    assert abs(C1 - dt * np.linalg.norm(controls, 1) * np.linalg.norm(H1, "fro")) < 1e-14

    FT = 0.999
    L = lipschitz_constant(FT, 2, C0)
    B = np.sqrt((1 - FT**2) / 2)
    assert abs(L - B * C0) < 1e-14


def test_perfect_fidelity_zeta():
    H = np.array([[0, 1], [1, 0]], dtype=complex)
    dt = 0.2
    from scipy.linalg import expm

    Uf = expm(-1j * dt * H)
    zeta = differential_sensitivity([H], [H], dt, Uf, n_quad=48)
    assert abs(zeta) < 1e-9


def test_iterative_margin_synthetic():
    FT = 0.99
    fidelity_fn = lambda mu: max(0.0, 1.0 - 0.05 * abs(mu))
    L = 0.05
    res = iterative_margin(fidelity_fn, L, FT, mu0=0.0, eta=1e-8, k_max=1000)
    assert abs(res.M_minus - 0.2) < 1e-5
    assert abs(res.M_plus - 0.2) < 1e-5
    assert abs(res.M - 0.2) < 1e-5
    assert res.converged_minus and res.converged_plus

    res_b = iterative_margin(
        fidelity_fn, L, FT, mu0=0.0, eta=1e-8, omega=(-0.05, np.inf), k_max=1000
    )
    assert abs(res_b.M_minus - 0.05) < 1e-8

    res_big = iterative_margin(fidelity_fn, 10.0, FT, mu0=0.0, eta=1e-6, k_max=5000)
    assert abs(res_big.M - 0.2) < 5e-4


def test_threshold_error():
    with pytest.raises(ValueError):
        iterative_margin(lambda mu: 0.95, 1.0, 0.99, mu0=0.0)


def test_load_case_study_smoke():
    CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
    problem = load_problem(CTRL / "problem9.mat")
    assert problem["dim"] == 8
    controllers = load_controllers(CTRL / "controllers.csv", 1e-4)
    assert len(controllers) == 61

    c = controllers[0]
    dt = c["tf"] / c["tau"]
    H_list = perturbed_hamiltonians(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], "H0", 0.0
    )
    U = propagator(H_list, dt)
    F = gate_fidelity(U, problem["Uf"])
    assert abs(F - c["fid"]) < 1e-4

    dH = dH_structure(problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], "H0")
    zeta = differential_sensitivity(H_list, dH, dt, problem["Uf"], n_quad=24)
    assert np.isfinite(zeta)


def test_iterative_margin_case_study_certificate():
    """Algorithm 1 on real case-study fidelity: F(+/-M) >= FT and M = min(M+/-)."""
    CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
    FT = 0.999
    eta = 1e-6
    problem = load_problem(CTRL / "problem9.mat")
    controllers = load_controllers(CTRL / "controllers.csv", 1e-4)
    c = controllers[0]  # controller 1 in 1-based paper indexing
    dt = c["tf"] / c["tau"]
    C = structure_constant("drift", problem["H0"], dt, c["tau"])
    L = lipschitz_constant(FT, problem["dim"], C)
    fid_fn = make_fidelity_fn(
        problem["H0"],
        problem["H1"],
        problem["H2"],
        c["u1"],
        c["u2"],
        problem["Uf"],
        dt,
        "H0",
    )
    res = iterative_margin(fid_fn, L, FT, mu0=0.0, eta=eta)
    assert res.M == min(res.M_minus, res.M_plus)
    assert res.converged_minus and res.converged_plus
    assert fid_fn(-res.M) >= FT - 1e-12
    assert fid_fn(res.M) >= FT - 1e-12
    # Match published MATLAB table for ctrl1 H0
    table_csv = ROOT / "results/lipschitz-margin-matlab/margins_table_0.999.csv"
    if table_csv.is_file():
        import csv

        with table_csv.open(newline="") as f:
            row = next(csv.DictReader(f))
        assert abs(res.M - float(row["M_H0"])) < 1e-10
        assert abs(res.M_minus - float(row["Mm_H0"])) < 1e-10
        assert abs(res.M_plus - float(row["Mp_H0"])) < 1e-10


def test_certified_bracket_does_not_jump_safe_islands():
    """With a nonmonotone fidelity (an unsafe dip followed by a safe
    island), the certified lower margin must stop at the nominal
    component's boundary rather than promoting pointwise-safe island
    samples (round-2 review, Appendix A semantics)."""
    import numpy as np
    from qrobustness import iterative_margin

    FT = 0.9
    # F: safe plateau to |mu| ~ 0.1, dip below FT on [0.12, 0.2],
    # safe island beyond. Lipschitz constant consistent with slopes.
    def fid(mu):
        x = abs(mu)
        if x < 0.1:
            return 0.99 - 0.5 * x
        if x < 0.16:
            return 0.94 - 0.5 * (x - 0.1) * 10.0   # crosses FT at 0.108
        if x < 0.24:
            return 0.64 + 0.5 * (x - 0.16) * 10.0  # recovers, crosses up at 0.212
        return 0.99

    L = 5.0  # valid Lipschitz constant for the profile above
    res = iterative_margin(fid, L, FT, mu0=0.0, eta=1e-6, margin_tol=1e-6)
    # True first boundary at |mu| = 0.108: certified M must not exceed
    # it, and the upper witness must lie beyond it but well before the
    # island could be mistaken for the component (upper < 0.2).
    assert res.M <= 0.108 + 1e-6
    assert res.M >= 0.108 - 1e-3
    assert res.M_upper <= 0.2


def test_certified_bracket_partial_when_island_masks_first_crossing():
    """Discriminating island case: Algorithm 1 stops early on an
    eta-band plateau, and the outward geometric probe's step grows
    large enough to leap a narrow unsafe dip straight into a wide safe
    island.  Uncertified promotion would report the island's far
    boundary (~0.3) as the margin; the certified-promotion rule must
    keep M at the nominal component and report a rigorous (if wide)
    'partial' bracket that still contains the true first crossing."""
    import numpy as np
    from qrobustness import iterative_margin

    FT = 0.9
    # |mu| profile (L = 300 valid throughout):
    #   descent slope 2 to an eta-band plateau F = FT + 5e-7 from ~0.045,
    #   narrow dip on [0.1, 0.1005] (slope 100) crossing FT at 0.1+5e-9,
    #   recovery slope 280 into a safe island at 0.99 up to 0.3,
    #   final descent slope 300 crossing FT at ~0.3003.
    def fid(mu):
        x = abs(mu)
        if x < 0.1:
            return max(FT + 5e-7, 0.99 - 2.0 * x)
        if x < 0.1005:
            return FT + 5e-7 - 100.0 * (x - 0.1)
        if x < 0.3:
            return min(0.99, FT + 5e-7 - 0.05 + 280.0 * (x - 0.1005))
        return max(0.0, 0.99 - 300.0 * (x - 0.3))

    res = iterative_margin(fid, 300.0, FT, mu0=0.0, eta=1e-6, margin_tol=1e-6)
    # True first boundary at |mu| ~= 0.1: the certified margin must not
    # be inflated to the island's far edge (~0.3003) ...
    assert res.M <= 0.1 + 1e-6
    assert res.M >= 0.04
    # ... the bracket must still contain the true first crossing ...
    assert np.isfinite(res.M_upper)
    assert res.M_upper >= 0.1
    # ... and the unreached tolerance must be reported, not hidden.
    assert res.reason_minus == "partial"
    assert res.reason_plus == "partial"


# --------------------------------------------------------------------------
# Traceless centring of the perturbation structure (paper Sec. IV)
# --------------------------------------------------------------------------


def test_traceless_centring_removes_only_the_trace():
    import numpy as np
    from qrobustness import structure_constant
    from qrobustness.core import traceless

    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H = A + A.conj().T
    Hc = traceless(H)
    assert abs(np.trace(Hc)) < 1e-12
    # Centring is a projection: it removes only the identity component.
    assert np.allclose(Hc, traceless(Hc))
    assert np.linalg.norm(Hc, "fro") <= np.linalg.norm(H, "fro") + 1e-12


def test_structure_constant_is_gauge_invariant():
    """C_Hhat must not change when a multiple of I is added to the structure.

    The trace part contributes only a global phase to the propagator, which
    the trace-amplitude fidelity ignores, so a constant shift of the structure
    must leave the certificate untouched.
    """
    import numpy as np
    from qrobustness import structure_constant

    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H = A + A.conj().T
    u = rng.normal(size=7)
    for c in (-3.0, 0.5, 11.0):
        assert structure_constant("drift", H + c * np.eye(4), 0.3, 7) == pytest.approx(
            structure_constant("drift", H, 0.3, 7), rel=1e-12
        )
        assert structure_constant(
            "control", H + c * np.eye(4), 0.3, 7, u
        ) == pytest.approx(structure_constant("control", H, 0.3, 7, u), rel=1e-12)


def test_structure_constant_vanishes_for_identity_structure():
    """A pure-identity structure only rephases U, so it cannot move F."""
    import numpy as np
    from qrobustness import structure_constant

    assert structure_constant("drift", np.eye(8), 0.5, 32) == 0.0
    assert structure_constant("control", 2.5 * np.eye(8), 0.5, 32, np.ones(32)) == 0.0


def test_structure_constant_tightens_for_nontraceless_sparse_structure():
    """A sparse, non-traceless Hermitian structure -- e.g. a detuning on one
    level -- is exactly the case the centring is for: the constant must be
    strictly smaller than the uncentred one, hence the margin strictly larger."""
    import numpy as np
    from qrobustness import lipschitz_constant, structure_constant

    N = 8
    H = np.zeros((N, N), dtype=complex)
    H[0, 0] = 1.0  # single-level detuning: Tr = 1, not traceless
    C = structure_constant("drift", H, 0.5, 32)
    C_uncentred = 32 * 0.5 * float(np.linalg.norm(H, "fro"))
    assert C < C_uncentred
    assert C == pytest.approx(32 * 0.5 * np.sqrt(1.0 - 1.0 / N))
    # A smaller constant is a smaller Lipschitz constant, hence a larger
    # certified radius for the same fidelity surplus.
    assert lipschitz_constant(0.999, N, C) < lipschitz_constant(0.999, N, C_uncentred)


def test_case_study_structures_are_unaffected_by_centring():
    """The three case-study structures are traceless, so nothing moves."""
    import numpy as np
    from qrobustness import load_problem, structure_constant

    CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
    problem = load_problem(CTRL / "problem9.mat")
    for tag in ("H0", "H1", "H2"):
        H = problem[tag]
        assert abs(np.trace(H)) < 1e-12
        assert structure_constant("drift", H, 0.46875, 32) == pytest.approx(
            32 * 0.46875 * float(np.linalg.norm(H, "fro")), rel=1e-15
        )


def test_structure_constant_rejects_bad_structures():
    import numpy as np
    from qrobustness import structure_constant

    with pytest.raises(ValueError, match="square"):
        structure_constant("drift", np.ones((2, 3)), 0.5, 4)
    with pytest.raises(ValueError, match="Hermitian"):
        structure_constant("drift", np.array([[0.0, 1.0], [0.0, 0.0]]), 0.5, 4)


# --------------------------------------------------------------------------
# Algorithm 1 stopping status
# --------------------------------------------------------------------------


def _tent(FT=0.999, surplus=0.01):
    """F is a tent of slope 1 peaking at mu=0; crossing at |mu| = surplus."""
    return lambda mu: FT + surplus - abs(mu)


def test_status_reports_eta_band_in_both_directions():
    from qrobustness import iterative_margin

    r = iterative_margin(_tent(), 1.0, 0.999, eta=1e-6)
    assert r.status_minus == "eta_band"
    assert r.status_plus == "eta_band"
    assert r.converged_minus and r.converged_plus


def test_status_reports_domain_truncation_in_both_directions():
    """A domain-truncated margin is not a resolved margin; converged_* alone
    cannot tell them apart, which is why status_* exists."""
    from qrobustness import iterative_margin

    r = iterative_margin(_tent(), 1.0, 0.999, eta=1e-6, omega=(-0.002, 0.002))
    assert r.status_minus == "domain_truncated"
    assert r.status_plus == "domain_truncated"
    assert r.M == pytest.approx(0.002)
    # ... and the flag that predates it cannot distinguish the two cases.
    assert r.converged_minus and r.converged_plus


def test_status_reports_domain_truncation_on_one_side_only():
    from qrobustness import iterative_margin

    r = iterative_margin(_tent(), 1.0, 0.999, eta=1e-6, omega=(-0.002, np.inf))
    assert r.status_minus == "domain_truncated"
    assert r.status_plus == "eta_band"


def test_k_max_counts_evaluated_steps_exactly():
    """K_max is the number of evaluated trial points per direction, matching
    the paper's Algorithm 1 (k starts at 1 and the guard is k >= K_max).

    The off-by-one matters only when the limit actually binds, but the paper
    and both engines must agree on the convention.
    """
    from qrobustness import iterative_margin

    for k_max in (1, 2, 3):
        n = [0]

        def counted(mu, n=n):
            n[0] += 1
            return _tent()(mu)

        # L far above the true slope, so the eta band is never reached first.
        r = iterative_margin(counted, 1e4, 0.999, eta=1e-12, k_max=k_max,
                             return_diagnostics=True)
        assert r.status_minus == "iteration_limit"
        # n_steps counts recentrings, one fewer than the evaluated steps.
        assert r.n_steps // 2 == k_max - 1


def test_status_reports_iteration_limit():
    from qrobustness import iterative_margin

    # L far larger than the true slope makes every certified step tiny, so the
    # eta band is never reached within k_max.
    r = iterative_margin(_tent(), 1e4, 0.999, eta=1e-12, k_max=2)
    assert r.status_minus == "iteration_limit"
    assert r.status_plus == "iteration_limit"
    assert not r.converged_minus
    assert not r.converged_plus


def test_status_is_populated_without_margin_tol():
    """The whole point: the default path reports how it stopped."""
    from qrobustness import iterative_margin
    from qrobustness.core import MARGIN_STATUS

    r = iterative_margin(_tent(), 1.0, 0.999)
    assert r.status_minus in MARGIN_STATUS
    assert r.status_plus in MARGIN_STATUS
    # reason_* is a different quantity and stays unset without margin_tol.
    assert r.reason_minus == "unknown"


def test_safeguard_flag_tracks_the_bisection_fallback():
    from qrobustness import iterative_margin

    # An under-estimated L is precisely the case the safeguard exists for:
    # the "certified" step overshoots the threshold and must be bisected back.
    assert iterative_margin(_tent(), 0.5, 0.999, eta=1e-6).safeguard_minus
    # A valid L cannot overshoot in exact arithmetic, so it never fires.
    assert not iterative_margin(_tent(), 2.0, 0.999, eta=1e-6).safeguard_minus

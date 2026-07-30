# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Error control: every approximated quantity must carry a usable certificate.

Two quantities in the package are not exact to roundoff:

* ``kosut.uncertainty_rates`` -> ``w_dev``, a supremum recovered from samples;
* ``iterative_margin`` -> ``M``, terminated on a *fidelity* band ``eta``.

Both must (a) be conservative in a stated direction and (b) reach a requested
precision when asked.
"""
from pathlib import Path

import numpy as np
import pytest

from qrobustness import (
    dH_structure,
    iterative_margin,
    lipschitz_constant,
    load_controllers,
    load_problem,
    make_fidelity_fn,
    structure_constant,
)
from qrobustness import kosut

ROOT = Path(__file__).resolve().parents[2]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
FT = 0.999
STRUCTURES = ["H0", "H1", "H2"]


@pytest.fixture(scope="module")
def case():
    problem = load_problem(CTRL / "problem9.mat")
    controllers = load_controllers(CTRL / "controllers.csv")
    return problem, controllers


def _rates(problem, c, structure, **kw):
    dt = c["tf"] / c["tau"]
    H_list = [
        problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
        for k in range(c["tau"])
    ]
    dH_list = dH_structure(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], structure
    )
    return kosut.uncertainty_rates(H_list, dH_list, dt, **kw)


def _margin_inputs(problem, c, structure):
    dt = c["tf"] / c["tau"]
    Hhat = {"H0": problem["H0"], "H1": problem["H1"], "H2": problem["H2"]}[structure]
    if structure == "H0":
        C = structure_constant("drift", Hhat, dt, c["tau"])
    else:
        controls = c["u1"] if structure == "H1" else c["u2"]
        C = structure_constant("control", Hhat, dt, c["tau"], controls)
    L = lipschitz_constant(FT, problem["dim"], C)
    fn = make_fidelity_fn(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"],
        problem["Uf"], dt, structure,
    )
    return fn, L


# --------------------------------------------------------------------------
# Kosut uncertainty rates
# --------------------------------------------------------------------------


@pytest.mark.parametrize("structure", STRUCTURES)
def test_w_avg_is_exact_not_quadrature(case, structure):
    """The closed-form time average must match a high-order quadrature."""
    problem, controllers = case
    c = controllers[0]
    dt = c["tf"] / c["tau"]
    H_list = [
        problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
        for k in range(c["tau"])
    ]
    dH_list = dH_structure(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], structure
    )
    N = problem["H0"].shape[0]

    # Reference: brute-force Gauss-Legendre on the interaction-picture integrand.
    x, w = np.polynomial.legendre.leggauss(200)
    nodes = 0.5 * dt * (x + 1.0)
    weights = 0.5 * dt * w
    eigs = [np.linalg.eigh(0.5 * (H + H.conj().T)) for H in H_list]
    Pref = [np.eye(N, dtype=complex)]
    for lam, V in eigs:
        Pref.append(((V * np.exp(-1j * dt * lam)) @ V.conj().T) @ Pref[-1])
    acc = np.zeros((N, N), dtype=complex)
    for k, (lam, V) in enumerate(eigs):
        for s, wq in zip(nodes, weights):
            E = (V * np.exp(1j * s * lam)) @ V.conj().T
            acc += wq * (Pref[k].conj().T @ (E @ dH_list[k] @ E.conj().T) @ Pref[k])
    ref = float(np.linalg.norm(acc / (c["tau"] * dt), 2))

    got = _rates(problem, c, structure).w_avg
    assert got == pytest.approx(ref, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("structure", STRUCTURES)
def test_w_dev_respects_both_certificates(case, structure):
    """The sampled/polished sup must sit inside both rigorous bounds."""
    problem, controllers = case
    for c in controllers[:4]:
        r = _rates(problem, c, structure)
        assert r.w_dev_bracket_lo - 1e-12 <= r.w_dev <= r.w_dev_bracket_hi + 1e-12
        assert r.w_dev <= r.w_dev_certified + 1e-12
        assert r.w_dev_bracket_lo >= 0.0


@pytest.mark.parametrize("structure", STRUCTURES)
def test_w_dev_converges_and_beats_plain_sampling(case, structure):
    """Polishing must reach the brute-force supremum; sampling alone need not."""
    problem, controllers = case
    c = controllers[0]
    adaptive = _rates(problem, c, structure)
    brute = _rates(problem, c, structure, n_dev=4001, adaptive_dev=False).w_dev

    assert adaptive.dev_converged
    # Sampling can only under-estimate a supremum, so even a fine sampled
    # grid must not exceed the polished value -- and must be close to it.
    assert brute <= adaptive.w_dev + 1e-12
    assert adaptive.w_dev == pytest.approx(brute, rel=1e-7)
    # The default 17-point grid is much coarser, so it must also fall short.
    sampled = _rates(problem, c, structure, adaptive_dev=False).w_dev
    assert sampled <= adaptive.w_dev + 1e-12


def test_w_dev_sampling_is_biased_optimistic(case):
    """Under-estimating w_dev inflates the Kosut margin; pin the direction."""
    problem, controllers = case
    c = controllers[0]
    sampled = _rates(problem, c, "H1", adaptive_dev=False)
    polished = _rates(problem, c, "H1")
    assert sampled.w_dev < polished.w_dev  # sampling fell short
    m_sampled = kosut.margin(sampled, FT, nominal_error=c["error"])
    m_polished = kosut.margin(polished, FT, nominal_error=c["error"])
    # ... and that bias favours the bound.
    assert m_sampled > m_polished


@pytest.mark.parametrize("structure", STRUCTURES)
def test_w_dev_grid_is_derived_from_bandwidth(case, structure):
    """The seed grid must follow the Bohr bandwidth, not a fixed constant."""
    problem, controllers = case
    c = controllers[0]
    r = _rates(problem, c, structure)
    assert r.dev_resolved
    assert r.dev_cycles_max > 0
    # At least the requested density (default 16 samples/cycle).
    assert r.dev_samples_per_cycle >= 16.0


def test_w_dev_grid_scales_with_interval_length(case):
    """A longer interval means more cycles, so the grid must grow with it.

    A fixed grid fails here: at approximately 21 cycles per interval a 17-point
    grid substantially under-resolves the supremum, whereas the bandwidth-derived
    grid follows it.
    """
    problem, controllers = case
    c = controllers[0]
    H_list = [
        problem["H0"] + c["u1"][k] * problem["H1"] + c["u2"][k] * problem["H2"]
        for k in range(c["tau"])
    ]
    dH_list = dH_structure(
        problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], "H1"
    )
    dt_long = 20.0 * c["tf"] / c["tau"]

    derived = kosut.uncertainty_rates(H_list, dH_list, dt_long)
    fixed = kosut.uncertainty_rates(H_list, dH_list, dt_long, n_dev=17, adaptive_dev=False)
    reference = kosut.uncertainty_rates(
        H_list, dH_list, dt_long, n_dev=40001, n_dev_max=40001, adaptive_dev=False
    ).w_dev

    assert derived.dev_cycles_max > 10.0          # a stress case
    assert derived.dev_resolved
    assert derived.n_dev_used > 17                # grid grew with the bandwidth
    # A sampled reference can only under-estimate, so the polished value is above it.
    assert derived.w_dev >= reference - 1e-12
    # And the derived grid is far closer to the truth than the fixed one.
    err_derived = abs(derived.w_dev - reference) / reference
    err_fixed = abs(fixed.w_dev - reference) / reference
    assert err_derived < err_fixed / 100.0


def test_n_quad_is_accepted_but_ignored(case):
    """<Htil> is closed form now; n_quad must not change the answer."""
    problem, controllers = case
    c = controllers[0]
    a = _rates(problem, c, "H1", n_quad=4)
    b = _rates(problem, c, "H1", n_quad=512)
    assert a.w_avg == b.w_avg


# --------------------------------------------------------------------------
# Margin
# --------------------------------------------------------------------------


@pytest.mark.parametrize("structure", STRUCTURES)
def test_margin_reaches_requested_precision(case, structure):
    problem, controllers = case
    for c in controllers[:3]:
        fn, L = _margin_inputs(problem, c, structure)
        for tol in (1e-6, 1e-10):
            r = iterative_margin(fn, L, FT, margin_tol=tol)
            assert r.reason_minus == "bracketed"
            assert r.reason_plus == "bracketed"
            assert r.margin_uncertainty / r.M <= tol * 1.000001


@pytest.mark.parametrize("structure", STRUCTURES)
def test_margin_bracket_is_a_true_bracket(case, structure):
    """F >= FT at the reported margin, and F < FT just past M_upper."""
    problem, controllers = case
    c = controllers[0]
    fn, L = _margin_inputs(problem, c, structure)
    r = iterative_margin(fn, L, FT, margin_tol=1e-10)
    assert fn(r.M) >= FT
    assert fn(-r.M) >= FT
    assert min(fn(r.M_upper_plus), fn(-r.M_upper_minus)) < FT


@pytest.mark.parametrize("structure", STRUCTURES)
def test_margin_tol_only_tightens(case, structure):
    """The refined margin is never smaller than the default one."""
    problem, controllers = case
    for c in controllers[:3]:
        fn, L = _margin_inputs(problem, c, structure)
        assert iterative_margin(fn, L, FT, margin_tol=1e-10).M >= (
            iterative_margin(fn, L, FT).M - 1e-15
        )


def test_default_eta_uncertainty_is_material(case):
    """Document the headline number: eta=1e-6 leaves ~5e-4 relative in M."""
    problem, controllers = case
    c = controllers[0]
    fn, L = _margin_inputs(problem, c, "H1")
    coarse = iterative_margin(fn, L, FT).M
    fine = iterative_margin(fn, L, FT, margin_tol=1e-12).M
    rel = abs(fine - coarse) / fine
    assert 1e-5 < rel < 1e-2, f"expected ~5e-4 relative, got {rel:.2e}"


@pytest.mark.parametrize(
    "method,expected",
    [
        ("algorithm1", "segment"),
        ("lipschitz_brent", "segment"),
        ("lipschitz_toms748", "segment"),
        ("doubling", "endpoint"),
    ],
)
def test_certificate_class_is_reported(case, method, expected):
    """Callers must not have to interpret the method string themselves."""
    problem, controllers = case
    fn, L = _margin_inputs(problem, controllers[0], "H0")
    assert iterative_margin(fn, L, FT, method=method).certificate == expected


def test_margin_tol_must_be_positive(case):
    problem, controllers = case
    fn, L = _margin_inputs(problem, controllers[0], "H0")
    with pytest.raises(ValueError, match="margin_tol must be positive"):
        iterative_margin(fn, L, FT, margin_tol=0.0)

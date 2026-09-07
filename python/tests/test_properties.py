# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Checks of some properties.

* the Choi conversion is an exact rearrangement of stored entries, which
  is what lets the verified SDP bound speak about the represented
  superoperator (Appendix A);
* the positive-semidefiniteness verification is a genuine predicate --
  it accepts matrices that are PSD with room to spare and rejects
  indefinite ones -- and the repaired blocks it returns are the ones the
  objective is evaluated on;
* the segment derivative is the true Frechet derivative on a
  NON-COMMUTING example, checked against the block-exponential
  reference; the commuting case alone would not detect a commuting
  approximation;
* the static joint gauge is NOT invariant under individual amplitude
  sign changes for non-orthogonal structures (the counterexample behind
  the restricted Proposition on sign invariance), while the trajectory
  box gauge is;
* the analytic zero-Hamiltonian rate responses and their first-order
  slopes, the independent check on normalisation, propagation and the
  rate-response constants;
* the coherence-time conversion returns the pure-dephasing time and
  composes both channels for the total T2.
"""

import numpy as np
import pytest
from scipy.linalg import expm

from qrobustness import lindblad as lb
from qrobustness.core import dU_dmu_exact, segment_eig
from qrobustness.lengthspace import PathGauge, interval_grams

SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
SM = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)


# -- Choi conversion is a rearrangement, not a computation --------------


@pytest.mark.parametrize("N", [2, 3, 4])
def test_choi_conversion_is_exact_reindexing(N):
    rng = np.random.default_rng(7)
    S = rng.normal(size=(N * N, N * N)) + 1j * rng.normal(size=(N * N, N * N))
    J = lb.choi_matrix(S)
    # Every entry of J is an entry of S, and the inverse map returns the
    # stored bits, not merely a close value.
    assert sorted(J.ravel().tolist(), key=lambda z: (z.real, z.imag)) == sorted(
        S.ravel().tolist(), key=lambda z: (z.real, z.imag)
    )
    assert np.array_equal(lb.superop_from_choi(J, N), S)
    assert lb.choi_roundtrip_exact(S)


def test_choi_matches_the_defining_sum():
    """The reindexing agrees bit for bit with the definition."""
    N = 3
    rng = np.random.default_rng(11)
    S = rng.normal(size=(N * N, N * N)) + 1j * rng.normal(size=(N * N, N * N))
    ref = np.zeros((N * N, N * N), dtype=complex)
    for i in range(N):
        for j in range(N):
            E = np.zeros((N, N), dtype=complex)
            E[i, j] = 1.0
            out = (S @ E.reshape(-1, order="F")).reshape((N, N), order="F")
            ref += np.kron(out, E)
    assert np.array_equal(lb.choi_matrix(S), ref)


# -- the PSD verification is a predicate, not an assertion --------------


def test_verify_psd_accepts_and_rejects():
    rng = np.random.default_rng(3)
    n = 6
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    G = A @ A.conj().T  # positive definite with room to spare
    G = (G + G.conj().T) / 2
    assert lb._verify_psd(G)
    # An indefinite matrix must fail, however small the negative
    # eigenvalue: shift below the smallest eigenvalue.
    lam = float(np.linalg.eigvalsh(G).min())
    assert not lb._verify_psd(G - (lam + 1.0) * np.eye(n))
    assert not lb._verify_psd(-G)


def test_verify_psd_rejects_a_marginal_matrix():
    """A matrix whose smallest eigenvalue is below the Cholesky backward
    error must not be accepted: the criterion has to be strict enough to
    cover its own rounding."""
    n = 8
    A = np.eye(n, dtype=complex)
    A[0, 0] = -1e-13
    assert not lb._verify_psd(A)


def test_diag_shift_is_verified_exactly():
    from fractions import Fraction

    for a, c in ((1.0, 1e-16), (3.25, 1e-9), (1e-8, 1e-12)):
        t = lb._shift_diag_down(a, c)
        assert Fraction(t) <= Fraction(a) - Fraction(c)


def test_verified_repair_returns_the_blocks_it_proved():
    """The repaired pair is feasible as stored, and the shift is the one
    that produced those stored matrices."""
    N = 2
    d = N * N
    rng = np.random.default_rng(5)
    S = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    J = lb.choi_matrix(S)
    Y0 = np.zeros((d, d), dtype=complex)
    Y1 = np.zeros((d, d), dtype=complex)
    Y0r, Y1r, eps = lb._verified_psd_repair(Y0, Y1, J, 0.0)
    assert np.array_equal(Y0r, Y0 + eps * np.eye(d))
    B = np.block([[Y0r, -J], [-J.conj().T, Y1r]])
    assert lb._verify_psd(B)


def test_verification_failure_is_raised_not_papered_over():
    N = 2
    d = N * N
    J = np.full((d, d), np.nan, dtype=complex)
    with pytest.raises(lb.VerificationFailure):
        lb._verified_psd_repair(
            np.zeros((d, d), dtype=complex),
            np.zeros((d, d), dtype=complex),
            J,
            0.0,
            max_tries=3,
        )


def test_solver_free_bound_is_verified_and_above_the_closed_form():
    for n in (1, 2):
        G = sum(lb.dissipator(V) for V in lb.local_ops(SZ, n))
        dn = lb.diamond_norm_free(G)
        assert dn.value_certified >= 2 * n
        assert dn.value_certified <= 2 * n + 1e-6


def test_common_rate_local_dnorm_is_the_closed_form():
    for n in (1, 2, 3):
        dn = lb.common_rate_local_dnorm(n)
        assert dn.value_certified == float(2 * n)
        assert dn.status == "analytic"
        assert dn.feas_shift == 0.0
    with pytest.raises(ValueError):
        lb.common_rate_local_dnorm(0)


# -- the segment derivative is Frechet, not a commuting shortcut --------


@pytest.mark.parametrize(
    "H, dH",
    [
        (SZ, SX),  # non-commuting: the case that separates the two
        (SZ, SZ),  # commuting: passes either way, kept as the control
        (0.7 * SZ + 0.3 * SX, SX),
    ],
)
def test_segment_derivative_against_block_exponential(H, dH):
    dt = 0.37
    lam, V = segment_eig(H)
    got = dU_dmu_exact(lam, V, dH, dt)
    # Independent reference: the upper-right block of the block
    # exponential, which assumes nothing about commutation.
    n = H.shape[0]
    M = np.zeros((2 * n, 2 * n), dtype=complex)
    M[:n, :n] = -1j * H
    M[:n, n:] = -1j * dH
    M[n:, n:] = -1j * H
    ref = expm(dt * M)[:n, n:]
    assert np.allclose(got, ref, atol=1e-12)


def test_commuting_shortcut_would_fail_the_non_commuting_case():
    """The check above has teeth: -i dt dH U is wrong when [H, dH] != 0."""
    dt = 0.37
    lam, V = segment_eig(SZ)
    exact = dU_dmu_exact(lam, V, SX, dt)
    naive = -1j * dt * SX @ ((V * np.exp(-1j * dt * lam)) @ V.conj().T)
    assert not np.allclose(exact, naive, atol=1e-6)


# -- sign invariance of the static gauge is conditional -----------------


def test_static_gauge_is_not_sign_invariant_for_equal_structures():
    """H_1 = H_2 = Z on one interval: amplitudes (1, 1) give 2Z and
    (1, -1) give 0.  An implementation returning the same gauge for both
    has discarded the Gram off-diagonal."""
    gauge = PathGauge(grams=interval_grams([[SZ], [SZ]], make_traceless=True), dt=1.0)
    same = gauge.C(np.array([1.0, 1.0]))
    opposite = gauge.C(np.array([1.0, -1.0]))
    assert same == pytest.approx(2.0 * np.linalg.norm(SZ, "fro"))
    assert opposite == pytest.approx(0.0, abs=1e-12)
    assert same != pytest.approx(opposite)


def test_offdiagonal_grams_are_retained():
    G = interval_grams([[SZ], [SZ]], make_traceless=True)[0]
    assert G[0, 1] == pytest.approx(G[0, 0])
    assert G[0, 1] != 0.0


def test_trajectory_box_gauge_absorbs_the_sign_change():
    """The box gauge maximises over sign vertices, so it sees magnitudes
    only and is sign-invariant where the static gauge is not."""
    gauge = PathGauge(grams=interval_grams([[SZ], [SZ]], make_traceless=True), dt=1.0)
    assert gauge.C_box(np.array([1.0, 1.0])) == pytest.approx(
        gauge.C_box(np.array([1.0, 1.0]))
    )
    # The worst vertex of the box |x_j| <= 1 is the aligned one, whatever
    # the sign of the requested magnitudes.
    assert gauge.C_box(np.array([1.0, 1.0])) == pytest.approx(
        gauge.C(np.array([1.0, 1.0]))
    )


# -- analytic rate responses --------------------------------------------


def _analytic_dephasing(gamma, t_f, n):
    return ((1.0 + np.exp(-2.0 * gamma * t_f)) / 2.0) ** n


def _analytic_amp_damping(gamma, t_f, n):
    return ((1.0 + np.exp(-gamma * t_f / 2.0)) ** 2 / 4.0) ** n


@pytest.mark.parametrize("n", [1, 2])
def test_zero_hamiltonian_rate_responses_match_the_closed_forms(n):
    t_f = 1.7
    Uf = np.eye(2**n, dtype=complex)
    for op, ref in ((SZ, _analytic_dephasing), (SM, _analytic_amp_damping)):
        G = sum(lb.dissipator(V) for V in lb.local_ops(op, n))
        for gamma in (0.0, 0.03, 0.5):
            S = lb.channel([gamma * G], t_f)
            assert lb.process_fidelity(S, Uf) == pytest.approx(
                ref(gamma, t_f, n), abs=1e-12
            )


@pytest.mark.parametrize("n", [1, 2])
def test_first_order_rate_slopes(n):
    """d F_pro/d gamma at zero is -n t_f (dephasing) and -n t_f/2
    (amplitude damping): the independent check of normalisation,
    propagation and the rate-response constants."""
    t_f = 1.7
    Uf = np.eye(2**n, dtype=complex)
    h = 1e-6
    for op, want in ((SZ, -n * t_f), (SM, -n * t_f / 2.0)):
        G = sum(lb.dissipator(V) for V in lb.local_ops(op, n))

        def F(g, G=G):
            return lb.process_fidelity(lb.channel([g * G], t_f), Uf)

        slope = (F(h) - F(-h)) / (2 * h)
        assert slope == pytest.approx(want, rel=1e-6)


# -- coherence-time conversion ------------------------------------------


def test_dephasing_conversion_returns_the_pure_dephasing_time():
    assert lb.dephasing_time(0.5) == pytest.approx(1.0)
    # With amplitude damping present, T2 is strictly shorter than T_phi.
    assert lb.coherence_time(0.5, 0.0) == pytest.approx(lb.dephasing_time(0.5))
    assert lb.coherence_time(0.5, 1.0) < lb.dephasing_time(0.5)
    assert 1.0 / lb.coherence_time(0.5, 1.0) == pytest.approx(
        1.0 / lb.dephasing_time(0.5) + 1.0 / (2.0 * lb.relaxation_time(1.0))
    )


def test_zero_rates_give_infinite_times():
    assert np.isinf(lb.dephasing_time(0.0))
    assert np.isinf(lb.relaxation_time(0.0))
    assert np.isinf(lb.coherence_time(0.0, 0.0))


def test_rates_from_times_round_trip():
    for T1, T2 in ((1.0, 1.0), (2.0, 3.0), (5.0, 10.0)):
        gz, gm = lb.rates_from_times(T1, T2)
        assert lb.relaxation_time(gm) == pytest.approx(T1)
        assert lb.coherence_time(gz, gm) == pytest.approx(T2)
    with pytest.raises(ValueError):
        lb.rates_from_times(1.0, 2.5)  # T2 > 2 T1 is unphysical


# -- an unsafe witness must be STRICTLY unsafe --------------------------


def test_touching_the_threshold_is_not_an_upper_witness():
    """Safety is F >= FT, so a point AT the threshold is safe and bounds
    nothing. A fidelity that plateaus exactly at FT beyond the margin
    must therefore yield no finite upper bracket."""
    from qrobustness.core import iterative_margin

    FT = 0.999

    def fn(mu):
        # Strictly above FT near the origin, exactly FT further out, and
        # never below it: there is no strictly unsafe point anywhere.
        return max(FT, 1.0 - 0.01 * abs(float(mu)))

    res = iterative_margin(fn, 1.0, FT, margin_tol=1e-6, omega=(-5.0, 5.0))
    assert not np.isfinite(res.M_upper)
    assert res.reason_minus in ("boundary", "exhausted")
    assert res.reason_plus in ("boundary", "exhausted")


def test_eval_tol_requires_the_stated_numerical_criterion():
    """With an evaluation tolerance, an unsafe witness needs
    F_computed + eps_num < FT; a point inside the band is unresolved and
    moves neither end."""
    from qrobustness.core import iterative_margin

    FT = 0.999
    tol = 1e-9

    def fn(mu):
        # Dips just below FT, but by less than the evaluation tolerance.
        return FT + 1e-3 - 1e-3 * abs(float(mu)) if abs(mu) < 1.0 else FT - 5e-10

    res = iterative_margin(
        fn, 1.0, FT, margin_tol=1e-6, omega=(-5.0, 5.0), eval_tol=tol
    )
    assert not np.isfinite(res.M_upper)
    assert res.n_unresolved > 0

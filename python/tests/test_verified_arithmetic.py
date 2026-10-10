# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The verified arithmetic behind the diamond-norm bounds, checked against
exact rational values on inputs with cancellation and mixed magnitudes."""

from decimal import ROUND_FLOOR, Context, Decimal
from fractions import Fraction

import numpy as np
import pytest

from qrobustness.lindblad import (
    _chol_error_bound,
    _fro_upper,
    _partial_trace_specnorm_upper,
    _shift_diag_down,
    _sum_upward,
    _verify_psd,
)

CTX = Context(prec=80, rounding=ROUND_FLOOR)


def _sqrt_lower(q: Fraction) -> Fraction:
    """A rational lower bound on sqrt(q) (Decimal square root rounded down)."""
    d = CTX.divide(Decimal(q.numerator), Decimal(q.denominator))
    return Fraction(CTX.sqrt(d)) * (1 - Fraction(1, 10**70))


def _mixed(rng, shape):
    """Values of both signs over sixteen decades."""
    return rng.standard_normal(shape) * 10.0 ** rng.integers(-8, 9, size=shape)


@pytest.mark.parametrize("seed", range(5))
def test_sum_upward_bounds_the_exact_sum(seed):
    rng = np.random.default_rng(seed)
    v = np.abs(_mixed(rng, 2000))
    exact = sum(Fraction(float(x)) for x in v)
    assert Fraction(_sum_upward(v)) >= exact


@pytest.mark.parametrize("seed", range(5))
def test_partial_trace_bound_exceeds_the_exact_row_sums(seed):
    """The bound is at least the exact largest row sum of |Tr_out Y|, which
    bounds the spectral norm of the Hermitian partial trace."""
    rng = np.random.default_rng(seed)
    N = 4
    Z = _mixed(rng, (N * N, N * N)) + 1j * _mixed(rng, (N * N, N * N))
    Y = (Z + Z.conj().T) / 2
    Yr = Y.reshape(N, N, N, N)
    rows = []
    for i in range(N):
        total = Fraction(0)
        for j in range(N):
            re = sum(Fraction(float(Yr[k, i, k, j].real)) for k in range(N))
            im = sum(Fraction(float(Yr[k, i, k, j].imag)) for k in range(N))
            total += _sqrt_lower(re * re + im * im)
        rows.append(total)
    assert Fraction(_partial_trace_specnorm_upper(Y, N)) >= max(rows)


@pytest.mark.parametrize("seed", range(3))
def test_frobenius_bound_exceeds_the_exact_norm(seed):
    rng = np.random.default_rng(seed)
    A = _mixed(rng, (12, 12)) + 1j * _mixed(rng, (12, 12))
    sq = sum(
        Fraction(float(z.real)) ** 2 + Fraction(float(z.imag)) ** 2 for z in A.ravel()
    )
    assert Fraction(_fro_upper(A)) ** 2 >= sq


def test_cholesky_bound_keeps_rumps_underflow_term():
    """For the zero matrix only the underflow term n M eta remains, with
    M = 3(2n + max a_ii) and eta = 2^-1074."""
    n = 5
    exact = Fraction(n * 3 * (2 * n), 2**1074)
    assert Fraction(_chol_error_bound(np.zeros((n, n)))) >= exact


def test_non_finite_and_overflowing_inputs_are_not_verified():
    assert _shift_diag_down(float("inf"), 1.0) is None
    assert _shift_diag_down(1.0, float("nan")) is None
    big = np.diag([1e308, 1e308]).astype(complex)
    assert _verify_psd(big) is False

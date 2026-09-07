# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The solver-free diamond norm: an upper bound, and a tight one.

The check that matters is against the ANALYTIC value 2n of the paper's
lem:2n, not against cvxpy: agreeing with another implementation only
says the two agree. cvxpy is used as a second opinion where no closed
form is available, and there the requirement is one-sided -- the
solver-free value must not fall below it by more than the solver's own
accuracy, because a robustness constant must over-estimate.
"""

from __future__ import annotations

import numpy as np
import pytest

from qrobustness.lindblad import (
    diamond_norm_free,
    dissipator,
    hamiltonian_superop,
)

SZ = np.array([[1, 0], [0, -1]], dtype=complex)
# sigma_-, the amplitude-damping jump operator, in the same convention as
# scripts/run_open_amplitude_damping.py: one name, one operator.
SM = np.array([[0, 0], [1, 0]], dtype=complex)
I2 = np.eye(2)


def _sum_local(op, n):
    """Sum of the single-qubit dissipators for ``op`` on each of ``n`` qubits."""
    tot = None
    for q in range(n):
        ops = [I2] * n
        ops[q] = op
        V = ops[0]
        for o in ops[1:]:
            V = np.kron(V, o)
        D = dissipator(V)
        tot = D if tot is None else tot + D
    return tot


@pytest.mark.parametrize("n", [1, 2, 3])
def test_dephasing_family_is_exactly_2n(n):
    """||sum_q D[sigma_z^(q)]||_diamond = 2n exactly (lem:2n)."""
    r = diamond_norm_free(_sum_local(SZ, n))
    assert r.value >= 2 * n - 1e-9  # never under-estimates
    assert r.value == pytest.approx(2 * n, abs=1e-6)
    assert r.raw <= r.value
    assert r.status == "solver_free"
    assert r.feas_shift >= 0.0


def test_amplitude_damping_is_exactly_two():
    """||D[sigma_-]||_diamond = 2 exactly, and never under-estimated."""
    r = diamond_norm_free(dissipator(SM))
    assert r.value >= 2.0 - 1e-9
    assert r.value == pytest.approx(2.0, abs=1e-6)


def test_positive_homogeneity():
    """The norm is absolutely homogeneous, so scaling the generator scales
    it by the same factor -- what lets one SDP cover a whole structure."""
    D = dissipator(SM)
    a = diamond_norm_free(D).value
    b = diamond_norm_free(2.5 * D).value
    assert b == pytest.approx(2.5 * a, rel=1e-6)


def test_dominates_the_closed_form_start():
    """The refinement never makes the starting feasible point worse."""
    r = diamond_norm_free(dissipator(SM))
    assert r.gap >= -1e-12


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_agrees_with_cvxpy_where_available(seed):
    """Second opinion where no closed form exists: the solver-free value may
    exceed the SDP optimum but must never fall below it.

    Tightness is asserted on ``raw``, the floating objective at the
    repaired feasible point. ``value``/``value_certified`` is that same
    point bounded upward by Gershgorin discs with the rounding
    enclosures, which is rigorous and correspondingly looser; the
    one-sided check below holds for it too."""
    cp = pytest.importorskip("cvxpy")  # noqa: F841
    from qrobustness.lindblad import diamond_norm

    rng = np.random.default_rng(seed)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    S = dissipator(A)
    ref = diamond_norm(S).raw
    free = diamond_norm_free(S)
    # One-sided: over-estimating is safe, under-estimating is not. The
    # slack absorbs the SDP solver's own inaccuracy.
    assert free.raw >= ref * (1 - 1e-6)
    assert free.raw <= ref * 1.05
    # The rigorous bound sits above the tight one and never below the
    # optimum.
    assert free.value_certified >= free.raw
    assert free.value_certified >= ref * (1 - 1e-6)


def test_hamiltonian_superoperator_is_bounded_not_exact():
    """Documented limitation, asserted so it cannot regress silently.

    For a pure Hamiltonian superoperator the subgradient stalls on a
    degenerate spectrum and the bound is a few percent conservative.
    """
    cp = pytest.importorskip("cvxpy")  # noqa: F841
    from qrobustness.lindblad import diamond_norm

    rng = np.random.default_rng(0)
    H = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H = (H + H.conj().T) / 2
    S = hamiltonian_superop(H)
    ref = diamond_norm(S).raw
    free = diamond_norm_free(S)
    assert free.raw >= ref * (1 - 1e-6)
    assert free.raw <= ref * 1.10
    # Gershgorin adds its own conservatism on a non-diagonal partial
    # trace; it stays bounded, and it is the price of rigour.
    assert free.value_certified >= free.raw
    assert free.value_certified <= ref * 1.60

# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Implied margins from the algorithm-level worst-case fidelity bound of
Berberich, Fellner, Kosut and Holm, arXiv:2509.08481 (Theorem 2.1).

Their setting: an ideal circuit ``Ubar = Ubar_N ... Ubar_1`` with per-gate
coherent error unitaries ``U_{e,j} = exp(-i H_{e,j})``, interaction
Hamiltonians ``G_j = Vbar_j^dag H_{e,j} Vbar_j`` (``Vbar_j`` the partial
ideal products) and average ``G = (1/N) sum_j G_j``.  With
``||H_{e,j}||_2 <= delta`` and ``||G||_2 <= gamma delta``, Theorem 2.1
gives the worst-case bound (their Eq. 14; ``F_B = |tr(Ubar^dag U)|^2/d^2``
is the SQUARE of the trace fidelity used in this package)

    F_B >= 1 - X^2,   X = delta N ((N-1)/2 delta + gamma).

Specialisation to a piecewise-constant controller: each control interval
is one "gate" ``Ubar_k = exp(-i Delta H^(k))`` (N = tau gates), and a
structured perturbation ``delta(t) Hhat^(k)`` on interval ``k`` produces
the error unitary ``W_k = Ubar_k^dag U_k`` whose principal-log generator
satisfies the path-length bound

    ||H_{e,k}||_2 <= int_k |delta(t)| ||Hhat^(k)||_2 dt
                  <= Delta m ||Hhat^(k)||_2

for ANY measurable trajectory with ``||delta||_inf <= m`` (the geodesic
distance on the unitary group in the bi-invariant spectral metric is
bounded by the path length).  Hence their per-gate set-membership model
covers the same sup-norm trajectory class as the trajectory certificates
of this package, and the comparison is like for like.

Two uncertainty classes mirror the Kosut comparison:

* ``independent`` (trajectory class): only the per-gate norm bounds are
  used; ``||G|| <= (1/tau) sum_k ||H_{e,k}|| <= Delta m wbar`` with
  ``wbar = mean_k ||Hhat^(k)||_2`` (triangle inequality; unitary
  conjugation is norm-preserving), i.e. ``gamma = wbar/wmax``.
* ``systematic`` (constant class, ``delta(t) = mu``, ``|mu| <= m``):
  the first-order interaction average is exact and coherent,
  ``||G|| <= (1/tau)(m T w_avg + m^2 Delta^2 sum_k ||Hhat^(k)||_2^2 / 2)``,
  where ``w_avg`` is the Kosut interaction-picture average measure
  (closed form, ``kosut.uncertainty_rates``) and the quadratic term is
  the Magnus remainder ``||log W_k + i mu int_k Htil_I|| <=
  (1/2) mu^2 Delta^2 ||Hhat^(k)||_2^2`` (valid whenever
  ``m Delta ||Hhat^(k)||_2 < pi``, checked).

The nominal deficit is absorbed exactly as for the Kosut margin: their
bound controls the angle to the NOMINAL product,
``theta(Ubar, U) <= arcsin X``, and the Fubini--Study triangle
inequality requires ``arcsin X <= arccos FT - arccos F0``, i.e.
``X(m) <= sin(arccos FT - theta_0) =: s_T``.  In both classes ``X`` is
a quadratic ``a m^2 + b m``, solved by the cancellation-free root

    m = 2 s_T / (b + sqrt(b^2 + 4 a s_T)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .kosut import UncertaintyRates, _spectral_norm, uncertainty_rates

Array = np.ndarray
HList = Sequence[Array]

UNCERTAINTIES = ("independent", "systematic")

__all__ = ["BerberichMargin", "margin", "UNCERTAINTIES"]


@dataclass
class BerberichMargin:
    """Implied margin from arXiv:2509.08481 Theorem 2.1 (Eq. 14)."""

    m: float  #: certified sup-norm (or constant) budget
    uncertainty: str  #: 'independent' (trajectory) or 'systematic'
    s_T: float  #: angle budget sin(arccos FT - theta_0)
    a: float  #: quadratic coefficient of X(m) = a m^2 + b m
    b: float  #: linear coefficient
    w_max: float  #: max_k ||Hhat^(k)||_2
    w_mean: float  #: mean_k ||Hhat^(k)||_2
    gamma: float  #: effective gamma at the returned margin
    magnus_ok: bool  #: systematic only: m Delta w_max < pi holds
    vacuous: bool  #: True when the nominal angle exhausts the budget


def margin(
    H_list: HList,
    dH_list: HList,
    dt: float,
    FT: float,
    nominal_error: float = 0.0,
    uncertainty: str = "independent",
    rates: Optional[UncertaintyRates] = None,
) -> BerberichMargin:
    """Largest budget certified by their Theorem 2.1 for this controller.

    ``H_list``/``dH_list`` are the per-interval nominal Hamiltonians and
    perturbation structures, ``dt`` the interval length.  For
    ``uncertainty='systematic'`` the Kosut interaction-picture measures
    are needed; pass precomputed ``rates`` to avoid recomputation.
    """
    if uncertainty not in UNCERTAINTIES:
        raise ValueError(
            f"Unknown uncertainty={uncertainty!r}; expected one of {UNCERTAINTIES}"
        )
    if not (0.0 < FT < 1.0):
        raise ValueError("FT must satisfy 0 < FT < 1")
    tau = len(H_list)
    w = np.array([_spectral_norm(np.asarray(Hh)) for Hh in dH_list])
    w_max = float(np.max(w))
    w_mean = float(np.mean(w))

    # Reject rather than clamp, matching kosut.effective_threshold. Without
    # this, nominal_error > 2 makes arccos return nan, budget nan, the
    # `budget <= 0` test False (NaN compares false), and the function returns
    # a nan margin with vacuous=False -- a silently wrong answer.
    if not 0.0 <= nominal_error <= 1.0:
        raise ValueError("nominal_error must satisfy 0 <= nominal_error <= 1")
    theta_0 = float(np.arccos(min(1.0, 1.0 - nominal_error)))
    budget = float(np.arccos(FT)) - theta_0
    if budget <= 0.0 or w_max <= 0.0:
        return BerberichMargin(
            m=0.0,
            uncertainty=uncertainty,
            s_T=0.0,
            a=np.nan,
            b=np.nan,
            w_max=w_max,
            w_mean=w_mean,
            gamma=np.nan,
            magnus_ok=True,
            vacuous=True,
        )
    s_T = float(np.sin(budget))

    # X(m) = tau * ((tau-1)/2 * delta(m)^2 + Gbar(m)),
    # delta(m) = dt * m * w_max.
    a_depth = tau * (tau - 1) / 2.0 * (dt * w_max) ** 2
    if uncertainty == "independent":
        a = a_depth
        b = tau * dt * w_mean
    else:  # systematic
        if rates is None:
            rates = uncertainty_rates(H_list, dH_list, dt)
        T = tau * dt
        a = a_depth + dt**2 * float(np.sum(w**2)) / 2.0
        b = T * rates.w_avg
    m = 2.0 * s_T / (b + np.sqrt(b * b + 4.0 * a * s_T))
    delta = dt * m * w_max
    gamma = (b * m + (a - a_depth) * m * m) / (tau * delta) if delta > 0 else np.nan
    return BerberichMargin(
        m=float(m),
        uncertainty=uncertainty,
        s_T=s_T,
        a=float(a),
        b=float(b),
        w_max=w_max,
        w_mean=w_mean,
        gamma=float(gamma),
        magnus_ok=bool(m * dt * w_max < np.pi),
        vacuous=False,
    )

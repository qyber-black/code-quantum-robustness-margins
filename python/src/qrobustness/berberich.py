# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Margins from the bound of Berberich et al., arXiv:2509.08481, Theorem 2.1.

Each control interval is one gate. The per-interval error generator is
bounded by the spectral-norm path length Delta m ||Hhat^(k)||_2.
``uncertainty='independent'`` returns the trajectory margin M^B_tv.
``uncertainty='systematic'`` returns the constant-perturbation margin M^B.
The nominal error is absorbed angularly, as for the Kosut margin M^K.
Their F_B is the square of the trace fidelity used in this package.
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
    """Margin implied by the Berberich et al. bound, with the bound's coefficients."""

    m: float  #: certified sup-norm (or constant) budget
    uncertainty: str  #: 'independent' (trajectory) or 'systematic'
    s_T: float  #: angle budget sin(arccos FT - theta_0)
    a: float  #: quadratic coefficient of X(m) = a m^2 + b m
    b: float  #: linear coefficient
    w_max: float  #: max_k ||Hhat^(k)||_2
    w_mean: float  #: mean_k ||Hhat^(k)||_2
    gamma: float  #: effective gamma at the returned margin
    magnus_ok: bool  #: systematic only: m Delta w_max < pi holds
    vacuous: bool  #: True if the nominal angle exhausts the budget


def margin(
    H_list: HList,
    dH_list: HList,
    dt: float,
    FT: float,
    nominal_error: float = 0.0,
    uncertainty: str = "independent",
    rates: Optional[UncertaintyRates] = None,
) -> BerberichMargin:
    """Largest budget m certified by the Berberich et al. bound for one controller.

    Parameters
    ----------
    H_list : sequence of (N, N) arrays
        Nominal Hamiltonians H^(k), one per control interval.
    dH_list : sequence of (N, N) arrays
        Perturbation structures Hhat^(k), one per interval.
    dt : float
        Interval length Delta.
    FT : float
        Fidelity threshold F_T, 0 < F_T < 1.
    nominal_error : float
        Nominal error 1 - F_0, in [0, 1].
    uncertainty : str
        ``'independent'`` (trajectory class, M^B_tv) or ``'systematic'``
        (constant class, M^B).
    rates : UncertaintyRates, optional
        Precomputed Kosut measures; used only for ``'systematic'``.

    Returns
    -------
    BerberichMargin
        ``m`` is the margin; ``vacuous`` is True (and ``m = 0``) when the
        nominal angle theta_0 exhausts arccos F_T. For ``'systematic'`` the
        result is valid only if ``magnus_ok``.
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

    # Reject rather than clamp: an out-of-range value would give a NaN margin.
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

    # X(m) = tau ((tau-1)/2 delta(m)^2 + ||G||(m)), delta(m) = dt m w_max.
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
    magnus_ok = bool(m * dt * w_max < np.pi)
    if uncertainty == "systematic" and not magnus_ok:
        raise ValueError("Magnus condition failed for systematic uncertainty")
    return BerberichMargin(
        m=float(m),
        uncertainty=uncertainty,
        s_T=s_T,
        a=float(a),
        b=float(b),
        w_max=w_max,
        w_mean=w_mean,
        gamma=float(gamma),
        magnus_ok=magnus_ok,
        vacuous=False,
    )

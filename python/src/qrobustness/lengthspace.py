# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared pieces of the gauge-budget certificates.

The module holds the traceless per-interval Gram matrices, the angle budget
arccos F_T - arccos F_0, and the quadratic path gauge :class:`PathGauge`
(``Delta sum_k sqrt(x^T G^(k) x)``), with its box bound and certified
inradius. It also refines interval lists onto a finer grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

# Re-exported from core.
from .core import traceless

Array = np.ndarray
HList = Sequence[Array]

__all__ = [
    "traceless",
    "interval_grams",
    "angle_budget",
    "margin_from",
    "refine",
    "PathGauge",
]


def interval_grams(
    Hhat_lists: Sequence[HList],
    make_traceless: bool = False,
    normalise: bool = False,
) -> List[Array]:
    """Per-interval Gram matrices ``G^(k)_ij = Re Tr(A_i^(k)' A_j^(k))``.

    With ``make_traceless`` these are P^(k) (C_joint); with both flags set they
    are Q^(k) = P^(k)/N (C^stat_FS and the trajectory gauge).

    Parameters
    ----------
    Hhat_lists : sequence of p lists of tau ndarrays, shape (N, N)
        Per-structure, per-interval structures.
    make_traceless :
        Use the traceless parts ``A = Hbar`` instead of ``A = Hhat``.
    normalise :
        Divide by the Hilbert-space dimension N.

    Returns
    -------
    list of tau ndarrays, shape (p, p)
    """
    p_structs = len(Hhat_lists)
    tau = len(Hhat_lists[0])
    N = np.asarray(Hhat_lists[0][0]).shape[0]
    grams: List[Array] = []
    for k in range(tau):
        A = [np.asarray(Hhat_lists[j][k]) for j in range(p_structs)]
        if make_traceless:
            A = [traceless(H) for H in A]
        G = np.empty((p_structs, p_structs))
        for i in range(p_structs):
            for j in range(i, p_structs):
                g = float(np.real(np.trace(A[i].conj().T @ A[j])))
                if normalise:
                    g /= N
                G[i, j] = G[j, i] = g
        grams.append(G)
    return grams


def angle_budget(F0: float, FT: float) -> float:
    """Angle budget ``arccos F_T - arccos F_0`` (``F0`` clipped at 1)."""
    return float(np.arccos(FT) - np.arccos(min(F0, 1.0)))


def margin_from(budget: float, speed: float) -> float:
    """``budget / speed``, or ``inf`` when ``speed <= 0``."""
    return budget / speed if speed > 0 else float("inf")


def refine(H_list: HList, Hhat_list: HList, q: int) -> Tuple[list, list]:
    """Split each control interval into ``q`` equal sub-intervals.

    Parameters
    ----------
    H_list, Hhat_list : lists of tau ndarrays, shape (N, N)
        Interval Hamiltonians and structures; must have equal length.
    q :
        Positive integer refinement factor.

    Returns
    -------
    H_ref, Hhat_ref : lists of q*tau ndarrays
        Each entry repeated ``q`` times; the caller divides ``dt`` by ``q``.
    """
    if not isinstance(q, (int, np.integer)) or q < 1:
        raise ValueError(f"q must be a positive integer, got {q!r}")
    if len(H_list) != len(Hhat_list):
        raise ValueError(
            f"lists must have equal length, got {len(H_list)} and {len(Hhat_list)}"
        )
    H_ref = [np.asarray(H) for H in H_list for _ in range(q)]
    Hhat_ref = [np.asarray(H) for H in Hhat_list for _ in range(q)]
    return H_ref, Hhat_ref


@dataclass
class PathGauge:
    """Quadratic path gauge ``C(x) = Delta sum_k sqrt(x^T G^(k) x)``, a seminorm.

    Attributes
    ----------
    grams : list of tau ndarrays, shape (p, p)
        Per-interval Gram matrices G^(k) (P^(k) or Q^(k)).
    dt : float
        Interval length Delta.
    """

    grams: List[Array]
    dt: float

    def C(self, x) -> float:
        """Gauge value C(x) for a displacement ``x`` of shape (p,)."""
        x = np.asarray(x, dtype=float)
        return self.dt * float(sum(np.sqrt(max(0.0, x @ G @ x)) for G in self.grams))

    def C_box(self, m) -> float:
        """Upper bound on ``max C(x)`` over the box ``|x_j| <= m_j``.

        Each interval term is maximised over the ``2^p`` sign vertices
        independently, so the result is an upper bound, attained when one
        sign vertex is worst on every interval.
        """
        m = np.asarray(m, dtype=float)
        p = m.size
        signs = np.array(
            [[1 if (v >> j) & 1 else -1 for j in range(p)] for v in range(2**p)],
            dtype=float,
        )
        z = signs * m
        total = 0.0
        for G in self.grams:
            total += float(np.sqrt(max(0.0, np.max(np.einsum("vi,ij,vj->v", z, G, z)))))
        return self.dt * total

    def radius(self, d, budget: float) -> float:
        """Radius ``budget / C(d)`` along direction ``d`` (``inf`` if ``C(d) = 0``)."""
        return margin_from(budget, self.C(d))

    def alpha_cs(self) -> float:
        """Upper bound ``sqrt(t_f lambda_max(Delta sum_k G^(k)))`` on ``max_{|d|_2 = 1} C(d)``.

        From Cauchy-Schwarz over intervals; possibly conservative.
        """
        Gsum = self.dt * np.sum(self.grams, axis=0)
        tf = self.dt * len(self.grams)
        lam = float(np.max(np.linalg.eigvalsh((Gsum + Gsum.T) / 2)))
        return float(np.sqrt(max(0.0, tf * lam)))

    def inradius(self, budget: float) -> float:
        """Certified Euclidean inradius ``budget / alpha_cs()`` of ``C(x) <= budget``."""
        return margin_from(budget, self.alpha_cs())

# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared core of the gauge--budget certificates (the master lemma).

Every certificate of the paper is one lemma instantiated with a gauge
and a budget: a positively homogeneous functional ``c(x)`` of the
displacement is compared against a fidelity or angle budget, and
safety follows by metric contraction or Lipschitz continuation.  This
module holds the computations shared by the instantiations:

* the traceless projection and the per-interval Gram matrices from
  which every quadratic path gauge ``dt sum_k sqrt(x^T G^(k) x)`` is
  built (traceless Frobenius grams for the joint Lipschitz gauge,
  normalised grams for the Choi-angular and trajectory gauges);
* the angle budget ``arccos F_T - arccos F_0`` of the metric case;
* :class:`PathGauge`, the quadratic path gauge with its sign-vertex
  box variant, budget-to-radius conversion, and certified
  Cauchy--Schwarz inradius;
* grid refinement of interval lists (sub-interval trajectories).

The scalar diamond-norm route of :mod:`qrobustness.lindblad` follows
the same pattern with the diamond norm in place of the Frobenius
norm; a quadratic Gram representation does not exist there (the
diamond norm of a combination needs an SDP per direction), so it is
not routed through this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

# One definition, in core, which validates that the structure is square and
# Hermitian: every gauge here is built on centred structures, so what counts
# as a valid structure is decided in one place. Re-exported for importers of
# this module.
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
    """Per-interval Gram matrices of the structure lists.

    ``G^(k)_ij = Re Tr(A_i^(k)' A_j^(k))`` with ``A = Hhat`` (raw) or
    the traceless part (``make_traceless``), divided by the Hilbert
    dimension ``N`` when ``normalise`` is set.  Traceless grams build
    the joint Lipschitz gauge ``C_joint``; traceless normalised grams
    build the Choi-angular and trajectory gauges (the exact Choi
    speed).  Raw grams are retained for callers that need the
    uncentred Frobenius geometry.
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
    """The angle budget ``arccos F_T - arccos F_0`` of the metric case."""
    return float(np.arccos(FT) - np.arccos(min(F0, 1.0)))


def margin_from(budget: float, speed: float) -> float:
    """``budget / speed`` with the zero-speed convention ``inf``."""
    return budget / speed if speed > 0 else float("inf")


def refine(H_list: HList, Hhat_list: HList, q: int) -> Tuple[list, list]:
    """Split every control interval into ``q`` equal sub-intervals.

    Returns the refined lists; the caller scales its time step by
    ``1/q``.

    ``q`` must be a positive integer and the two lists must have equal
    length. Unequal inputs would give refined lists that no longer
    correspond, and a caller zipping them would propagate a wrong grid
    rather than fail.
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
    """A quadratic path gauge ``C(x) = dt sum_k sqrt(x^T G^(k) x)``.

    Positively homogeneous and convex (a seminorm); the shared shape
    of the joint Lipschitz gauge (traceless grams) and the Choi-angular /
    trajectory gauges (traceless normalised grams).
    """

    grams: List[Array]
    dt: float

    def C(self, x) -> float:
        """The gauge value; positively homogeneous of degree one."""
        x = np.asarray(x, dtype=float)
        return self.dt * float(sum(np.sqrt(max(0.0, x @ G @ x)) for G in self.grams))

    def C_box(self, m) -> float:
        """Worst case of ``C`` over the box ``|x_j| <= m_j``.

        A convex function of ``x`` is maximised over the box at a
        sign vertex; enumerated exactly (``2^p`` vertices per call,
        vectorised over intervals).
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
        """Certified radius ``budget / C(d)`` along direction ``d``."""
        return margin_from(budget, self.C(d))

    def alpha_cs(self) -> float:
        """Certified upper bound on ``max_{|d|_2 = 1} C(d)``.

        Cauchy--Schwarz over intervals:
        ``C(d) <= sqrt(t_f * d' (sum_k dt G^(k)) d)``, so the sphere
        maximum is bounded by
        ``sqrt(t_f * lambda_max(sum_k dt G^(k)))`` -- one eigenvalue
        computation, a certified (possibly conservative) bound; a
        sphere-sampled value is only an estimate.
        """
        Gsum = self.dt * np.sum(self.grams, axis=0)
        tf = self.dt * len(self.grams)
        lam = float(np.max(np.linalg.eigvalsh((Gsum + Gsum.T) / 2)))
        return float(np.sqrt(max(0.0, tf * lam)))

    def inradius(self, budget: float) -> float:
        """Certified Euclidean inradius ``budget / alpha_cs()``."""
        return margin_from(budget, self.alpha_cs())

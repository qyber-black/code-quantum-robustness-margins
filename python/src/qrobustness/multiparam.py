# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-parameter certified robustness margins.

For p simultaneous structured perturbations

    Htil^(k)(mu) = H^(k) + sum_j (mu_j - mu0_j) Hhat_j^(k),

per-parameter Lipschitz constants L_j = B_T C_j certify the weighted
cross-polytope

    P_nu = { mu : sum_j L_j |mu_j - nu_j| <= F_nu - F_T }

around any safe point nu, at no additional fidelity evaluations, and the
scalar certified iteration of :func:`qrobustness.iterative_margin` applies
along any ray with the directional constant L(d) = sum_j L_j |d_j|.  The
scalar case p = 1 is recovered exactly.

The cross-polytope is the separable relaxation, and this module also builds
the two sharper regions it is contained in:

* :class:`JointGauge`, the combined-structure Frobenius gauge
  ``C_joint(x) = sum_k Delta ||sum_j x_j Hbar_j^(k)||_F``, which keeps the
  cancellations between structures that the parameter-wise triangle
  inequality discards, and gives the sharper directional constant
  ``L_dir(d) = B_T C_joint(d) <= sum_j L_j |d_j|``;
* :class:`AngularGauge`, the static Choi-angular gauge, which contains the
  combined-structure region in turn by the dominance theorem.

Both are free: neither costs a fidelity evaluation beyond the structure
constants. On the shipped ensemble the gauge region's certified Euclidean
inradius exceeds the polytope's in 61 of 61 controllers.

The scalar iteration is reused, not reimplemented, so every directional
margin inherits its error control (``margin_tol`` brackets, ``certificate``
classes) unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .core import (
    Array,
    HList,
    MarginResult,
    gate_fidelity,
    iterative_margin,
    lipschitz_constant,
    propagator,
    structure_constant,
)
from .lengthspace import PathGauge, angle_budget, interval_grams, margin_from

__all__ = [
    "structure_constants",
    "JointGauge",
    "joint_gauge",
    "AngularGauge",
    "angular_gauge",
    "SafePolytope",
    "safe_polytope",
    "make_multiparam_fidelity_fn",
    "make_ray_fn",
    "directional_margin",
    "SafeUnion",
    "axis_directions",
    "diagonal_directions",
    "sphere_directions",
]


def structure_constants(
    specs: Sequence[tuple],
    dt: float,
    tau: int,
    FT: float,
    N: int,
) -> Tuple[Array, Array]:
    """Per-parameter constants ``(C, L)`` with ``L_j = B_T C_j``.

    Each spec is ``('drift', Hhat)`` or ``('control', Hhat, controls)``,
    exactly the cases of :func:`qrobustness.structure_constant`.
    """
    C = []
    for spec in specs:
        kind = spec[0]
        if kind == "drift":
            C.append(structure_constant("drift", spec[1], dt, tau))
        elif kind == "control":
            C.append(structure_constant("control", spec[1], dt, tau, spec[2]))
        else:
            raise ValueError(f"Unknown structure kind {kind!r}")
    C = np.asarray(C, dtype=float)
    L = np.array([lipschitz_constant(FT, N, c) for c in C])
    return C, L


@dataclass
class SafePolytope:
    """Certified cross-polytope ``sum_j L_j |mu_j - centre_j| <= surplus``.

    Every point of the polytope satisfies ``F >= F_T``; interior points lie
    in the connected safe component of the centre.
    """

    centre: Array
    L: Array
    surplus: float  #: F(centre) - F_T

    @property
    def axis_radii(self) -> Array:
        """Extent along each parameter axis (cross-polytope vertices)."""
        return self.surplus / self.L

    @property
    def inradius_linf(self) -> float:
        """Half-width of the largest certified box (uniform per axis)."""
        return self.surplus / float(np.sum(self.L))

    @property
    def inradius_l2(self) -> float:
        """Radius of the largest certified Euclidean ball."""
        return self.surplus / float(np.linalg.norm(self.L))

    def contains(self, mu: Array) -> bool:
        d = np.abs(np.asarray(mu, dtype=float) - self.centre)
        return bool(np.dot(self.L, d) <= self.surplus)

    def boundary_point(self, d: Array) -> Array:
        """The boundary point of the polytope in direction ``d``."""
        d = np.asarray(d, dtype=float)
        s = self.surplus / float(np.dot(self.L, np.abs(d)))
        return self.centre + s * d


def safe_polytope(centre: Array, L: Array, F: float, FT: float) -> SafePolytope:
    """The certified free region around a nominal point.

    The weighted cross-polytope ``sum_j L_j |x_j - centre_j| <= F - FT``:
    every point inside meets the threshold, at no off-nominal fidelity
    evaluations. Requires a strict surplus, since a nominal point already at
    the threshold certifies nothing.
    """
    if not (F > FT):
        raise ValueError("Require F > FT at the centre")
    return SafePolytope(
        centre=np.asarray(centre, dtype=float),
        L=np.asarray(L, dtype=float),
        surplus=float(F - FT),
    )


def make_multiparam_fidelity_fn(
    H0: Array,
    H1: Array,
    H2: Array,
    u1: Array,
    u2: Array,
    Uf: Array,
    dt: float,
    structures: Sequence[str] = ("H0", "H1", "H2"),
) -> Callable[[Array], float]:
    """Fidelity as a function of the joint perturbation vector.

    ``structures`` selects which relative-amplitude perturbations the
    components of ``mu`` act on, in order.  With the default all three, the
    perturbed interval Hamiltonians are

        H^(k)(mu) = H0 (1+mu_0) + u1_k H1 (1+mu_1) + u2_k H2 (1+mu_2).
    """
    u1 = np.asarray(u1, dtype=float).ravel()
    u2 = np.asarray(u2, dtype=float).ravel()
    tau = u1.size
    idx = {s: i for i, s in enumerate(structures)}

    def fn(mu: Array) -> float:
        mu = np.atleast_1d(np.asarray(mu, dtype=float))
        d0 = mu[idx["H0"]] if "H0" in idx else 0.0
        d1 = mu[idx["H1"]] if "H1" in idx else 0.0
        d2 = mu[idx["H2"]] if "H2" in idx else 0.0
        H_list = [
            H0 * (1.0 + d0) + u1[k] * H1 * (1.0 + d1) + u2[k] * H2 * (1.0 + d2)
            for k in range(tau)
        ]
        return gate_fidelity(propagator(H_list, dt), Uf)

    return fn


def make_ray_fn(
    fidelity_fn: Callable[[Array], float],
    mu0: Array,
    d: Array,
) -> Callable[[float], float]:
    """Restrict a multi-parameter fidelity function to the ray ``mu0 + s d``."""
    mu0 = np.asarray(mu0, dtype=float)
    d = np.asarray(d, dtype=float)

    def ray(s: float) -> float:
        return float(fidelity_fn(mu0 + s * d))

    return ray


def directional_margin(
    fidelity_fn: Callable[[Array], float],
    L: Array,
    FT: float,
    d: Array,
    mu0: Optional[Array] = None,
    L_dir: Optional[float] = None,
    angular_gauge: Optional["AngularGauge"] = None,
    **kwargs,
) -> MarginResult:
    """Certified margin along the ray ``mu0 + s d``.

    Calls :func:`qrobustness.iterative_margin` with the directional
    Lipschitz constant.  Pass ``L_dir`` explicitly to use the sharper
    joint-gauge constant ``B_T C_joint(d)`` of
    :meth:`JointGauge.L_dir` (Theorem gauge); otherwise the separable
    relaxation ``L(d) = sum_j L_j |d_j|`` is used.  Pass
    ``angular_gauge`` (an :class:`AngularGauge`) to step and certify
    with the Choi-angular safe radius
    ``(arccos FT - arccos F)/C_FS(d)`` instead of the Lipschitz
    surplus rule -- by full-gauge dominance the angular step is never
    smaller, so the same crossing is resolved with fewer evaluations;
    ``L_dir`` still feeds the aggressive-probe floor and is required
    positive, with one exception: a direction in which the gauge
    vanishes changes no relevant dynamics, so the whole admissible ray
    is certified and no division is performed.  All options, including ``margin_tol``, pass through
    unchanged.  The returned margins are in units of ``s``: the
    certified parameter excursion is ``M * d``.
    """
    d = np.asarray(d, dtype=float)
    L = np.asarray(L, dtype=float)
    if mu0 is None:
        mu0 = np.zeros_like(d)
    if L_dir is None:
        L_dir = float(np.dot(L, np.abs(d)))
    if L_dir == 0.0 and (angular_gauge is None or angular_gauge.C(d) == 0.0):
        # Zero gauge: the combined structure vanishes on every interval
        # in this direction, so the perturbation changes nothing the
        # certificate depends on and the whole admissible ray is
        # certified. Dividing the budget by it is the bug this branch
        # exists to prevent -- the scalar iteration would reject L = 0
        # outright, which reports a failure where the answer is "no
        # bound needed".
        # ``omega`` bounds the RAY parameter s, whose centre is s = 0,
        # so the certified reach is the distance to whichever end of the
        # admissible interval is nearer (infinite when unbounded).
        s_lo, s_hi = kwargs.get("omega", (-np.inf, np.inf))
        M_zero = min(
            float("inf") if not np.isfinite(s_lo) else abs(float(s_lo)),
            float("inf") if not np.isfinite(s_hi) else abs(float(s_hi)),
        )
        return MarginResult(
            M_minus=M_zero,
            M_plus=M_zero,
            M=M_zero,
            converged_minus=True,
            converged_plus=True,
            mu_minus=-M_zero,
            mu_plus=M_zero,
            method="zero_gauge",
            status_minus="zero_gauge",
            status_plus="zero_gauge",
            certificate="segment",
            reason_minus="zero_gauge",
            reason_plus="zero_gauge",
        )
    if angular_gauge is not None and "safe_radius_fn" not in kwargs:
        C_FS = angular_gauge.C(d)
        acFT = float(np.arccos(FT))
        kwargs["safe_radius_fn"] = lambda F: margin_from(
            acFT - float(np.arccos(min(F, 1.0))), C_FS
        )
    return iterative_margin(make_ray_fn(fidelity_fn, mu0, d), L_dir, FT, **kwargs)


@dataclass
class SafeUnion:
    """Certified union of safe polytopes with per-centre provenance."""

    L: Array
    polytopes: List[SafePolytope] = field(default_factory=list)
    provenance: List[str] = field(default_factory=list)

    def add(self, centre: Array, F: float, FT: float, note: str = "") -> None:
        self.polytopes.append(safe_polytope(centre, self.L, F, FT))
        self.provenance.append(note)

    def contains(self, mu: Array) -> bool:
        return any(P.contains(mu) for P in self.polytopes)


def axis_directions(p: int) -> Array:
    """The ``2p`` signed coordinate directions (unit in every norm)."""
    eye = np.eye(p)
    return np.vstack([eye, -eye])


def diagonal_directions(p: int) -> Array:
    """All ``2^p`` diagonal directions, Euclidean-normalised."""
    from itertools import product

    signs = np.array(list(product((-1.0, 1.0), repeat=p)))
    return signs / np.sqrt(p)


def sphere_directions(p: int, n: int, seed: Optional[int] = None) -> Array:
    """``n`` quasi-uniform Euclidean unit directions."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, p))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


@dataclass
class JointGauge(PathGauge):
    """The combined-structure gauge ``C_joint`` and its certified region.

    ``C_joint(x) = sum_k Delta ||sum_j x_j Hbar_j^(k)||_F``, with ``Hbar``
    the traceless part of the structure, is a
    homogeneous convex gauge on parameter displacements: applying the
    scalar sensitivity bound along the ray ``nu + s x`` with the
    combined structure ``sum_j x_j Hhat_j^(k)`` certifies the region
    ``B_T C_joint(x) <= F_nu - F_T``, which captures cancellations and
    near-dependencies between structures that the parameter-wise
    triangle inequality (the weighted cross-polytope) discards, and
    detects exact null directions.  ``C_joint(x)^2 = sum_k Delta^2 *
    (x^T P^(k) x)`` with per-interval Gram matrices
    ``P^(k)_ij = Re Tr(Hbar_i^(k)' Hbar_j^(k))``, so the region is
    second-order-cone representable.

    The gauge arithmetic (``C``, the Cauchy--Schwarz sphere bound)
    lives in :class:`qrobustness.lengthspace.PathGauge` (master-lemma
    case (b), traceless Frobenius grams); this class adds the
    Lipschitz-constant conversion via ``B_T``.
    """

    def L_dir(self, d, FT: float, N: int) -> float:
        """Sharp directional Lipschitz constant ``B_T C_joint(d)``.

        Never exceeds the separable ``sum_j L_j |d_j|``; use it as the
        scalar constant when iterating along the ray ``d``.
        """
        from .core import lipschitz_constant

        return lipschitz_constant(FT, N, self.C(d))

    def contains(self, x, surplus: float, FT: float, N: int) -> bool:
        from .core import lipschitz_constant

        return lipschitz_constant(FT, N, self.C(x)) <= surplus

    def boundary_radius(self, d, surplus: float, FT: float, N: int) -> float:
        """Certified radius along direction ``d`` (free, no evaluations)."""
        L = self.L_dir(d, FT, N)
        return surplus / L if L > 0 else float("inf")

    def alpha2_certified(self) -> float:
        """Certified upper bound on ``max_{|d|_2 = 1} C_joint(d)``.

        The Cauchy--Schwarz sphere bound of
        :meth:`~qrobustness.lengthspace.PathGauge.alpha_cs`.
        """
        return self.alpha_cs()

    def inradius_certified(self, surplus: float, FT: float, N: int) -> float:
        """Certified Euclidean inradius of the gauge region."""
        from .core import lipschitz_constant

        L = lipschitz_constant(FT, N, self.alpha2_certified())
        return surplus / L if L > 0 else float("inf")


def joint_gauge(Hhat_lists: Sequence[HList], dt: float) -> JointGauge:
    """Build the combined-structure gauge from per-structure interval lists.

    The structures are centred to their traceless parts first, exactly as
    :func:`qrobustness.structure_constant` centres the scalar constant: the
    trace part of a perturbation contributes only a global phase, to which
    the trace-amplitude fidelity is invariant.  Centring therefore leaves the
    certified region valid while shrinking the gauge, and it makes the static
    Choi-angular identity ``C_FS_stat(x) = C_joint(x)/sqrt(N)`` exact rather
    than an inequality.  For the traceless case-study structures this changes
    nothing.
    """
    return JointGauge(grams=interval_grams(Hhat_lists, make_traceless=True), dt=dt)


@dataclass
class AngularGauge(PathGauge):
    """The static Choi-angular gauge: the strongest zero-evaluation
    static region of the paper.

    A constant displacement is a trajectory, so the exact Choi-speed
    certificate applies with the signed static gauge
    ``C_FS(x) = dt sum_k sqrt(x^T Q^(k) x)``, where
    ``Q^(k)_ij = Re Tr(Hbar_i^(k) Hbar_j^(k))/N`` are the traceless
    interval Gram matrices, and the budget is the fidelity ANGLE
    ``arccos F_T - arccos F_nu``.  The region contains the joint
    Lipschitz gauge region of :class:`JointGauge` (full-gauge
    dominance: ``C_FS(x) <= C_joint(x)/sqrt(N)`` and the angle budget
    dominates ``surplus/sqrt(1-F_T^2)``), is insensitive to identity
    components (pure global phase), convex, centrally symmetric about
    the centre before intersection with the admissible domain, and
    second-order-cone representable.

    The gauge arithmetic lives in
    :class:`qrobustness.lengthspace.PathGauge` (master-lemma case
    (a-ii), traceless normalised grams); this class binds the angle
    budget.
    """

    @staticmethod
    def budget(F_nu: float, FT: float) -> float:
        return angle_budget(F_nu, FT)

    def contains(self, x, F_nu: float, FT: float) -> bool:
        return self.C(x) <= self.budget(F_nu, FT)

    def boundary_radius(self, d, F_nu: float, FT: float) -> float:
        return self.radius(d, self.budget(F_nu, FT))

    def inradius_certified(self, F_nu: float, FT: float) -> float:
        """Certified Euclidean inradius via the Cauchy-Schwarz bound
        ``C_FS(d) <= sqrt(t_f * d^T (sum_k dt Q^(k)) d)``."""
        return self.inradius(self.budget(F_nu, FT))


def angular_gauge(Hhat_lists: Sequence[HList], dt: float) -> AngularGauge:
    """Build the static Choi-angular gauge (traceless interval Grams)."""
    return AngularGauge(
        grams=interval_grams(Hhat_lists, make_traceless=True, normalise=True), dt=dt
    )

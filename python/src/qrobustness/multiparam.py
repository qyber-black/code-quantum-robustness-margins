# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Certified robustness regions and directional margins for p simultaneous structures.

The per-parameter constants are L_j = B_T C_j. The separable cross-polytope
is sum_j L_j |mu_j - nu_j| <= F_nu - F_T. The combined-structure gauge
C_joint is :class:`JointGauge`. The static angular gauge C^stat_FS is
:class:`AngularGauge`. Directional margins along rays go through
:func:`qrobustness.iterative_margin`.
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
    """Per-parameter structure constants C_j and Lipschitz constants L_j = B_T C_j.

    Parameters
    ----------
    specs :
        One spec per parameter: ``('drift', Hhat)`` or
        ``('control', Hhat, controls)``, as in
        :func:`qrobustness.structure_constant`.
    dt :
        Interval length Delta.
    tau :
        Number of control intervals.
    FT :
        Fidelity threshold F_T.
    N :
        Hilbert-space dimension.

    Returns
    -------
    C : ndarray, shape (p,)
        Structure constants C_j.
    L : ndarray, shape (p,)
        Lipschitz constants L_j.
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

    Every point of the polytope has ``F >= F_T``.

    Attributes
    ----------
    centre : ndarray, shape (p,)
        Safe centre nu.
    L : ndarray, shape (p,)
        Per-parameter Lipschitz constants L_j.
    surplus : float
        F(centre) - F_T.
    """

    centre: Array
    L: Array
    surplus: float  #: F(centre) - F_T

    @property
    def axis_radii(self) -> Array:
        """Reach along each parameter axis (vertices of the cross-polytope)."""
        return self.surplus / self.L

    @property
    def inradius_linf(self) -> float:
        """Half-width of the largest certified box (the same on every axis)."""
        return self.surplus / float(np.sum(self.L))

    @property
    def inradius_l2(self) -> float:
        """Radius of the largest certified Euclidean ball."""
        return self.surplus / float(np.linalg.norm(self.L))

    def contains(self, mu: Array) -> bool:
        """True if ``mu`` lies in the polytope."""
        d = np.abs(np.asarray(mu, dtype=float) - self.centre)
        return bool(np.dot(self.L, d) <= self.surplus)

    def boundary_point(self, d: Array) -> Array:
        """Boundary point of the polytope along direction ``d``."""
        d = np.asarray(d, dtype=float)
        s = self.surplus / float(np.dot(self.L, np.abs(d)))
        return self.centre + s * d


def safe_polytope(centre: Array, L: Array, F: float, FT: float) -> SafePolytope:
    """Certified cross-polytope ``sum_j L_j |x_j - centre_j| <= F - F_T`` about a safe point.

    Parameters
    ----------
    centre : array_like, shape (p,)
        Safe point nu.
    L : array_like, shape (p,)
        Per-parameter Lipschitz constants L_j.
    F :
        Fidelity at ``centre``; must exceed ``FT``.
    FT :
        Fidelity threshold F_T.

    Returns
    -------
    SafePolytope
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
    """Gate fidelity as a function of a relative-amplitude perturbation vector ``mu``.

    With all three structures the interval Hamiltonians are
    ``H^(k)(mu) = H0 (1+mu_0) + u1_k H1 (1+mu_1) + u2_k H2 (1+mu_2)``.

    Parameters
    ----------
    H0, H1, H2 : ndarray, shape (N, N)
        Drift and two control Hamiltonians.
    u1, u2 : array_like, shape (tau,)
        Piecewise-constant control amplitudes.
    Uf : ndarray, shape (N, N)
        Target unitary.
    dt :
        Interval length Delta.
    structures :
        Which of ``"H0"``, ``"H1"``, ``"H2"`` the components of ``mu``
        perturb, in order; the others are left unperturbed.

    Returns
    -------
    callable
        ``fn(mu) -> float``, the gate fidelity at ``mu``.
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
    """Restrict a multi-parameter fidelity map to the ray ``mu0 + s d``.

    Parameters
    ----------
    fidelity_fn :
        Map ``mu -> F``.
    mu0 : array_like, shape (p,)
        Ray origin.
    d : array_like, shape (p,)
        Ray direction.

    Returns
    -------
    callable
        ``ray(s) -> float``, equal to ``fidelity_fn(mu0 + s d)``.
    """
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
    """Certified margin along the ray ``mu0 + s d``, via :func:`qrobustness.iterative_margin`.

    Parameters
    ----------
    fidelity_fn :
        Map ``mu -> F``.
    L : array_like, shape (p,)
        Per-parameter Lipschitz constants L_j.
    FT :
        Fidelity threshold F_T.
    d : array_like, shape (p,)
        Ray direction.
    mu0 : array_like, shape (p,), optional
        Ray origin; default the zero vector.
    L_dir :
        Directional Lipschitz constant; default the separable
        ``sum_j L_j |d_j|``. Pass ``JointGauge.L_dir(d, FT, N)`` =
        B_T C_joint(d) for the sharper constant.
    angular_gauge :
        If given, steps use the angular safe radius
        ``(arccos F_T - arccos F)/C^stat_FS(d)`` instead of the Lipschitz
        rule, unless ``safe_radius_fn`` is passed explicitly.
    **kwargs :
        Passed unchanged to :func:`qrobustness.iterative_margin`
        (``omega`` bounds the ray parameter ``s``).

    Returns
    -------
    MarginResult
        Margins in units of ``s``; the certified excursion is ``M * d``.
        If the gauge vanishes along ``d`` (C^stat_FS(d) = 0 when
        ``angular_gauge`` is given, else ``L_dir == 0``) the whole admissible
        ray is certified: ``M`` is the distance to the nearer end of
        ``omega`` (``inf`` if unbounded) and ``method``, ``status_*`` and
        ``reason_*`` are ``'zero_gauge'``.
    """
    d = np.asarray(d, dtype=float)
    L = np.asarray(L, dtype=float)
    if mu0 is None:
        mu0 = np.zeros_like(d)
    if L_dir is None:
        L_dir = float(np.dot(L, np.abs(d)))
    zero_gauge = (
        angular_gauge.C(d) == 0.0 if angular_gauge is not None else L_dir == 0.0
    )
    if zero_gauge:
        # The perturbation does not act along d: the whole admissible ray is safe.
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
    """Union of certified safe polytopes sharing the constants ``L``.

    Attributes
    ----------
    L : ndarray, shape (p,)
        Per-parameter Lipschitz constants L_j.
    polytopes : list of SafePolytope
        Member polytopes.
    provenance : list of str
        One note per polytope.
    """

    L: Array
    polytopes: List[SafePolytope] = field(default_factory=list)
    provenance: List[str] = field(default_factory=list)

    def add(self, centre: Array, F: float, FT: float, note: str = "") -> None:
        """Add the polytope about safe ``centre`` with fidelity ``F``, recording ``note``."""
        self.polytopes.append(safe_polytope(centre, self.L, F, FT))
        self.provenance.append(note)

    def contains(self, mu: Array) -> bool:
        """True if ``mu`` lies in any member polytope."""
        return any(P.contains(mu) for P in self.polytopes)


def axis_directions(p: int) -> Array:
    """The ``2p`` signed coordinate directions, shape (2p, p)."""
    eye = np.eye(p)
    return np.vstack([eye, -eye])


def diagonal_directions(p: int) -> Array:
    """All ``2^p`` sign diagonals, Euclidean unit length, shape (2^p, p)."""
    from itertools import product

    signs = np.array(list(product((-1.0, 1.0), repeat=p)))
    return signs / np.sqrt(p)


def sphere_directions(p: int, n: int, seed: Optional[int] = None) -> Array:
    """``n`` random Euclidean unit directions (normalised Gaussians), shape (n, p)."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, p))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


@dataclass
class JointGauge(PathGauge):
    """Combined-structure gauge C_joint and its certified region.

    ``C_joint(x) = Delta sum_k sqrt(x^T P^(k) x)`` with traceless interval Grams
    ``P^(k)_ij = Re Tr(Hbar_i^(k)' Hbar_j^(k))``. The region
    ``B_T C_joint(x) <= F_nu - F_T`` about a safe point nu is certified and
    contains the separable cross-polytope. Gauge arithmetic is inherited from
    :class:`qrobustness.lengthspace.PathGauge`.
    """

    def L_dir(self, d, FT: float, N: int) -> float:
        """Directional Lipschitz constant B_T C_joint(d) (at most ``sum_j L_j |d_j|``).

        Parameters
        ----------
        d : array_like, shape (p,)
            Direction.
        FT :
            Fidelity threshold F_T.
        N :
            Hilbert-space dimension.

        Returns
        -------
        float
        """
        from .core import lipschitz_constant

        return lipschitz_constant(FT, N, self.C(d))

    def contains(self, x, surplus: float, FT: float, N: int) -> bool:
        """True if displacement ``x`` satisfies ``B_T C_joint(x) <= surplus``."""
        from .core import lipschitz_constant

        return lipschitz_constant(FT, N, self.C(x)) <= surplus

    def boundary_radius(self, d, surplus: float, FT: float, N: int) -> float:
        """Certified radius ``surplus / (B_T C_joint(d))`` along ``d`` (``inf`` if zero)."""
        L = self.L_dir(d, FT, N)
        return surplus / L if L > 0 else float("inf")

    def alpha2_certified(self) -> float:
        """Upper bound on ``max_{|d|_2 = 1} C_joint(d)`` (see ``PathGauge.alpha_cs``)."""
        return self.alpha_cs()

    def inradius_certified(self, surplus: float, FT: float, N: int) -> float:
        """Certified Euclidean inradius of the gauge region for the given surplus F_nu - F_T."""
        from .core import lipschitz_constant

        L = lipschitz_constant(FT, N, self.alpha2_certified())
        return surplus / L if L > 0 else float("inf")


def joint_gauge(Hhat_lists: Sequence[HList], dt: float) -> JointGauge:
    """Build the combined-structure gauge C_joint.

    Structures are projected to their traceless parts (the trace part is a
    global phase and does not change the fidelity).

    Parameters
    ----------
    Hhat_lists : sequence of p lists of tau ndarrays, shape (N, N)
        Per-structure, per-interval perturbation structures.
    dt :
        Interval length Delta.

    Returns
    -------
    JointGauge
    """
    return JointGauge(grams=interval_grams(Hhat_lists, make_traceless=True), dt=dt)


@dataclass
class AngularGauge(PathGauge):
    """Static angular gauge C^stat_FS and its certified region.

    ``C^stat_FS(x) = Delta sum_k sqrt(x^T Q^(k) x)`` with ``Q^(k) = P^(k)/N``.
    The region ``C^stat_FS(x) <= arccos F_T - arccos F_nu`` about a safe point nu
    is certified and contains the :class:`JointGauge` region. Gauge arithmetic is
    inherited from :class:`qrobustness.lengthspace.PathGauge`.
    """

    @staticmethod
    def budget(F_nu: float, FT: float) -> float:
        """Angle budget ``arccos F_T - arccos F_nu``."""
        return angle_budget(F_nu, FT)

    def contains(self, x, F_nu: float, FT: float) -> bool:
        """True if displacement ``x`` lies in the region about a point of fidelity ``F_nu``."""
        return self.C(x) <= self.budget(F_nu, FT)

    def boundary_radius(self, d, F_nu: float, FT: float) -> float:
        """Certified radius along ``d``: budget / C^stat_FS(d) (``inf`` if zero)."""
        return self.radius(d, self.budget(F_nu, FT))

    def inradius_certified(self, F_nu: float, FT: float) -> float:
        """Certified Euclidean inradius of the region (see ``PathGauge.inradius``)."""
        return self.inradius(self.budget(F_nu, FT))


def angular_gauge(Hhat_lists: Sequence[HList], dt: float) -> AngularGauge:
    """Build the static angular gauge C^stat_FS.

    Parameters
    ----------
    Hhat_lists : sequence of p lists of tau ndarrays, shape (N, N)
        Per-structure, per-interval perturbation structures.
    dt :
        Interval length Delta.

    Returns
    -------
    AngularGauge
    """
    return AngularGauge(
        grams=interval_grams(Hhat_lists, make_traceless=True, normalise=True), dt=dt
    )

# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fidelity-based robustness margins for finite-time quantum control."""

from .core import (
    DU_METHODS,
    MARGIN_METHODS,
    ROOT_SOLVERS,
    MarginResult,
    dH_structure,
    differential_sensitivity,
    fidelity_vs_delta,
    gate_fidelity,
    iterative_margin,
    lipschitz_constant,
    load_controllers,
    load_problem,
    make_fidelity_fn,
    perturbed_hamiltonians,
    propagator,
    structure_constant,
)
from .kosut import (
    T_OMEGA_MAX,
    UncertaintyRates,
    fidelity_bound,
    fidelity_bound_at,
    threshold_time_bandwidth,
    time_bandwidth,
    uncertainty_rates,
)
from .kosut import margin as kosut_margin
from .optimize import (
    OptimizeResult,
    fidelity_and_gradient,
    optimize_controller,
    pack_controls,
    unpack_controls,
)
from .plotting import (
    apply_plot_style,
    save_fig,
    log10_axis,
    plot_fidelity_error_sweeps,
    plot_margins_vs_index,
    plot_margins_vs_sensitivity,
)

__all__ = [
    "propagator",
    "gate_fidelity",
    "lipschitz_constant",
    "structure_constant",
    "perturbed_hamiltonians",
    "dH_structure",
    "differential_sensitivity",
    "iterative_margin",
    "DU_METHODS",
    "MARGIN_METHODS",
    "ROOT_SOLVERS",
    "MarginResult",
    "fidelity_vs_delta",
    "load_problem",
    "load_controllers",
    "make_fidelity_fn",
    "T_OMEGA_MAX",
    "UncertaintyRates",
    "uncertainty_rates",
    "time_bandwidth",
    "fidelity_bound",
    "fidelity_bound_at",
    "threshold_time_bandwidth",
    "kosut_margin",
    "fidelity_and_gradient",
    "optimize_controller",
    "OptimizeResult",
    "pack_controls",
    "unpack_controls",
    "apply_plot_style",
    "save_fig",
    "log10_axis",
    "plot_margins_vs_index",
    "plot_margins_vs_sensitivity",
    "plot_fidelity_error_sweeps",
]

__version__ = "1.0.0"

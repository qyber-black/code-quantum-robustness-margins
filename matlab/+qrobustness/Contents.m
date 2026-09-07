% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
% QROBUSTNESS  Certified robustness margins for finite-time quantum gates.
%
%   Given a controller that reaches a target gate, this toolbox answers: how
%   large a structured perturbation can the gate absorb and still meet a
%   fidelity threshold FT? Every number it returns is a point at which the
%   fidelity was evaluated and found to meet the threshold, so it is a lower
%   bound on the true margin -- never an estimate of it.
%
%   Read a result's fields, not the number alone. ITERATIVE_MARGIN reports
%   which stopping rule fired and whether the whole segment or only the
%   endpoint is certified, and a margin found under a domain truncation is
%   not the same claim as a resolved one.
%
%   FOUR LAYERS, in dependency order.
%
%   1. Propagation and fidelity. PROPAGATOR composes the piecewise-constant
%      intervals; GATE_FIDELITY is the trace-amplitude |Tr(Uf' U)|/N used
%      throughout, the overlap of normalised Choi states, which is what
%      makes acos(F) a metric and every angular certificate possible.
%
%   2. Structure constants. TRACELESS centres a perturbation structure --
%      its trace part is a global phase the fidelity cannot see -- and
%      STRUCTURE_CONSTANT and LIPSCHITZ_CONSTANT turn it into the
%      sensitivity bound L = B_T C. Every gauge is built on centred
%      structures, so what counts as a valid structure is decided by
%      TRACELESS and nowhere else.
%
%   3. Exact interval derivatives. DU_DMU_EXACT evaluates the segment
%      derivative in the Hermitian eigenbasis, closed form and free of
%      quadrature error; DU_DMU_INTEGRAL is the Gauss-Legendre alternative,
%      kept as a cross-check. The open-system layer must use neither:
%      Lindblad generators can be defective, so +LINDBLAD uses the block
%      Frechet method instead.
%
%   4. Algorithm 1. ITERATIVE_MARGIN chains a certified safe radius outward
%      from the nominal point.
%
%   SUBPACKAGES. "help qrobustness.<name>" for each.
%     +multiparam   joint certificates over several parameters at once
%     +timevarying  certificates against within-gate fluctuations
%     +lengthspace  the path gauge those two share
%     +lindblad     open-system certificates in Liouville space
%     +kosut        the Kosut-Lidar-Rabitz bound, specialised
%     +berberich    the Berberich et al. bound, specialised
%     +compat       MATLAB/Octave shims and the shared CSV schemas
%
%   Propagation and fidelity
%     propagator                - ordered product of the interval unitaries
%     gate_fidelity             - |Tr(Uf' U)|/N
%     fidelity_vs_delta         - dense sweep for plotting, not a certificate
%     make_fidelity_fn          - F = fn(delta) for one structure
%     perturbed_hamiltonians    - per-interval H with one structure perturbed
%     dH_structure              - the per-interval structure dH/dmu
%
%   Structure constants
%     traceless                 - remove the trace part
%     structure_constant        - C for a drift or control structure
%     lipschitz_constant        - L = B_T C, B_T = sqrt((1 - FT^2)/N)
%
%   Derivatives and sensitivity
%     segment_eig               - Hermitian eigendecomposition of a segment
%     segment_propagator        - expm(-1i dt H) from that decomposition
%     dU_dmu_exact              - closed-form segment derivative
%     dU_dmu_integral           - Gauss-Legendre alternative
%     gauss_legendre_01         - nodes and weights on [0, 1]
%     parse_dU_options          - which derivative path and how many nodes
%     differential_sensitivity  - fidelity sensitivity zeta
%     fidelity_and_gradient     - fidelity and the GRAPE control gradients
%
%   Margins and synthesis
%     iterative_margin          - Algorithm 1, the certified margin
%     optimize_controller       - fidelity maximisation via fminunc + GRAPE
%
%   Data
%     load_problem              - read a problem definition
%     load_controllers          - read an ensemble, filtered by nominal error
%
%   Figures
%     plot_margins_vs_index     - margins against controller index
%     plot_margins_vs_sensitivity - two-panel M against |zeta|
%     plot_fidelity_error_sweeps  - fidelity error against perturbation size
%     apply_plot_style          - light theme for manuscript figures
%     log10_axis                - linear axis holding log10 data
%     convert_log_axis_to_log10_data - rewrite log axes as log10 data
%
%   Python is the reference implementation; this package is its peer, held
%   to it by the cross-engine comparisons of the committed result tables.
%   See README.md for what the peer does not cover.

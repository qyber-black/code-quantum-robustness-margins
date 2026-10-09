% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
% QROBUSTNESS  Robustness margins based on fidelity for finite-time quantum control.
%
%   Gate fidelity, structure constants C_{\hat H}, Lipschitz constants
%   L = B_T C, exact interval derivatives, the iterative margin M and
%   controller synthesis. Peer of python/src/qrobustness.
%
%   Subpackages ("help qrobustness.<name>"):
%     +multiparam   joint margins over several parameters
%     +timevarying  margins against time-varying perturbations
%     +lengthspace  path gauge shared by +multiparam and +timevarying
%     +lindblad     open-system margins in Liouville space
%     +states       state-fidelity margins
%     +openstates   open-system state-fidelity margins
%     +kosut        Kosut-Lidar-Rabitz fidelity bound
%     +berberich    Berberich et al. fidelity bound
%     +compat       MATLAB/Octave shims and CSV schemas
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
%     structure_constant        - C_{\hat H} for a drift or control structure
%     lipschitz_constant        - L = B_T C_{\hat H}, B_T = sqrt((1 - F_T^2)/N)
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
%     iterative_margin          - margin M of one parameter
%     optimize_controller       - fidelity maximisation via fminunc + GRAPE
%
%   Data
%     load_problem              - read a problem definition
%     load_controllers          - read controllers, filtered by nominal error
%
%   Figures
%     plot_margins_vs_index     - margins against controller index
%     plot_margins_vs_sensitivity - two-panel M against |zeta|
%     plot_fidelity_error_sweeps  - fidelity error against perturbation size
%     apply_plot_style          - light theme for manuscript figures
%     log10_axis                - linear axis holding log10 data
%     convert_log_axis_to_log10_data - rewrite log axes as log10 data

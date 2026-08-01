% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
% +KOSUT  Kosut-Lidar-Rabitz fundamental bound (arXiv:2507.01215), specialised.
%
%   Reference implementation of Theorem 1 of
%     R. L. Kosut, D. A. Lidar, H. Rabitz, "A Fundamental Bound for Robust
%     Quantum Gate Control", arXiv:2507.01215v2 (2025),
%   specialised to the closed-system, purely coherent, scalar structured
%   perturbation model of this toolbox, so that the perturbation margin it
%   implies can be compared with qrobustness.iterative_margin.
%
%   Specialisation.  Their uncertain Hamiltonian (their Eqs. 3-8) is bipartite,
%   H_unc = H_S^coh (x) I_B + I_S (x) H_B + H_SB.  Here the bath is absent
%   (H_B = H_SB = 0) and the coherent error is the structured scalar
%   perturbation H_unc^(k) = delta * Hhat^(k).  With U_S(t) the nominal
%   propagator, Htil(t) = U_S(t)' * H_unc(t) * U_S(t) and
%   <A> = (1/T) int_0^T A dt, their Eq. 28 becomes
%
%     Omega_unc     = max_t ||H_unc(t)||         = |delta| * w_unc
%     Omega_avg     = ||<Htil>||                 = |delta| * w_avg
%     Omega_avg^dev = max_t ||Htil(t) - <Htil>|| = |delta| * w_dev
%
%   in the induced 2-norm, all linear in delta, so the per-unit rates w_* are
%   controller/structure properties (uncertainty_rates) and
%
%     T*Omega_bnd(delta) = sqrt(a*delta^2 + b*|delta|),
%       a = T^2 * w_unc * w_dev,   b = 4 * T * w_avg
%     F_lb = max(1 - 0.5*(exp((T*Omega_bnd/2)^2) - 1)^2, 0)
%
%   (their Eqs. 29-30), which is monotone in |delta| and therefore invertible
%   in closed form (margin).
%
%   Caveats (see README.md and docs/time-bandwidth-bound.md).
%     1. Their Theorem 1 assumes exact nominal fidelity F_nom = 1 and bounds
%        the fidelity to the ACHIEVED nominal gate, not the target; pass
%        'nominal_error' to absorb the nominal deficit into the threshold.
%        The absorption is angular by default, F_eff = cos(acos(F_T) -
%        acos(1 - eps_0)) (effective_threshold), which is the sufficient
%        condition; the additive form F_T + eps_0 is not.
%     2. The implied margin is the CONSTANT structured-parameter
%        specialisation: it certifies constant |delta| <= M^K and is not a
%        supremum-norm time-varying margin.
%     3. Their bound is worst-case over all uncertainty consistent with the
%        norm bounds, including bath coupling; a larger margin here quantifies
%        the value of structural knowledge, not a defect of either bound.
%     4. F_lb is non-trivial only for T*Omega_bnd <= t_omega_max().
%
%   Functions
%     uncertainty_rates       - w_unc, w_avg, w_dev, T (their Eq. 28)
%     time_bandwidth          - T*Omega_bnd (their Eq. 29)
%     fidelity_bound          - F_lb from T*Omega_bnd (their Eq. 30)
%     fidelity_bound_at       - F_lb at a given delta
%     effective_threshold     - achieved-gate threshold implied by F_T
%     threshold_time_bandwidth- closed-form inverse of F_lb
%     margin                  - implied perturbation margin
%     t_omega_max             - 2*sqrt(log(1+sqrt(2))), vacuity threshold

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
% +KOSUT  Kosut-Lidar-Rabitz time-bandwidth bound (arXiv:2507.01215, Theorem 1).
%   Specialised to the closed-system scalar structured model H_unc = delta \hat H:
%   the per-unit rates w_unc, w_avg, w_dev, the fidelity lower bound F_lb and
%   the implied margin M^K (constant delta) or M^K_tv (trajectories
%   |delta(t)| <= m), with angular absorption of the nominal error. M^K is not
%   a sup-norm time-varying margin; use uncertainty 'trajectory' for that.
%
%   Functions
%     uncertainty_rates        - per-unit rates w_unc, w_avg, w_dev and T
%     time_bandwidth           - T*Omega_bnd at a given delta
%     fidelity_bound           - F_lb from T*Omega_bnd
%     fidelity_bound_at        - F_lb at a given delta
%     effective_threshold      - achieved-gate threshold implied by F_T
%     threshold_time_bandwidth - T*Omega_bnd at which F_lb equals the threshold
%     select_rates             - (w_avg, w_dev) for an uncertainty class
%     margin                   - implied margin M^K or M^K_tv
%     t_omega_max              - largest T*Omega_bnd with F_lb > 0
%
%   Peer of python/src/qrobustness/kosut.py.

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
% +TIMEVARYING  Certified margins for time-varying structured uncertainty.
%   The Lipschitz radius r_0 and the Choi-Fubini-Study radius r_FS certify
%   every measurable trajectory; the iterated margin M certifies constant
%   perturbations only. The adversarial search for m_adv is Python-only.
%
%   Functions
%     uniform_margin   - Lipschitz radius r_0
%     fs_margin        - Fubini-Study radius r_FS
%     fs_margin_joint  - Choi-speed certificate for a joint box
%
%   Peer of python/src/qrobustness/timevarying.py.

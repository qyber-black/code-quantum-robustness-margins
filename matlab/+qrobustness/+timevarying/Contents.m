% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
% +TIMEVARYING  Certificates against within-gate fluctuations.
%
%   The margins of QROBUSTNESS.ITERATIVE_MARGIN certify a CONSTANT
%   perturbation. These certify every trajectory whose sup-norm stays inside
%   the budget -- the perturbation may vary from interval to interval, and
%   adversarially so. That is a strictly stronger claim, and the two must
%   not be read as the same number.
%
%   Two radii, both closed form and neither costing an off-nominal fidelity
%   evaluation: the uniform Lipschitz radius r_0 = (F0 - FT)/sum(L), and the
%   Fubini-Study radius r_FS from the Choi-state trajectory speed. Under the
%   dominance theorem r_FS >= r_0, so r_FS is the one to quote.
%
%   The adversarial search that produces UPPER witnesses on the true
%   trajectory margin has no peer here; it is Python-only.
%
%   Functions
%     uniform_margin   - certified uniform radius r_0
%     fs_margin        - Fubini-Study trajectory radius r_FS
%     fs_margin_joint  - exact Choi-speed certificate for a joint box
%
%   Peer of python/src/qrobustness/timevarying.py.

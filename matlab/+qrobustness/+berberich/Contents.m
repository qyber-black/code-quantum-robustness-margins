% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
% +BERBERICH  Margins implied by the Berberich et al. bound (arXiv:2509.08481, Theorem 2.1).
%   Each control interval is treated as one gate. 'independent' gives the
%   trajectory margin, 'systematic' the constant-perturbation margin; the
%   nominal error is absorbed as the angle theta_0.
%
%   Functions
%     margin  - margin implied by the Berberich et al. bound
%
%   Peer of python/src/qrobustness/berberich.py.

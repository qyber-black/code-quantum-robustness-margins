% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
% +BERBERICH  Berberich et al. algorithm-level bound (arXiv:2509.08481).
%
%   Reference implementation of Theorem 2.1 of
%     J. Berberich, D. Fink, C. Holm, "Robustness of quantum algorithms
%     against coherent control errors", arXiv:2509.08481,
%   specialised to the scalar structured perturbation model of this toolbox,
%   so that the margin it implies can be placed beside
%   QROBUSTNESS.ITERATIVE_MARGIN and QROBUSTNESS.KOSUT.MARGIN.
%
%   Their bound is stated per gate and composed over the sequence, and it
%   admits two uncertainty classes: 'independent', where each interval may
%   err on its own, and 'systematic', where one deviation is shared by the
%   whole gate and the Magnus argument applies.
%
%   Functions
%     margin  - implied margin of their Theorem 2.1 (their Eq. 14)
%
%   Peer of python/src/qrobustness/berberich.py.

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function L = joint_gauge_L_dir(g, d, FT, N)
%JOINT_GAUGE_L_DIR Directional Lipschitz constant B_T C_joint(d).
%   g  - from qrobustness.multiparam.joint_gauge
%   d  - direction
%   FT - fidelity threshold F_T
%   N  - Hilbert-space dimension
%
%   Never exceeds sum_j L_j |d_j|.
%
%   Peer of python/src/qrobustness/multiparam.py:JointGauge.L_dir.

    L = qrobustness.lipschitz_constant(FT, N, ...
        qrobustness.lengthspace.path_gauge_C(g, d));
end

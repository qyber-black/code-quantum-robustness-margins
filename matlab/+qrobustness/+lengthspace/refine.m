% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function [H_ref, Hhat_ref] = refine(H_list, Hhat_list, q)
%REFINE Split each control interval into q equal sub-intervals.
%   The caller scales its time step by 1/q.  Peer of
%   python/src/qrobustness/lengthspace.py:refine.

    if ~(isscalar(q) && q >= 1 && q == floor(q))
        error('qrobustness:lengthspace:refine', 'q must be a positive integer.');
    end
    if numel(H_list) ~= numel(Hhat_list)
        error('qrobustness:lengthspace:refine', 'Lists must have equal length.');
    end
    H_ref = cell(1, numel(H_list) * q);
    Hhat_ref = cell(1, numel(Hhat_list) * q);
    for k = 1:numel(H_list)
        indices = (k - 1) * q + (1:q);
        H_ref(indices) = H_list(k);
        Hhat_ref(indices) = Hhat_list(k);
end

function run_smoke_case_study()
%RUN_SMOKE_CASE_STUDY Fast paper-driver smoke (2 controllers, no heavy sweeps).
%   Writes under build/smoke/ so it does not clobber paper artefacts.
    this_dir = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(this_dir));
    smoke = fullfile(root, 'build', 'smoke');
    run_lipschitz_margin_case_study('root', root, 'max_controllers', 2, ...
        'do_sweep', false, ...
        'build_dir', smoke, 'publish_dir', fullfile(smoke, 'publish'));
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

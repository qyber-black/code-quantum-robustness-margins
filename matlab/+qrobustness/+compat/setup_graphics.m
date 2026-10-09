function toolkit = setup_graphics()
%SETUP_GRAPHICS Choose the graphics toolkit used for figure export.
%   toolkit = qrobustness.compat.setup_graphics()
%
%   Under Octave selects qt when usable, otherwise gnuplot with its
%   advisory suppressed, once per session; returns the toolkit name. Returns
%   '' under MATLAB. PNG output differs between toolkits.

    persistent chosen
    if ~isempty(chosen)
        toolkit = chosen;
        return
    end

    if ~qrobustness.compat.is_octave()
        chosen = '';
        toolkit = chosen;
        return
    end

    available = available_graphics_toolkits();
    if any(strcmp(available, 'qt'))
        try
            graphics_toolkit('qt');
            chosen = 'qt';
            toolkit = chosen;
            return
        catch
            % qt is listed but cannot be used, which is the headless case.
        end
    end

    % gnuplot remains the fallback; the advisory carries no action for file output.
    warning('off', 'Octave:gnuplot-graphics');
    try
        graphics_toolkit('gnuplot');
        chosen = 'gnuplot';
    catch
        chosen = graphics_toolkit();
    end
    toolkit = chosen;
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

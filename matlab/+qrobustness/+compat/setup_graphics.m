function toolkit = setup_graphics()
%SETUP_GRAPHICS Select the graphics toolkit used for figure export.
%   toolkit = SETUP_GRAPHICS() returns the name of the toolkit in use, or ''
%   under MATLAB, where the choice does not arise.
%
%   Octave recommends the qt toolkit and warns that gnuplot is unmaintained,
%   so qt is selected whenever it is available.  Octave disables GUI features
%   when no display is present, and qt is then unavailable however it is
%   requested; gnuplot is the only remaining option in that case, so its
%   advisory is suppressed rather than repeated for every figure.
%
%   The selection is made once per session.  It affects only rendering, not
%   any computed value, but note that PNG output differs between toolkits, so
%   figures produced with and without a display are not byte-identical.

    persistent chosen
    if ~isempty(chosen)
        toolkit = chosen;
        return;
    end

    if ~qrobustness.compat.is_octave()
        chosen = '';
        toolkit = chosen;
        return;
    end

    available = available_graphics_toolkits();
    if any(strcmp(available, 'qt'))
        try
            graphics_toolkit('qt');
            chosen = 'qt';
            toolkit = chosen;
            return;
        catch
            % qt is listed but unusable, which is the headless case.
        end
    end

    % gnuplot is the fallback; the advisory carries no action for file output.
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

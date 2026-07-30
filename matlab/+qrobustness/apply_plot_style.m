function apply_plot_style(fig)
%APPLY_PLOT_STYLE Force light theme suitable for manuscript figures.
    if nargin < 1 || isempty(fig)
        fig = gcf;
    end
    set(fig, 'Color', [1 1 1], 'InvertHardcopy', 'on');
    ax = findall(fig, 'Type', 'axes');
    for k = 1:numel(ax)
        set(ax(k), 'Color', [1 1 1], 'XColor', [0 0 0], 'YColor', [0 0 0], 'Box', 'on');
        try
            set(ax(k), 'GridColor', [0.15 0.15 0.15], 'GridAlpha', 0.15);
        catch
            % older Octave may lack GridColor/GridAlpha
        end
    end
    legs = findall(fig, 'Type', 'legend');
    for k = 1:numel(legs)
        set(legs(k), 'Color', [1 1 1], 'TextColor', [0 0 0], 'EdgeColor', [0.15 0.15 0.15]);
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

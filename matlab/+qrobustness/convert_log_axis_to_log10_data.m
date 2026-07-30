function convert_log_axis_to_log10_data(fig, which)
%CONVERT_LOG_AXIS_TO_LOG10_DATA Rewrite log-scale axes as log10 data + linear.
%
%   For each axes with WhichScale='log', replace child YData/XData by
%   log10(data) and configure decade tick labels on a linear axis.

    if nargin < 2
        which = 'y';
    end
    which = lower(which);
    ax_list = findall(fig, 'Type', 'axes');
    for a = 1:numel(ax_list)
        ax = ax_list(a);
        if strcmp(which, 'y') && ~strcmp(get(ax, 'YScale'), 'log')
            continue;
        end
        if strcmp(which, 'x') && ~strcmp(get(ax, 'XScale'), 'log')
            continue;
        end

        raw_lim = get(ax, 'YLim');
        if strcmp(which, 'x')
            raw_lim = get(ax, 'XLim');
        end

        ch = get(ax, 'Children');
        for c = 1:numel(ch)
            if ~isprop(ch(c), 'YData')
                continue;
            end
            if strcmp(which, 'y')
                yd = get(ch(c), 'YData');
                yd(~isfinite(yd) | yd <= 0) = realmin;
                set(ch(c), 'YData', log10(yd));
            else
                xd = get(ch(c), 'XData');
                xd(~isfinite(xd) | xd <= 0) = realmin;
                set(ch(c), 'XData', log10(xd));
            end
        end

        % Constant lines / yline objects may appear as lines with constant y
        qrobustness.log10_axis(ax, which, raw_lim);
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

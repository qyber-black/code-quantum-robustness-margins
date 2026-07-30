function log10_axis(ax, which, raw_lim, varargin)
%LOG10_AXIS Configure a linear axis whose data are already log10-transformed.
%
%   Plot log10(values) on a linear axis, then call:
%     qrobustness.log10_axis(gca, 'y', [1e-7, 1e-3])
%
%   which: 'x' or 'y'
%   raw_lim: [raw_min, raw_max] in original (pre-log) units; used for ticks.
%
%   Name-value:
%     'minor'  true (default) -- draw decade minor ticks 2..9

    p = inputParser;
    addParameter(p, 'minor', true);
    parse(p, varargin{:});

    if nargin < 1 || isempty(ax)
        ax = gca;
    end
    which = lower(which);
    lo = floor(log10(raw_lim(1)));
    hi = ceil(log10(raw_lim(2)));
    majors = lo:hi;
    labels = arrayfun(@(e) sprintf('10^{%d}', e), majors, 'UniformOutput', false);

    if strcmp(which, 'y')
        set(ax, 'YScale', 'linear', 'YTick', majors, 'YTickLabel', labels, ...
            'YLim', [log10(raw_lim(1)), log10(raw_lim(2))]);
        if p.Results.minor
            minors = [];
            for e = lo:(hi - 1)
                minors = [minors, e + log10(2:9)]; %#ok<AGROW>
            end
            set(ax, 'YMinorTick', 'on', 'YMinorGrid', 'on');
            if ~qrobustness.compat.is_octave()
                ax.YAxis.MinorTickValues = minors;
            end
        end
    else
        set(ax, 'XScale', 'linear', 'XTick', majors, 'XTickLabel', labels, ...
            'XLim', [log10(raw_lim(1)), log10(raw_lim(2))]);
        if p.Results.minor
            minors = [];
            for e = lo:(hi - 1)
                minors = [minors, e + log10(2:9)]; %#ok<AGROW>
            end
            set(ax, 'XMinorTick', 'on', 'XMinorGrid', 'on');
            if ~qrobustness.compat.is_octave()
                ax.XAxis.MinorTickValues = minors;
            end
        end
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

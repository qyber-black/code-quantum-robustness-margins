function fig = plot_fidelity_error_sweeps(X_cell, Y_cell, FT, varargin)
%PLOT_FIDELITY_ERROR_SWEEPS Plot fidelity error against perturbation for several controllers.
%   X_cell, Y_cell - cell arrays of perturbation values and fidelity errors
%   FT             - fidelity threshold F_T
%   fig            - figure handle
%
%   Name-value options:
%     'Visible'   - figure visibility (default 'off')
%     'xlabel'    - x-axis label (default 'Perturbation strength \mu')
%     'xlim'      - x limits (default [] for automatic)
%     'FontSize'  - font size (default 18)
%     'NumXTicks' - number of x ticks (default 5)
%     'ylim'      - y limits; default [1e-7, 1.2e-3], widened if data falls outside
%
%   The error is drawn as log10 data on a linear axis.

    p = inputParser;
    addParameter(p, 'Visible', 'off');
    addParameter(p, 'xlabel', 'Perturbation strength \mu');
    addParameter(p, 'xlim', []);
    addParameter(p, 'FontSize', 18);
    addParameter(p, 'NumXTicks', 5);
    addParameter(p, 'ylim', []);
    parse(p, varargin{:});

    fig = figure('Visible', p.Results.Visible, 'Color', [1 1 1]);
    ax = axes('Parent', fig);
    hold(ax, 'on');

    nC = numel(X_cell);
    xmax = 0;
    for n = 1:nC
        x = X_cell{n}(:);
        y = Y_cell{n}(:);
        y = max(y, realmin);  % avoid -Inf
        plot(ax, x, log10(y), '-');
        xmax = max(xmax, max(abs(x)));
    end

    thr = 1 - FT;
    plot(ax, [-xmax, xmax], [log10(thr), log10(thr)], '-.r', 'LineWidth', 2);

    if isempty(p.Results.xlim)
        set(ax, 'XLim', [-xmax, xmax]);
    else
        set(ax, 'XLim', p.Results.xlim);
    end
    xl = get(ax, 'XLim');
    set(ax, 'XTick', linspace(xl(1), xl(2), p.Results.NumXTicks));
    if isempty(p.Results.ylim)
        ys = thr;
        for n = 1:nC
            ys = [ys; Y_cell{n}(:)]; %#ok<AGROW>
        end
        yspan = span_limits(ys, [1e-7, 1.2e-3]);
    else
        yspan = p.Results.ylim;
    end
    qrobustness.log10_axis(ax, 'y', yspan);
    set(ax, 'XScale', 'linear');
    grid(ax, 'on');
    xlabel(ax, p.Results.xlabel);
    ylabel(ax, 'fidelity error');
    set(ax, 'FontName', 'Arial', 'FontSize', p.Results.FontSize);
    qrobustness.apply_plot_style(fig);
end

function lim = span_limits(values, default_lim)
    lim = default_lim;
    values = values(isfinite(values) & values > 0);
    if isempty(values)
        return
    end
    if min(values) < lim(1)
        lim(1) = min(values);
    end
    if max(values) > lim(2)
        lim(2) = max(values);
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

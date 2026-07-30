function fig = plot_fidelity_error_sweeps(X_cell, Y_cell, FT, varargin)
%PLOT_FIDELITY_ERROR_SWEEPS Spaghetti plot of fidelity error vs delta.
%
%   Y values are plotted as log10(error) on a linear axis (not YScale=log).

    p = inputParser;
    addParameter(p, 'Visible', 'off');
    addParameter(p, 'xlabel', 'Perturbation strength \mu');
    addParameter(p, 'xlim', []);
    addParameter(p, 'FontSize', 18);
    addParameter(p, 'NumXTicks', 5);
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
    qrobustness.log10_axis(ax, 'y', [1e-7, 1.2e-3]);
    set(ax, 'XScale', 'linear');
    grid(ax, 'on');
    xlabel(ax, p.Results.xlabel);
    ylabel(ax, 'fidelity error');
    set(ax, 'FontName', 'Arial', 'FontSize', p.Results.FontSize);
    qrobustness.apply_plot_style(fig);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

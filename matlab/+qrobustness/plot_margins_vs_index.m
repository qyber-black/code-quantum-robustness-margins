function fig = plot_margins_vs_index(err, M0, M1, M2, varargin)
%PLOT_MARGINS_VS_INDEX Aligned margin and nominal-error panels.
%
%   Controllers are ordered by increasing nominal fidelity error.
%   Vertical axis plots log10(values) on a linear scale (not YScale=log).

    p = inputParser;
    addParameter(p, 'Visible', 'off');
    parse(p, varargin{:});

    [err_s, ord] = sort(err(:));
    M0 = M0(ord); M1 = M1(ord); M2 = M2(ord);
    idx = (1:numel(err_s))';

    % Clamp for log10: near-perfect fidelity can yield eps<=0 from roundoff.
    floor_pos = realmin('double');
    err_s = max(real(err_s), floor_pos);
    M0 = max(real(M0(:)), floor_pos);
    M1 = max(real(M1(:)), floor_pos);
    M2 = max(real(M2(:)), floor_pos);

    fig = figure('Visible', p.Results.Visible, 'Color', [1 1 1], ...
        'Position', [100 100 499 360], 'PaperPositionMode', 'auto');
    ax = axes('Parent', fig, 'Position', [0.18 0.459 0.79 0.484]);
    hold(ax, 'on');

    plot(ax, idx, log10(M0), 's', 'Color', [0 0.447 0.698], ...
        'MarkerFaceColor', [0 0.447 0.698], ...
        'MarkerSize', 3.5, 'LineStyle', 'none', 'DisplayName', 'H_0');
    plot(ax, idx, log10(M1), '>', 'Color', [0 0.620 0.451], ...
        'MarkerFaceColor', 'none', ...
        'MarkerSize', 3.5, 'LineStyle', 'none', 'DisplayName', 'H_1');
    plot(ax, idx, log10(M2), '<', 'Color', [0.835 0.369 0], ...
        'MarkerFaceColor', [0.835 0.369 0], ...
        'MarkerSize', 3.5, 'LineStyle', 'none', 'DisplayName', 'H_2');

    limits = log10([min([M0; M1; M2])/1.15, max([M0; M1; M2])*1.4]);
    ticks = [];
    for exponent = floor(limits(1)):ceil(limits(2))
        candidates = [1 2 5] * 10^exponent;
        ticks = [ticks, candidates(log10(candidates) >= limits(1) & ...
            log10(candidates) <= limits(2))]; %#ok<AGROW>
    end
    labels = arrayfun(@(x) sprintf('%.3g', x), ticks, 'UniformOutput', false);
    set(ax, 'YLim', limits, 'YTick', log10(ticks), 'YTickLabel', labels, ...
        'XLim', [1 numel(idx)], 'XTickLabel', [], 'XScale', 'linear');
    ylabel(ax, 'Robustness margin');
    leg = legend(ax, 'Location', 'northeast', 'Orientation', 'horizontal');
    set(leg, 'FontSize', 10);
    text(ax, 0.02, 0.91, '(a)', 'Units', 'normalized', 'Color', [0 0 0]);

    ax_err = axes('Parent', fig, 'Position', [0.18 0.149 0.79 0.263]);
    plot(ax_err, idx, log10(err_s), '-', 'Color', [0.066 0.443 0.745], ...
        'LineWidth', 1.2, 'DisplayName', 'nominal fidelity error');
    qrobustness.log10_axis(ax_err, 'y', [min(err_s)*0.8, max(err_s)*1.2]);
    set(ax_err, 'XLim', [1 numel(idx)], 'XScale', 'linear');
    xlabel(ax_err, 'Controller index (increasing nominal error)');
    ylabel(ax_err, 'Nominal error \epsilon_0');
    text(ax_err, 0.02, 0.82, '(b)', 'Units', 'normalized', 'Color', [0 0 0]);
    linkaxes([ax ax_err], 'x');
    set([ax ax_err], 'FontName', 'Arial', 'FontSize', 11);
    grid(ax, 'on');
    grid(ax_err, 'on');
    qrobustness.apply_plot_style(fig);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

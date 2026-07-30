function fig = plot_margins_vs_index(err, M0, M1, M2, varargin)
%PLOT_MARGINS_VS_INDEX Manuscript-style margins vs controller index.
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
        'Position', [100 100 528 482]);
    ax = axes('Parent', fig);
    hold(ax, 'on');

    % log10-transformed values on linear axes
    plot(ax, idx, log10(err_s), '-', 'Color', [0.066 0.443 0.745], ...
        'LineWidth', 1.2, 'DisplayName', 'nominal fidelity error');
    plot(ax, idx, log10(M0), 's', 'Color', [0 0 1], 'MarkerFaceColor', [0 0 1], ...
        'MarkerSize', 6, 'LineStyle', 'none', 'DisplayName', 'H_0 robustness margins');
    plot(ax, idx, log10(M1), '>', 'Color', [0 1 0], 'MarkerFaceColor', [0 1 0], ...
        'MarkerSize', 6, 'LineStyle', 'none', 'DisplayName', 'H_1 robustness margins');
    plot(ax, idx, log10(M2), '<', 'Color', [1 0 0], 'MarkerFaceColor', [1 0 0], ...
        'MarkerSize', 6, 'LineStyle', 'none', 'DisplayName', 'H_2 robustness margins');

    qrobustness.log10_axis(ax, 'y', [1e-7, 1e-1]);
    set(ax, 'XLim', [1, numel(idx)], 'XScale', 'linear');
    grid(ax, 'on');
    set(ax, 'XMinorGrid', 'off');
    xlabel(ax, 'controller index');
    ylabel(ax, '');
    legend(ax, 'Location', 'southeast');
    set(ax, 'FontName', 'Arial', 'FontSize', 14);
    qrobustness.apply_plot_style(fig);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

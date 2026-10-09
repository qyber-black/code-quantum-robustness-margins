function fig = plot_margins_vs_index(err, M0, M1, M2, varargin)
%PLOT_MARGINS_VS_INDEX Plot margins against controller index.
%   err        - nominal fidelity errors; controllers are sorted by them
%   M0, M1, M2 - margins for the H0, H1 and H2 structures
%   fig        - figure handle
%
%   Name-value options:
%     'Visible' - figure visibility (default 'off')
%     'ylim'    - y limits; default [1e-7, 1e-1], widened if data falls outside
%
%   Values are drawn as log10 data on a linear axis.

    p = inputParser;
    addParameter(p, 'Visible', 'off');
    addParameter(p, 'ylim', []);
    parse(p, varargin{:});

    [err_s, ord] = sort(err(:));
    M0 = M0(ord); M1 = M1(ord); M2 = M2(ord);
    idx = (1:numel(err_s))';

    % Clamp for log10: near-perfect fidelity can produce eps<=0 from roundoff.
    floor_pos = realmin('double');
    err_s = max(real(err_s), floor_pos);
    M0 = max(real(M0(:)), floor_pos);
    M1 = max(real(M1(:)), floor_pos);
    M2 = max(real(M2(:)), floor_pos);

    fig = figure('Visible', p.Results.Visible, 'Color', [1 1 1], ...
        'Position', [100 100 528 482]);
    ax = axes('Parent', fig);
    hold(ax, 'on');

    % Marker colours; keep in step with COLOR_ERR / COLOR_H0 / COLOR_H1 / COLOR_H2 in plotting.py.
    color_err = [0.066 0.443 0.745];
    color_H0 = [0 0 1];
    color_H1 = [0 1 0];
    color_H2 = [1 0 0];

    % log10-transformed values on linear axes
    plot(ax, idx, log10(err_s), '-', 'Color', color_err, ...
        'LineWidth', 1.2, 'DisplayName', 'nominal fidelity error');
    plot(ax, idx, log10(M0), 's', 'Color', color_H0, 'MarkerFaceColor', color_H0, ...
        'MarkerSize', 6, 'LineStyle', 'none', 'DisplayName', 'H_0 robustness margins');
    plot(ax, idx, log10(M1), '>', 'Color', color_H1, 'MarkerFaceColor', color_H1, ...
        'MarkerSize', 6, 'LineStyle', 'none', 'DisplayName', 'H_1 robustness margins');
    plot(ax, idx, log10(M2), '<', 'Color', color_H2, 'MarkerFaceColor', color_H2, ...
        'MarkerSize', 6, 'LineStyle', 'none', 'DisplayName', 'H_2 robustness margins');

    if isempty(p.Results.ylim)
        yspan = span_limits([err_s(:); M0(:); M1(:); M2(:)], [1e-7, 1e-1]);
    else
        yspan = p.Results.ylim;
    end
    qrobustness.log10_axis(ax, 'y', yspan);
    set(ax, 'XLim', [1, numel(idx)], 'XScale', 'linear');
    grid(ax, 'on');
    set(ax, 'XMinorGrid', 'off');
    xlabel(ax, 'controller index');
    ylabel(ax, '');
    legend(ax, 'Location', 'southeast');
    set(ax, 'FontName', 'Arial', 'FontSize', 14);
    qrobustness.apply_plot_style(fig);
end

function lim = span_limits(values, default_lim)
    lim = default_lim;
    values = values(isfinite(values) & values > 0);
    if isempty(values)
        return
    end
    vmin = min(values);
    vmax = max(values);
    if vmin < lim(1)
        lim(1) = vmin;
    end
    if vmax > lim(2)
        lim(2) = vmax;
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

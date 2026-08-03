function fig = plot_margins_vs_sensitivity(abs_z0, abs_z1, abs_z2, M0, M1, M2, varargin)
%PLOT_MARGINS_VS_SENSITIVITY Two-panel M vs |zeta| (manuscript layout).
%
%   X coordinates are log10(|zeta|) on a linear axis (not XScale=log).
%   Y remains linear in the margin M.

    p = inputParser;
    addParameter(p, 'Visible', 'off');
    parse(p, varargin{:});

    z0 = max(abs_z0(:), realmin);
    z1 = max(abs_z1(:), realmin);
    z2 = max(abs_z2(:), realmin);

    fig = figure('Visible', p.Results.Visible, 'Color', [1 1 1], ...
        'Position', [100 100 520 520]);

    % Top: H0
    ax1 = subplot(2, 1, 1, 'Parent', fig);
    hold(ax1, 'on');
    plot(ax1, log10(z0), M0(:), 's', 'Color', [0 0 1], 'MarkerFaceColor', [0 0 1], ...
        'MarkerSize', 6, 'LineStyle', 'none', 'DisplayName', 'H_0 Perturbation');
    grid(ax1, 'on');
    xlabel(ax1, '|\zeta|');
    ylabel(ax1, 'Robustness margin');
    legend(ax1, 'Location', 'northwest');
    qrobustness.log10_axis(ax1, 'x', [min(z0)*0.8, max(z0)*1.2]);
    set(ax1, 'YScale', 'linear', 'FontName', 'Helvetica', 'FontSize', 12);

    % Bottom: H1 / H2
    ax2 = subplot(2, 1, 2, 'Parent', fig);
    hold(ax2, 'on');
    plot(ax2, log10(z1), M1(:), '>', 'Color', [0 1 0], 'MarkerFaceColor', [0 1 0], ...
        'MarkerSize', 6, 'LineStyle', 'none', 'DisplayName', 'H_1 Perturbation');
    plot(ax2, log10(z2), M2(:), '<', 'Color', [1 0 0], 'MarkerFaceColor', [1 0 0], ...
        'MarkerSize', 6, 'LineStyle', 'none', 'DisplayName', 'H_2 Perturbation');
    grid(ax2, 'on');
    xlabel(ax2, '|\zeta|');
    ylabel(ax2, 'Robustness margin');
    legend(ax2, 'Location', 'northwest');
    z12 = [z1; z2];
    qrobustness.log10_axis(ax2, 'x', [min(z12)*0.8, max(z12)*1.2]);
    set(ax2, 'YScale', 'linear', 'FontName', 'Helvetica', 'FontSize', 12);

    qrobustness.apply_plot_style(fig);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

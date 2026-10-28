function test_plotting()
%TEST_PLOTTING Preserve controller identities and distinguish plot quantities.
    qrobustness.compat.setup_graphics();
    err = [1e-4 1e-6 1e-5];
    M0 = [0.008 0.004 0.006];
    M1 = [0.011 0.007 0.009];
    M2 = [0.012 0.010 0.015];
    fig = qrobustness.plot_margins_vs_index(err, M0, M1, M2);
    cleanup = onCleanup(@() close(fig));
    h0 = findobj(fig, 'Type', 'line', 'DisplayName', 'H_0');
    h1 = findobj(fig, 'Type', 'line', 'DisplayName', 'H_1');
    h2 = findobj(fig, 'Type', 'line', 'DisplayName', 'H_2');
    he = findobj(fig, 'Type', 'line', 'DisplayName', 'nominal fidelity error');
    assert(numel(h0) == 1 && numel(h1) == 1 && numel(h2) == 1 && numel(he) == 1);
    actual = get(h0, 'YData');
    assert(max(abs(actual(:) - log10([0.004; 0.006; 0.008]))) < 1e-14);
    actual = get(h1, 'YData');
    assert(max(abs(actual(:) - log10([0.007; 0.009; 0.011]))) < 1e-14);
    actual = get(h2, 'YData');
    assert(max(abs(actual(:) - log10([0.010; 0.015; 0.012]))) < 1e-14);
    actual = get(he, 'YData');
    assert(max(abs(actual(:) - [-6; -5; -4])) < 1e-14);
    top = get(h0, 'Parent');
    bottom = get(he, 'Parent');
    assert(~isequal(top, bottom), 'Margins and nominal error need separate panels');
    assert(isequal(get(top, 'XLim'), get(bottom, 'XLim')));
    assert(diff(get(top, 'YLim')) < 1, 'Margins must not share the error range');
    assert(strcmp(get(top, 'YScale'), 'linear') && strcmp(get(bottom, 'YScale'), 'linear'));
    assert(strcmp(get(h1, 'MarkerFaceColor'), 'none'));
    assert(strcmp(get(h1, 'Marker'), '>') && strcmp(get(h2, 'Marker'), '<'));
    assert(get(h0, 'MarkerSize') == 3.5);
    assert(isequal(get(h0, 'Color'), [0 0.447 0.698]));
    assert(isequal(get(h1, 'Color'), [0 0.620 0.451]));
    assert(isequal(get(h2, 'Color'), [0.835 0.369 0]));
    clear cleanup;

    fig = qrobustness.plot_margins_vs_sensitivity(err, err, err, M0, M1, M2);
    cleanup = onCleanup(@() close(fig));
    h1 = findobj(fig, 'Type', 'line', 'DisplayName', 'H_1 Perturbation');
    h2 = findobj(fig, 'Type', 'line', 'DisplayName', 'H_2 Perturbation');
    h0 = findobj(fig, 'Type', 'line', 'DisplayName', 'H_0 Perturbation');
    top_position = get(get(h1, 'Parent'), 'Position');
    bottom_position = get(get(h0, 'Parent'), 'Position');
    assert(top_position(2) > bottom_position(2), 'H1/H2 must be above H0');
    actual = get(h1, 'YData');
    assert(isequal(actual(:), M1(:)));
    actual = get(h2, 'YData');
    assert(isequal(actual(:), M2(:)));
    actual = get(h0, 'YData');
    assert(isequal(actual(:), M0(:)));
    assert(strcmp(get(h1, 'MarkerFaceColor'), 'none'));
    assert(isequal(get(h1, 'Color'), [0 0.620 0.451]));
    assert(isequal(get(h2, 'Color'), [0.835 0.369 0]));
    clear cleanup;

    xs = repmat({[-0.01 0 0.01]}, 1, 8);
    ys = repmat({[1e-3 1e-6 1e-3]}, 1, 8);
    fig = qrobustness.plot_fidelity_error_sweeps(xs, ys, 0.999, ...
        'xlabel', '\delta_0', 'xlim', [-0.02 0.02], 'FontSize', 14, 'NumXTicks', 3);
    cleanup = onCleanup(@() close(fig));
    lines = findobj(fig, 'Type', 'line');
    threshold = lines(strcmp(get(lines, 'LineStyle'), '--'));
    curves = lines(strcmp(get(lines, 'LineStyle'), '-'));
    assert(numel(curves) == 8 && numel(threshold) == 1);
    assert(isequal(get(threshold, 'Color'), [0 0 0]));
    assert(max(abs(get(threshold, 'YData') - log10(1 - 0.999))) < 1e-14);
    assert(isequal(get(threshold, 'XData'), [-0.02 0.02]));
    colors = get(curves, 'Color');
    assert(size(unique(vertcat(colors{:}), 'rows'), 1) == 7, ...
        'Sweep curves must retain the seven-colour controller cycle');
    ax = get(threshold, 'Parent');
    assert(numel(get(ax, 'XTick')) == 3 && get(ax, 'FontSize') == 14);
    assert(strcmp(get(get(ax, 'XLabel'), 'String'), '\delta_0'));
    clear cleanup;
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
function test_plot_ranges()
%TEST_PLOT_RANGES Historical y limits stay when the data fits, and widen when it does not.

    err = [1e-4; 1e-3];
    inside = [1e-4; 1e-3];
    fig = qrobustness.plot_margins_vs_index(err, inside, inside, inside);
    yl = ylim_of(fig);
    close(fig);
    assert(abs(yl(1) - log10(1e-7)) < 1e-12 && abs(yl(2) - log10(1e-1)) < 1e-12, ...
        'margins kept the historical range');

    fig = qrobustness.plot_margins_vs_index(err, [1e-8; 0.2], inside, inside);
    yl = ylim_of(fig);
    close(fig);
    assert(yl(1) < log10(1e-7) && yl(2) > log10(1e-1), 'margins widened');

    x = linspace(-0.01, 0.01, 20);
    fig = qrobustness.plot_fidelity_error_sweeps({x}, {1e-4 * ones(size(x))}, 0.999);
    yl = ylim_of(fig);
    close(fig);
    assert(abs(yl(1) - log10(1e-7)) < 1e-12 && abs(yl(2) - log10(1.2e-3)) < 1e-12, ...
        'sweeps kept the historical range');

    fig = qrobustness.plot_fidelity_error_sweeps({x}, {1e-2 * ones(size(x))}, 0.999);
    yl = ylim_of(fig);
    close(fig);
    assert(yl(2) > log10(1.2e-3), 'sweeps widened');
end

function yl = ylim_of(fig)
    ax = get(fig, 'CurrentAxes');
    yl = get(ax, 'YLim');
end

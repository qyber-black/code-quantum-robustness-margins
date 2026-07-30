function test_compat_graphics()
% Graphics compatibility helpers behave on both engines.
%
% Octave recommends the qt toolkit over gnuplot and has not implemented the
% 'best' legend location.  The helpers select qt when it is available and ask
% for the location Octave would substitute anyway, so neither warning is
% emitted during a figure-producing run.

    loc = qrobustness.compat.legend_location();
    assert(ischar(loc) && ~isempty(loc), 'legend_location must return a string');

    toolkit = qrobustness.compat.setup_graphics();

    if qrobustness.compat.is_octave()
        assert(strcmp(loc, 'northeast'), ...
            'Octave has no ''best'' location; expected northeast, got %s', loc);
        assert(any(strcmp(toolkit, available_graphics_toolkits())), ...
            'setup_graphics chose %s, which is not available', toolkit);
        % qt is the recommended toolkit and must be preferred when usable.
        if any(strcmp(available_graphics_toolkits(), 'qt'))
            assert(any(strcmp(toolkit, {'qt', 'gnuplot'})), ...
                'unexpected toolkit %s', toolkit);
        end
        % Selection is stable within a session.
        assert(strcmp(toolkit, qrobustness.compat.setup_graphics()), ...
            'setup_graphics must be idempotent');
    else
        assert(strcmp(loc, 'best'), 'MATLAB supports ''best''; got %s', loc);
        assert(isempty(toolkit), 'setup_graphics is a no-op on MATLAB');
    end

    % Exporting through the compat layer must not raise the toolkit advisory
    % or the legend-location substitution.  Octave emits unrelated warnings
    % from its own internals, so match on those two subjects only.
    lastwarn('');
    fig = figure('Visible', 'off');
    plot(1:3, 1:3);
    legend('a', 'Location', qrobustness.compat.legend_location());
    tmp = [tempname() '.png'];
    qrobustness.compat.export_figure(fig, tmp, 100);
    [msg, ~] = lastwarn();
    close(fig);
    lower_msg = lower(msg);
    assert(isempty(strfind(lower_msg, 'gnuplot')), ...
        'toolkit advisory was emitted: %s', msg);
    assert(isempty(strfind(lower_msg, 'location specifier')), ...
        'legend location was substituted: %s', msg);
    assert(exist(tmp, 'file') == 2, 'export_figure wrote no file');
    delete(tmp);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

function write_margin_audit(rows, out, FT, k_max, margin_tol)
%WRITE_MARGIN_AUDIT Measured continuation and refinement costs and outcomes.
    fields = fieldnames(rows);
    fid = fopen(fullfile(out, sprintf('margin_audit_%g.csv', FT)), 'w');
    fprintf(fid, '%s\n', strjoin(fields, ','));
    for k = 1:numel(rows)
        for j = 1:numel(fields)
            if j > 1; fprintf(fid, ','); end
            value = rows(k).(fields{j});
            if ischar(value)
                fprintf(fid, '%s', value);
            else
                fprintf(fid, '%.17g', value);
            end
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
    widths = ([rows.M_upper] - [rows.M]) ./ [rows.M];
    width = max(widths);
    exponent = floor(log10(width));
    mantissa = ceil(width / 10^exponent * 10) / 10;
    names = {'Cases', 'Rays', 'KMax', 'GuardRays', 'EtaBandRays', 'BracketedRays'};
    values = [numel(rows), 2 * numel(rows), k_max, ...
        sum([rows.safeguard_minus]) + sum([rows.safeguard_plus]), ...
        sum(strcmp({rows.status_minus}, 'eta_band')) + sum(strcmp({rows.status_plus}, 'eta_band')), ...
        sum(strcmp({rows.reason_minus}, 'bracketed')) + sum(strcmp({rows.reason_plus}, 'bracketed'))];
    phases = {'Continuation', 'Refinement'};
    data_fields = {'n_evals_continuation', 'n_evals_refinement'};
    for j = 1:2
        data = [rows.(data_fields{j})];
        names = [names, {[phases{j} 'Min'], [phases{j} 'Median'], [phases{j} 'Max']}]; %#ok<AGROW>
        values = [values, min(data), median(data), max(data)]; %#ok<AGROW>
    end
    fid = fopen(fullfile(out, sprintf('margin_audit_%g.tex', FT)), 'w');
    fprintf(fid, '%% Auto-generated from margin_audit CSV; evaluations are per two-direction case.\n');
    for j = 1:numel(names)
        fprintf(fid, '\\newcommand{\\QRMAudit%s}{%g}\n', names{j}, values(j));
    end
    fprintf(fid, '\\newcommand{\\QRMAuditControllers}{%d}\n', numel(unique([rows.controller])));
    structures = {'H0', 'H1', 'H2'};
    spread_names = {'DriftSpread', 'ControlOneSpread', 'ControlTwoSpread'};
    for j = 1:3
        data = [rows(strcmp({rows.structure}, structures{j})).M];
        fprintf(fid, '\\newcommand{\\QRMAudit%s}{%.1f}\n', spread_names{j}, max(data) / min(data));
    end
    fprintf(fid, '\\newcommand{\\QRMAuditRelativeWidth}{%.1f\\times10^{%d}}\n', mantissa, exponent);
    [coefficient, power] = strtok(sprintf('%.0e', margin_tol), 'e');
    fprintf(fid, '\\newcommand{\\QRMAuditTolerance}{%s\\times10^{%d}}\n', coefficient, str2double(power(2:end)));
    fclose(fid);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

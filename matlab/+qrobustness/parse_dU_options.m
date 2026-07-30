function opts = parse_dU_options(varargin)
%PARSE_DU_OPTIONS Options controlling the segment-derivative evaluation.
%   opts = PARSE_DU_OPTIONS(...) returns a struct with fields
%     .method   'exact' (default) or 'quadrature'
%     .n_quad   Gauss-Legendre nodes, used only when method is 'quadrature'
%               (default 32)
%
%   A leading bare numeric argument is read as a positional n_quad, so calls
%   of the form f(..., 32) are accepted.  n_quad applies only to the
%   'quadrature' method; under the default 'exact' it is unused.
%
%   See also QROBUSTNESS.DU_DMU_EXACT, QROBUSTNESS.DU_DMU_QUAD.

    args = varargin;
    n_quad_positional = [];
    if ~isempty(args) && (isnumeric(args{1}) || isempty(args{1}))
        n_quad_positional = args{1};
        args = args(2:end);
    end

    p = inputParser;
    addParameter(p, 'method', 'exact');
    addParameter(p, 'n_quad', 32);
    parse(p, args{:});
    opts = p.Results;

    if ~isempty(n_quad_positional)
        opts.n_quad = n_quad_positional;
    end

    if ~(ischar(opts.method) || isstring(opts.method))
        error('qrobustness:dU:Method', 'method must be a string');
    end
    opts.method = lower(char(opts.method));
    if ~any(strcmp(opts.method, {'exact', 'quadrature'}))
        error('qrobustness:dU:Method', ...
            'Unknown method=''%s''; expected ''exact'' or ''quadrature''', opts.method);
    end
    if isempty(opts.n_quad)
        opts.n_quad = 32;
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

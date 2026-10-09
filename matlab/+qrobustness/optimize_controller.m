function res = optimize_controller(H0, H1, H2, Uf, tf, tau, varargin)
%OPTIMIZE_CONTROLLER Maximise gate fidelity with fminunc (quasi-Newton) and GRAPE gradients.
%   H0, H1, H2 - drift and control Hamiltonians
%   Uf         - target gate
%   tf, tau    - gate time and number of intervals
%   res        - struct with fields u1, u2, fid, error, fid_init, n_iter,
%                success, message
%
%   Name-value options:
%     'u1_init', 'u2_init' - initial controls (default sigma*randn)
%     'sigma'              - initial standard deviation (default 1)
%     'seed'               - RNG seed (default [])
%     'method'             - 'exact' (default) or 'quadrature'
%     'n_quad'             - Gauss-Legendre nodes for 'quadrature' (default 32)
%     'maxiter'            - maximum iterations (default 500)
%     'ftol'               - step and optimality tolerance (default 1e-12)
%
%   Peer of python/src/qrobustness/optimize.py:optimize_controller.

    p = inputParser;
    addParameter(p, 'u1_init', []);
    addParameter(p, 'u2_init', []);
    addParameter(p, 'sigma', 1.0);
    addParameter(p, 'seed', []);
    addParameter(p, 'method', 'exact');
    addParameter(p, 'n_quad', 32);
    addParameter(p, 'maxiter', 500);
    addParameter(p, 'ftol', 1e-12);
    parse(p, varargin{:});
    opt = p.Results;

    dt = tf / tau;
    if ~isempty(opt.seed)
        rng(opt.seed);
    end
    if isempty(opt.u1_init)
        u1_init = opt.sigma * randn(1, tau);
    else
        u1_init = opt.u1_init(:).';
    end
    if isempty(opt.u2_init)
        u2_init = opt.sigma * randn(1, tau);
    else
        u2_init = opt.u2_init(:).';
    end
    if numel(u1_init) ~= tau || numel(u2_init) ~= tau
        error('qrobustness:optimise:Size', 'u1_init/u2_init length must equal tau');
    end

    H_list0 = qrobustness.perturbed_hamiltonians(H0, H1, H2, u1_init, u2_init, 'H0', 0);
    fid_init = qrobustness.gate_fidelity(qrobustness.propagator(H_list0, dt), Uf);

    x0 = pack_controls(u1_init, u2_init);
    dU_opts = qrobustness.parse_dU_options('method', opt.method, 'n_quad', opt.n_quad);

    obj = @(x) error_and_grad(x, H0, H1, H2, Uf, dt, tau, dU_opts);

    % Octave has fminunc but not optimoptions, so use optimset there.
    if qrobustness.compat.is_octave()
        opts = optimset( ...
            'GradObj', 'on', ...
            'Display', 'off', ...
            'MaxIter', opt.maxiter, ...
            'TolFun', opt.ftol, ...
            'TolX', opt.ftol);
    else
        opts = optimoptions('fminunc', ...
            'Algorithm', 'quasi-newton', ...
            'SpecifyObjectiveGradient', true, ...
            'Display', 'off', ...
            'MaxIterations', opt.maxiter, ...
            'OptimalityTolerance', opt.ftol, ...
            'StepTolerance', opt.ftol);
    end

    [x, fval, exitflag, output] = fminunc(obj, x0, opts);
    [u1, u2] = unpack_controls(x, tau);
    % Clamp: roundoff can drive F slightly above 1 (negative error).
    fid = min(1, max(0, 1 - fval));

    res = struct();
    res.u1 = u1;
    res.u2 = u2;
    res.fid = fid;
    res.error = max(0, 1 - fid);
    res.fid_init = fid_init;
    % Octave's output struct includes iterations but lacks a message field.
    if isfield(output, 'iterations')
        res.n_iter = output.iterations;
    else
        res.n_iter = NaN;
    end
    res.success = exitflag > 0;
    if isfield(output, 'message')
        res.message = output.message;
    else
        res.message = sprintf('fminunc exitflag %d', exitflag);
    end
end

function [f, g] = error_and_grad(x, H0, H1, H2, Uf, dt, tau, dU_opts)
% Objective and gradient supplied to fminunc: gate error and its derivative
%   with respect to the packed control vector.
    [u1, u2] = unpack_controls(x, tau);
    [F, g1, g2] = qrobustness.fidelity_and_gradient(H0, H1, H2, u1, u2, Uf, dt, ...
        'method', dU_opts.method, 'n_quad', dU_opts.n_quad);
    f = 1 - F;
    g = -pack_controls(g1, g2);
end

function x = pack_controls(u1, u2)
% Pack the two control rows into one column, column-major, the
%   layout the controller CSV files use.
    u = [u1(:).'; u2(:).'];  % 2 x tau; u(:) is column-major interleave
    x = u(:);
end

function [u1, u2] = unpack_controls(x, tau)
%   Inverse of PACK_CONTROLS.
    u = reshape(x, 2, tau);
    u1 = u(1, :);
    u2 = u(2, :);
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

function result = iterative_margin(fidelity_fn, L, FT, varargin)
%ITERATIVE_MARGIN Certified / exploratory 1-D robustness margin.
%
%   result = qrobustness.iterative_margin(fidelity_fn, L, FT, ...)
%
%   Preferred certified default is method='algorithm1' (Lipschitz + bisection).
%   Lipschitz steps rarely overshoot, so Brent/TOMS748 (fzero) polish does not
%   meaningfully cut fidelity evals versus bisection; keep them as optional
%   polish only. Aggressive advance (doubling / newton_probe) can cut evals but
%   drops the full segment certificate unless F is monotone on the ray.
%   See docs/margin-solvers-notes.md.
%
%   Name-value options:
%     'mu0'               (default 0)
%     'eta'               (default 1e-6)
%     'omega'             (default [-Inf, Inf])  domain [mu_min, mu_max]
%     'k_max'             (default 10000)
%     'method'            (default 'algorithm1')
%         'algorithm1'         -- paper Alg. 1: Lipschitz + bisection (default)
%         'lipschitz_brent'    -- Lipschitz + fzero/Brent polish (full certificate;
%                                little speed gain when overshoot is rare)
%         'lipschitz_toms748'  -- same as lipschitz_brent in MATLAB (fzero)
%         'doubling'           -- geometric probe beyond Lip radius + bracket
%         'newton_probe'       -- Newton-sized probe via zeta_fn + bracket
%     'root_solver'       (default 'toms748')  'brent'|'toms748'|'bisection'
%                         (MATLAB maps brent/toms748 to fzero; ignored for algorithm1)
%     'zeta_fn'           (default [])  required for method='newton_probe'
%     'return_diagnostics'(default false)  add n_evals, n_steps fields
%     'margin_tol'        (default [])  if set, certify the margin to this
%                         relative precision (see ERROR CONTROL below)
%
%   result fields:
%     M_minus, M_plus, M, converged_minus, converged_plus,
%     mu_minus, mu_plus, method, certificate,
%     status_minus, status_plus, safeguard_minus, safeguard_plus,
%     M_upper_minus, M_upper_plus, M_upper, margin_uncertainty,
%     reason_minus, reason_plus
%     [, n_evals, n_steps]
%
%     'status_*' reports which Algorithm 1 stopping rule fired in that
%     direction -- 'eta_band', 'domain_truncated' or 'iteration_limit' -- and
%     is always populated, independently of margin_tol.  A 'domain_truncated'
%     result certifies only that the margin is at least the distance to the
%     edge of omega, so it must not be read as a resolved margin;
%     'converged_*' cannot distinguish the two and is kept for backward
%     compatibility.  'safeguard_*' is true if the bisection safeguard fired.
%     'reason_*' is a different quantity: the margin_tol bracket outcome.
%
%   Without margin_tol the bracket fields carry sentinels: M_upper* and
%   margin_uncertainty are Inf and reason_* is 'unknown'.
%
%   ERROR CONTROL
%     M is always the distance to a point with F >= FT, hence a lower bound on
%     the true margin: the reported margin is conservative, never optimistic.
%
%     'eta' is a fidelity band, not a margin band.  The induced uncertainty in
%     mu is ~eta/|zeta|, which grows without bound as zeta -> 0, i.e. exactly
%     for the flat, highly robust controllers of interest.  On the paper case
%     study the default eta=1e-6 leaves about 5e-4 relative error in M.
%
%     Set 'margin_tol' to convert that into a margin statement: the safe/unsafe
%     bracket is refined until (M_upper - M)/M <= margin_tol, so the true
%     margin lies in [M, M_upper].  M itself is tightened in the process.
%     margin_tol=1e-10 costs about 90 extra fidelity evaluations per
%     controller and reaches 1e-10, subject to the fp64 floor.
%
%     'certificate' is 'segment' when every point between mu0 and the endpoint
%     is certified F >= FT (algorithm1, lipschitz_*), and 'endpoint' when only
%     the endpoint is (doubling, newton_probe probe beyond the Lipschitz
%     radius, so a dip below FT in between is not excluded).
%
%     'reason_*' is 'bracketed' (an unsafe point was located and refined),
%     'boundary' (the domain edge was reached while still safe -- the margin is
%     a domain truncation, M_upper = Inf), or 'exhausted'.

    p = inputParser;
    addParameter(p, 'mu0', 0);
    addParameter(p, 'eta', 1e-6);
    addParameter(p, 'omega', [-Inf, Inf]);
    addParameter(p, 'k_max', 10000);
    addParameter(p, 'method', 'algorithm1');
    addParameter(p, 'root_solver', 'toms748');
    addParameter(p, 'zeta_fn', []);
    addParameter(p, 'return_diagnostics', false);
    addParameter(p, 'margin_tol', []);
    parse(p, varargin{:});
    margin_tol = p.Results.margin_tol;
    mu0 = p.Results.mu0;
    eta = p.Results.eta;
    omega = p.Results.omega;
    k_max = p.Results.k_max;
    method = lower(char(p.Results.method));
    root_solver = lower(char(p.Results.root_solver));
    zeta_fn = p.Results.zeta_fn;
    return_diagnostics = logical(p.Results.return_diagnostics);

    if L <= 0
        error('qrobustness:margin:L', 'Lipschitz constant L must be positive.');
    end

    valid_methods = {'algorithm1', 'lipschitz_brent', 'lipschitz_toms748', ...
        'doubling', 'newton_probe'};
    if ~any(strcmp(method, valid_methods))
        error('qrobustness:margin:method', 'Unknown method=%s.', method);
    end
    valid_rs = {'brent', 'toms748', 'bisection'};
    if ~any(strcmp(root_solver, valid_rs))
        error('qrobustness:margin:root_solver', 'Unknown root_solver=%s.', root_solver);
    end
    if strcmp(method, 'newton_probe') && isempty(zeta_fn)
        error('qrobustness:margin:zeta', ...
            'method=''newton_probe'' requires ''zeta_fn''.');
    end

    n_evals = 0;
    counted_fn = @counted_fidelity;

    F0 = counted_fn(mu0);
    if ~(FT < F0)
        error('qrobustness:margin:Threshold', ...
            'Require FT < F(mu0); got FT=%g, F=%g.', FT, F0);
    end

    [M_minus, conv_minus, mu_minus, steps_m, status_m, guard_m] = dispatch_one_direction( ...
        counted_fn, L, FT, mu0, eta, omega, k_max, 1, method, root_solver, zeta_fn);
    [M_plus, conv_plus, mu_plus, steps_p, status_p, guard_p] = dispatch_one_direction( ...
        counted_fn, L, FT, mu0, eta, omega, k_max, 2, method, root_solver, zeta_fn);

    result = struct();
    result.M_minus = M_minus;
    result.M_plus = M_plus;
    result.M = min(M_minus, M_plus);
    result.converged_minus = conv_minus;
    result.converged_plus = conv_plus;
    result.mu_minus = mu_minus;
    result.mu_plus = mu_plus;
    result.status_minus = status_m;
    result.status_plus = status_p;
    result.safeguard_minus = guard_m;
    result.safeguard_plus = guard_p;
    result.method = method;
    if any(strcmp(method, {'algorithm1', 'lipschitz_brent', 'lipschitz_toms748'}))
        result.certificate = 'segment';
    else
        result.certificate = 'endpoint';
    end

    % Sentinels matching the Python peer, so the fields exist whether or not
    % margin_tol was requested.
    result.M_upper_minus = Inf;
    result.M_upper_plus = Inf;
    result.M_upper = Inf;
    result.margin_uncertainty = Inf;
    result.reason_minus = 'unknown';
    result.reason_plus = 'unknown';

    if ~isempty(margin_tol)
        if ~(margin_tol > 0)
            error('qrobustness:margin:margin_tol', 'margin_tol must be positive.');
        end
        [lo_m, up_m, why_m] = certify_direction(counted_fn, mu0, mu_minus, FT, 1, omega, margin_tol);
        [lo_p, up_p, why_p] = certify_direction(counted_fn, mu0, mu_plus, FT, 2, omega, margin_tol);
        % The refined safe ends are tighter lower bounds than the eta-based ones.
        result.M_minus = max(result.M_minus, lo_m);
        result.M_plus = max(result.M_plus, lo_p);
        result.M = min(result.M_minus, result.M_plus);
        result.M_upper_minus = up_m;
        result.M_upper_plus = up_p;
        result.M_upper = min(up_m, up_p);
        result.margin_uncertainty = result.M_upper - result.M;
        result.reason_minus = why_m;
        result.reason_plus = why_p;
    end

    if return_diagnostics
        result.n_evals = n_evals;
        result.n_steps = steps_m + steps_p;
    end

    function y = counted_fidelity(mu)
        y = fidelity_fn(mu);
        n_evals = n_evals + 1;
    end
end

function [M_refined, M_upper, reason] = certify_direction(fidelity_fn, mu0, mu_end, FT, ell, omega, margin_tol)
%CERTIFY_DIRECTION Bracket the true ray margin to relative width margin_tol.
%   mu_end has F >= FT, so |mu0-mu_end| is a lower bound on the ray margin.
%   Step outward until F < FT, then bisect, so the true ray margin lies in
%   [M_refined, M_upper] with M_upper - M_refined <= margin_tol*M_refined.

    sign_step = (-1)^ell;
    mu_lo = omega(1);
    mu_hi = omega(2);
    scale = max(abs(mu_end - mu0), 1e-12);

    mu_safe = mu_end;
    mu_unsafe = [];
    step = max(margin_tol * scale, 1e-15);
    for i = 1:200
        cand = min(max(mu_safe + sign_step * step, mu_lo), mu_hi);
        if cand == mu_safe
            M_refined = abs(mu0 - mu_safe);
            M_upper = Inf;
            reason = 'boundary';
            return;
        end
        if fidelity_fn(cand) < FT
            mu_unsafe = cand;
            break;
        end
        mu_safe = cand;
        step = step * 2;
    end
    if isempty(mu_unsafe)
        M_refined = abs(mu0 - mu_safe);
        M_upper = Inf;
        reason = 'exhausted';
        return;
    end

    target = max(margin_tol * max(abs(mu_safe - mu0), 1e-300), ...
                 1e-16 * max(1, abs(mu_safe)));
    for i = 1:200
        if abs(mu_unsafe - mu_safe) <= target
            break;
        end
        mid = 0.5 * (mu_safe + mu_unsafe);
        if mid == mu_safe || mid == mu_unsafe
            break;  % fp64 floor
        end
        if fidelity_fn(mid) >= FT
            mu_safe = mid;
        else
            mu_unsafe = mid;
        end
    end
    M_refined = abs(mu0 - mu_safe);
    M_upper = abs(mu0 - mu_unsafe);
    reason = 'bracketed';
end

function [M, converged, mu_end, n_steps, status, guard] = dispatch_one_direction( ...
        fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, method, root_solver, zeta_fn)
    guard = false;
    switch method
        case 'algorithm1'
            [M, converged, mu_end, n_steps, status, guard] = one_direction_lipschitz( ...
                fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, 'bisection');
        case 'lipschitz_brent'
            [M, converged, mu_end, n_steps, status, guard] = one_direction_lipschitz( ...
                fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, 'brent');
        case 'lipschitz_toms748'
            % MATLAB has no TOMS748; use fzero (Brent-like) with the same API name.
            [M, converged, mu_end, n_steps, status, guard] = one_direction_lipschitz( ...
                fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, 'toms748');
        case 'doubling'
            rs = root_solver;
            if strcmp(rs, 'bisection'), rs = 'toms748'; end
            [M, converged, mu_end, n_steps, status, guard] = one_direction_doubling( ...
                fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, rs);
        case 'newton_probe'
            rs = root_solver;
            if strcmp(rs, 'bisection'), rs = 'toms748'; end
            [M, converged, mu_end, n_steps, status, guard] = one_direction_newton_probe( ...
                fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, rs, zeta_fn);
        otherwise
            error('qrobustness:margin:method', 'Unknown method=%s.', method);
    end
end

function [M, converged, mu_end, n_steps, status, guard] = one_direction_lipschitz( ...
        fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, root_solver)
    sign_step = (-1)^ell;
    mu_lo = omega(1);
    mu_hi = omega(2);
    k = 1;  % counts evaluated trial points, so k_max of them are allowed
    n_steps = 0;
    mu = mu0;
    Fmu = fidelity_fn(mu);
    mu_next = mu;
    F_next = Fmu;
    converged = true;
    guard = false;

    while true
        mu_next = min(max(mu + sign_step * (Fmu - FT) / L, mu_lo), mu_hi);
        F_next = fidelity_fn(mu_next);
        if F_next < FT
            guard = true;
            [mu_next, F_next] = bracket_root_safe( ...
                fidelity_fn, mu, mu_next, FT, eta, root_solver);
        end
        [done, converged, M, status] = stop_one_direction( ...
            mu0, mu_next, F_next, FT, eta, mu_lo, mu_hi, k, k_max);
        if done
            mu_end = mu_next;
            return;
        end
        k = k + 1;
        n_steps = n_steps + 1;
        mu = mu_next;
        Fmu = F_next;
    end
end

function [M, converged, mu_end, n_steps, status, guard] = one_direction_doubling( ...
        fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, root_solver)
    sign_step = (-1)^ell;
    mu_lo = omega(1);
    mu_hi = omega(2);
    n_steps = 0;
    mu_safe = mu0;
    F_safe = fidelity_fn(mu_safe);
    step = max((F_safe - FT) / L, eta / max(L, 1e-30));
    mu_probe = min(max(mu_safe + sign_step * step, mu_lo), mu_hi);
    F_probe = fidelity_fn(mu_probe);
    k = 1;  % counts evaluated trial points, so k_max of them are allowed
    guard = false;

    while F_probe >= FT
        if on_boundary(mu_probe, mu_lo, mu_hi)
            [~, converged, M, status] = stop_one_direction( ...
                mu0, mu_probe, F_probe, FT, eta, mu_lo, mu_hi, k, k_max);
            mu_end = mu_probe;
            return;
        end
        if (F_probe - FT >= 0) && (F_probe - FT < eta)
            M = abs(mu0 - mu_probe);
            converged = true;
            mu_end = mu_probe;
            status = 'eta_band';
            return;
        end
        if k >= k_max
            M = abs(mu0 - mu_probe);
            converged = false;
            mu_end = mu_probe;
            status = 'iteration_limit';
            return;
        end
        mu_safe = mu_probe;
        F_safe = F_probe; %#ok<NASGU>
        step = 2 * step;
        mu_probe = min(max(mu_safe + sign_step * step, mu_lo), mu_hi);
        if abs(mu_probe - mu_safe) <= 0
            % The doubled probe cannot move off mu_safe: the domain edge (or
            % the fp64 floor) is reached while still safe.
            M = abs(mu0 - mu_safe);
            converged = true;
            mu_end = mu_safe;
            status = 'domain_truncated';
            return;
        end
        F_probe = fidelity_fn(mu_probe);
        k = k + 1;
        n_steps = n_steps + 1;
    end

    [mu_end, ~] = bracket_root_safe( ...
        fidelity_fn, mu_safe, mu_probe, FT, eta, root_solver);
    M = abs(mu0 - mu_end);
    converged = true;
    status = 'eta_band';
    guard = true;
end

function [M, converged, mu_end, n_steps, status, guard] = one_direction_newton_probe( ...
        fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, root_solver, zeta_fn)
    sign_step = (-1)^ell;
    mu_lo = omega(1);
    mu_hi = omega(2);
    k = 1;  % counts evaluated trial points, so k_max of them are allowed
    n_steps = 0;
    mu = mu0;
    Fmu = fidelity_fn(mu);
    mu_next = mu;
    F_next = Fmu;
    guard = false;

    while true
        lip_step = (Fmu - FT) / L;
        zeta = zeta_fn(mu);
        if abs(zeta) > 1e-14
            newt_step = abs((Fmu - FT) / zeta);
        else
            newt_step = lip_step;
        end
        step = max(lip_step, newt_step);
        mu_next = min(max(mu + sign_step * step, mu_lo), mu_hi);
        F_next = fidelity_fn(mu_next);
        if F_next < FT
            [mu_next, F_next] = bracket_root_safe( ...
                fidelity_fn, mu, mu_next, FT, eta, root_solver);
            M = abs(mu0 - mu_next);
            converged = true;
            mu_end = mu_next;
            status = 'eta_band';
            guard = true;
            return;
        end
        [done, converged, M, status] = stop_one_direction( ...
            mu0, mu_next, F_next, FT, eta, mu_lo, mu_hi, k, k_max);
        if done
            mu_end = mu_next;
            return;
        end
        if step > lip_step * (1 + 1e-12) && (F_next - FT) >= eta
            step2 = 2 * step;
            mu_probe = min(max(mu_next + sign_step * step2, mu_lo), mu_hi);
            F_probe = fidelity_fn(mu_probe);
            n_steps = n_steps + 1;
            if F_probe < FT
                [mu_next, F_next] = bracket_root_safe( ...
                    fidelity_fn, mu_next, mu_probe, FT, eta, root_solver);
                M = abs(mu0 - mu_next);
                converged = true;
                mu_end = mu_next;
                status = 'eta_band';
                guard = true;
                return;
            end
            mu = mu_next;
            Fmu = F_next;
            mu_next = mu_probe;
            F_next = F_probe;
        end
        k = k + 1;
        n_steps = n_steps + 1;
        mu = mu_next;
        Fmu = F_next;
    end
end

function [done, converged, M, status] = stop_one_direction( ...
        mu0, mu_next, F_next, FT, eta, mu_lo, mu_hi, k, k_max)
    if on_boundary(mu_next, mu_lo, mu_hi) && (F_next - FT >= eta)
        done = true; converged = true; M = abs(mu0 - mu_next);
        status = 'domain_truncated'; return;
    end
    if (F_next - FT >= 0) && (F_next - FT < eta)
        done = true; converged = true; M = abs(mu0 - mu_next);
        status = 'eta_band'; return;
    end
    if k >= k_max
        done = true; converged = false; M = abs(mu0 - mu_next);
        status = 'iteration_limit'; return;
    end
    done = false; converged = true; M = abs(mu0 - mu_next);
    status = 'running';
end

function tf = on_boundary(mu, mu_lo, mu_hi)
    tf = false;
    if isfinite(mu_lo) && abs(mu - mu_lo) <= max(1e-15, 10 * eps * abs(mu_lo))
        tf = true;
    end
    if isfinite(mu_hi) && abs(mu - mu_hi) <= max(1e-15, 10 * eps * abs(mu_hi))
        tf = true;
    end
end

function [mu_safe, F_safe] = bracket_root_safe( ...
        fidelity_fn, mu_safe0, mu_bad, FT, eta, root_solver)
    if strcmp(root_solver, 'bisection')
        [mu_safe, F_safe] = bisect_safe(fidelity_fn, mu_safe0, mu_bad, FT, eta);
        return;
    end

    g = @(mu) fidelity_fn(mu) - FT;
    if g(mu_safe0) < 0
        error('qrobustness:margin:bracket', 'mu_safe must satisfy F >= FT');
    end
    if g(mu_bad) >= 0
        mu_safe = mu_safe0;
        F_safe = fidelity_fn(mu_safe0);
        return;
    end

    xtol = max(eta / 10, 1e-14 * max([1, abs(mu_safe0), abs(mu_bad)]));
    a = min(mu_safe0, mu_bad);
    b = max(mu_safe0, mu_bad);
    % fzero is Brent-like; used for both 'brent' and 'toms748' in MATLAB.
    % TolX must match the Python xtol, otherwise fzero solves to ~eps while
    % brentq/toms748 solve to eta/10 and the two engines land on different
    % points at the xtol level.
    root = fzero(g, [a, b], optimset('TolX', xtol));
    toward_safe = sign(mu_safe0 - root);
    if toward_safe == 0
        toward_safe = sign(mu_safe0 - mu_bad);
        if toward_safe == 0
            toward_safe = 1;
        end
    end
    mu_try = root + toward_safe * xtol;
    mu_try = min(max(mu_try, a), b);
    F_try = fidelity_fn(mu_try);
    if F_try >= FT
        mu_safe = mu_try;
        F_safe = F_try;
        return;
    end
    [mu_safe, F_safe] = bisect_safe(fidelity_fn, mu_safe0, mu_bad, FT, eta);
end

function [mu_safe, F_safe] = bisect_safe(fidelity_fn, mu_safe0, mu_bad, FT, eta)
    % mu_safe0 has F >= FT (caller invariant); mu_bad has F < FT
    a = mu_safe0;
    b = mu_bad;
    Fa = fidelity_fn(a);
    for it = 1:60
        mid = 0.5 * (a + b);
        Fm = fidelity_fn(mid);
        if Fm >= FT
            a = mid;
            Fa = Fm;
            if (Fa - FT) < eta
                break;
            end
        else
            b = mid;
        end
    end
    mu_safe = a;
    F_safe = Fa;
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

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
%     'safe_radius_fn'    (default [])  pluggable certified safe-radius rule
%                         r(F); the default [] reproduces the Lipschitz
%                         surplus rule (F - FT)/L exactly. The Choi-angular
%                         rule of qrobustness.multiparam supplies its own.
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
%     'reason_*' is 'bracketed' (an unsafe point was located and the bracket
%     refined to margin_tol), 'partial' (a rigorous bracket wider than
%     margin_tol, returned when safe-radius continuation stalls below a
%     pointwise-safe sample), 'boundary' (the domain edge was reached while
%     still safe -- the margin is a domain truncation, M_upper = Inf), or
%     'exhausted'. The set matches the Python reference implementation.

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
    addParameter(p, 'safe_radius_fn', []);
    parse(p, varargin{:});
    margin_tol = p.Results.margin_tol;
    mu0 = p.Results.mu0;
    eta = p.Results.eta;
    omega = p.Results.omega;
    k_max = p.Results.k_max;
    method = lower(char(p.Results.method));
    root_solver = lower(char(p.Results.root_solver));
    zeta_fn = p.Results.zeta_fn;
    safe_radius_fn = p.Results.safe_radius_fn;
    if isempty(safe_radius_fn)
        % Default: the Lipschitz surplus rule, reproduced exactly.
        safe_radius_fn = @(F) (F - FT) / L;
    end
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
        counted_fn, L, FT, mu0, eta, omega, k_max, 1, method, root_solver, zeta_fn, safe_radius_fn);
    [M_plus, conv_plus, mu_plus, steps_p, status_p, guard_p] = dispatch_one_direction( ...
        counted_fn, L, FT, mu0, eta, omega, k_max, 2, method, root_solver, zeta_fn, safe_radius_fn);

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
        [lo_m, up_m, why_m] = certify_direction(counted_fn, mu0, mu_minus, FT, 1, omega, margin_tol, L, safe_radius_fn);
        [lo_p, up_p, why_p] = certify_direction(counted_fn, mu0, mu_plus, FT, 2, omega, margin_tol, L, safe_radius_fn);
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

function [M_refined, M_upper, reason] = certify_direction(fidelity_fn, mu0, mu_end, FT, ell, omega, margin_tol, L, safe_radius_fn)
%CERTIFY_DIRECTION Bracket the first component boundary along the ray.
%   mu_end is the endpoint of safe-radius continuation from mu0, so
%   |mu0-mu_end| is a certified lower bound on the ray margin of the NOMINAL
%   safe component. This routine (i) probes outward for an unsafe point,
%   which is a rigorous upper witness because the first boundary precedes
%   it, and (ii) refines the bracket.
%
%   Certified-promotion rule: the certified lower end advances to a
%   pointwise-safe candidate only when the gap from the current certified
%   end is covered by that candidate's own safe radius,
%   |cand - mu_cert| <= safe_radius_fn(F(cand)); the segment then lies in
%   the safe set and connects the candidate to the nominal component. When
%   the gap is larger, continuation steps bridge as far as they certify.
%   Without this rule a pointwise-safe island beyond the first boundary is
%   promoted directly and the reported margin becomes OPTIMISTIC, which
%   contradicts the guarantee in the header of this file. Python is the
%   reference implementation (core._certify_direction); this mirrors it.
%
%   reason is 'bracketed' (width at tolerance), 'partial' (rigorous
%   bracket, width above tolerance), 'boundary' (domain edge reached while
%   certified safe) or 'exhausted' (no unsafe point found).

    if nargin < 8 || isempty(L)
        L = 0.0;
    end
    if nargin < 9
        safe_radius_fn = [];
    end
    if isempty(safe_radius_fn) && L > 0.0
        safe_radius_fn = @(F) (F - FT) / L;
    end

    sign_step = (-1)^ell;
    mu_lo = omega(1);
    mu_hi = omega(2);
    scale = max(abs(mu_end - mu0), 1e-12);

    % Outward geometric probe for an unsafe upper witness; the certified
    % end advances only under the promotion rule.
    mu_cert = mu_end;
    frontier = mu_end;
    mu_unsafe = [];
    step = max(margin_tol * scale, 1e-15);
    for i = 1:200
        cand = min(max(frontier + sign_step * step, mu_lo), mu_hi);
        if cand == frontier
            M_refined = abs(mu0 - mu_cert);
            M_upper = Inf;
            reason = 'boundary';
            return;
        end
        if fidelity_fn(cand) < FT
            mu_unsafe = cand;
            break;
        end
        if promote_ok(fidelity_fn, cand, mu_cert, FT, safe_radius_fn)
            mu_cert = cand;
        else
            % A stall below a pointwise-safe sample leaves the boundary
            % between mu_cert and cand; keep probing for an unsafe witness.
            mu_cert = continue_toward(fidelity_fn, mu_cert, cand, FT, ...
                                      sign_step, safe_radius_fn);
        end
        frontier = cand;
        step = step * 2;
    end
    if isempty(mu_unsafe)
        M_refined = abs(mu0 - mu_cert);
        M_upper = Inf;
        reason = 'exhausted';
        return;
    end

    % Refine: unsafe midpoints always tighten the upper witness; safe
    % midpoints advance the certified end only via the promotion rule or
    % bridged continuation.
    target = max(margin_tol * max(abs(mu_cert - mu0), 1e-300), ...
                 1e-16 * max(1, abs(mu_cert)));
    for i = 1:200
        if abs(mu_unsafe - mu_cert) <= target
            break;
        end
        mid = 0.5 * (mu_cert + mu_unsafe);
        if mid == mu_cert || mid == mu_unsafe
            break;  % fp64 floor
        end
        if fidelity_fn(mid) < FT
            mu_unsafe = mid;
            continue;
        end
        if promote_ok(fidelity_fn, mid, mu_cert, FT, safe_radius_fn)
            mu_cert = mid;
        else
            reached = continue_toward(fidelity_fn, mu_cert, mid, FT, ...
                                      sign_step, safe_radius_fn);
            if reached == mu_cert
                % Continuation stalled: rigorous bracket, above tolerance.
                M_refined = abs(mu0 - mu_cert);
                M_upper = abs(mu0 - mu_unsafe);
                reason = 'partial';
                return;
            end
            mu_cert = reached;
        end
    end
    M_refined = abs(mu0 - mu_cert);
    M_upper = abs(mu0 - mu_unsafe);
    if abs(mu_unsafe - mu_cert) <= max(target, 2e-16 * max(1, abs(mu_cert)))
        reason = 'bracketed';
    else
        reason = 'partial';
    end
end

function ok = promote_ok(fidelity_fn, cand, cert, FT, safe_radius_fn)
%PROMOTE_OK Whether the certified end may advance to cand in one step.
    F = fidelity_fn(cand);
    if F < FT
        ok = false;
        return;
    end
    if isempty(safe_radius_fn)
        % No radius rule: continuation-only promotion, never across a gap.
        ok = (cand == cert);
        return;
    end
    ok = abs(cand - cert) <= safe_radius_fn(F);
end

function cert = continue_toward(fidelity_fn, cert, target_pt, FT, sign_step, safe_radius_fn)
%CONTINUE_TOWARD Safe-radius continuation from cert toward target_pt.
%   Returns the furthest certified point reached.
    if isempty(safe_radius_fn)
        return;
    end
    for i = 1:64
        F = fidelity_fn(cert);
        if F < FT
            return;  % cannot happen for a certified point
        end
        step_r = safe_radius_fn(F);
        if step_r <= abs(target_pt - cert) * 1e-15 + 1e-300
            return;  % stalled at the band
        end
        nxt = cert + sign_step * min(step_r, abs(target_pt - cert));
        if sign_step * (nxt - target_pt) >= 0
            if fidelity_fn(target_pt) >= FT
                cert = target_pt;
            end
            return;
        end
        cert = nxt;
    end
end

function [M, converged, mu_end, n_steps, status, guard] = dispatch_one_direction( ...
        fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, method, root_solver, zeta_fn, safe_radius_fn)
    guard = false;
    switch method
        case 'algorithm1'
            [M, converged, mu_end, n_steps, status, guard] = one_direction_lipschitz( ...
                fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, 'bisection', safe_radius_fn);
        case 'lipschitz_brent'
            [M, converged, mu_end, n_steps, status, guard] = one_direction_lipschitz( ...
                fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, 'brent', safe_radius_fn);
        case 'lipschitz_toms748'
            % MATLAB has no TOMS748; use fzero (Brent-like) with the same API name.
            [M, converged, mu_end, n_steps, status, guard] = one_direction_lipschitz( ...
                fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, 'toms748', safe_radius_fn);
        case 'doubling'
            rs = root_solver;
            if strcmp(rs, 'bisection'), rs = 'toms748'; end
            [M, converged, mu_end, n_steps, status, guard] = one_direction_doubling( ...
                fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, rs, safe_radius_fn);
        case 'newton_probe'
            rs = root_solver;
            if strcmp(rs, 'bisection'), rs = 'toms748'; end
            [M, converged, mu_end, n_steps, status, guard] = one_direction_newton_probe( ...
                fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, rs, zeta_fn, safe_radius_fn);
        otherwise
            error('qrobustness:margin:method', 'Unknown method=%s.', method);
    end
end

function [M, converged, mu_end, n_steps, status, guard] = one_direction_lipschitz( ...
        fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, root_solver, safe_radius_fn)
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
        mu_next = min(max(mu + sign_step * safe_radius_fn(Fmu), mu_lo), mu_hi);
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
        fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, root_solver, safe_radius_fn)
    sign_step = (-1)^ell;
    mu_lo = omega(1);
    mu_hi = omega(2);
    n_steps = 0;
    mu_safe = mu0;
    F_safe = fidelity_fn(mu_safe);
    step = max(safe_radius_fn(F_safe), eta / max(L, 1e-30));
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
        fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, root_solver, zeta_fn, safe_radius_fn)
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
        lip_step = safe_radius_fn(Fmu);
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

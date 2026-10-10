function result = iterative_margin(fidelity_fn, L, FT, varargin)
%ITERATIVE_MARGIN Robustness margin of a scalar perturbation parameter mu.
%
%   result = qrobustness.iterative_margin(fidelity_fn, L, FT, ...)
%
%   Walks from mu0 in both directions until the fidelity F(mu) drops to the
%   threshold F_T and returns the margin M = min(M_minus, M_plus), a lower
%   bound on the distance from mu0 to the nearest point with F < F_T.
%
%   Inputs:
%     fidelity_fn - handle mu -> F(mu)
%     L           - Lipschitz constant of F in mu (e.g. L_j), > 0
%     FT          - fidelity threshold F_T, with F_T < F(mu0)
%
%   Name-value options:
%     'mu0'                - nominal parameter value (default 0)
%     'eta'                - fidelity stopping band (default 1e-6)
%     'omega'              - domain [mu_min, mu_max] (default [-Inf, Inf])
%     'k_max'              - evaluated trial points per direction (default 10000)
%     'method'             - (default 'algorithm1')
%                            'algorithm1': Lipschitz steps plus bisection
%                            'lipschitz_brent', 'lipschitz_toms748': Lipschitz
%                              steps plus fzero polish
%                            'doubling': geometric probes, then bracket
%                            'newton_probe': probes sized via zeta_fn, then bracket
%     'root_solver'        - 'toms748' (default), 'brent' or 'bisection'; brent
%                            and toms748 both use fzero; ignored for algorithm1
%     'zeta_fn'            - handle mu -> zeta(mu); required for newton_probe
%                            (default [])
%     'return_diagnostics' - add evaluation counts (default false)
%     'margin_tol'         - if set, refine the bracket [M, M_upper] until
%                            (M_upper - M)/M <= margin_tol (default [])
%     'safe_radius_fn'     - handle F -> certified safe radius; default []
%                            uses (F - F_T)/L
%     'eval_tol'           - evaluation band (default 0); continuation treats
%                            a point as safe only above FT + eval_tol, and a
%                            probe within the band is unresolved
%
%   Output result fields:
%     M_minus, M_plus, M    - margins below, above mu0 and their minimum
%     mu_minus, mu_plus     - endpoints reached
%     converged_minus, converged_plus - false if the iteration limit was
%                            reached
%     status_minus, status_plus - stopping rule: 'eta_band',
%                            'domain_truncated', 'iteration_limit' or 'stalled'
%     safeguard_minus, safeguard_plus - true if the bisection safeguard fired
%     method                - method used
%     certificate           - 'segment' (every point from mu0 to the endpoint
%                            has F >= F_T; algorithm1, lipschitz_*) or
%                            'endpoint' (only the endpoint; doubling,
%                            newton_probe)
%     M_upper_minus, M_upper_plus, M_upper - upper bounds M_upper (Inf
%                            without margin_tol)
%     margin_uncertainty    - M_upper - M (Inf without margin_tol)
%     reason_minus, reason_plus - margin_tol outcome: 'bracketed' (width at
%                            tolerance), 'partial' (rigorous, wider than
%                            tolerance), 'unresolved' (halted at an
%                            unresolved probe), 'boundary' (domain edge
%                            reached while safe; M_upper = Inf), 'exhausted'
%                            (no unsafe point found); 'unknown' without
%                            margin_tol
%     n_unresolved          - probes within eval_tol of F_T
%     n_evals, n_steps      - fidelity evaluations and steps (with
%                            return_diagnostics)
%     n_evals_minus, n_evals_plus - evaluations per direction (with
%                            return_diagnostics)
%
%   A 'domain_truncated' status only bounds the margin by the distance to the
%   edge of omega.
%
%   Peer of python/src/qrobustness/core.py:iterative_margin.

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
    addParameter(p, 'eval_tol', 0);
    parse(p, varargin{:});
    eval_tol = p.Results.eval_tol;
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
    cache_x = zeros(1, 1024);
    cache_y = zeros(1, 1024);
    counted_fn = @counted_fidelity;

    F0 = counted_fn(mu0);
    if ~(FT + eval_tol < F0)
        error('qrobustness:margin:Threshold', ...
            'Require FT + eval_tol < F(mu0); got FT=%g, eval_tol=%g, F=%g.', FT, eval_tol, F0);
    end

    if eval_tol > 0
        % The true fidelity may be as low as F - eval_tol: take radii there.
        raw_radius = safe_radius_fn;
        safe_radius_fn = @(F) lower_radius(raw_radius, F, FT, eval_tol);
    end

    % Continuation accepts a point as safe only above the evaluation band.
    FT_safe = FT + eval_tol;
    n0 = n_evals;
    [M_minus, conv_minus, mu_minus, steps_m, status_m, guard_m] = dispatch_one_direction( ...
        counted_fn, L, FT_safe, mu0, eta, omega, k_max, 1, method, root_solver, zeta_fn, safe_radius_fn, eval_tol > 0);
    n_m = n_evals - n0;
    n0 = n_evals;
    [M_plus, conv_plus, mu_plus, steps_p, status_p, guard_p] = dispatch_one_direction( ...
        counted_fn, L, FT_safe, mu0, eta, omega, k_max, 2, method, root_solver, zeta_fn, safe_radius_fn, eval_tol > 0);
    n_p = n_evals - n0;

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

    % Sentinel values aligned with the Python peer, so the fields exist whether or not
    % margin_tol was requested.
    result.M_upper_minus = Inf;
    result.M_upper_plus = Inf;
    result.M_upper = Inf;
    result.margin_uncertainty = Inf;
    result.reason_minus = 'unknown';
    result.reason_plus = 'unknown';
    result.n_unresolved = 0;

    if ~isempty(margin_tol)
        if ~(margin_tol > 0)
            error('qrobustness:margin:margin_tol', 'margin_tol must be positive.');
        end
        n0 = n_evals;
        [lo_m, up_m, why_m, unres_m] = certify_direction(counted_fn, mu0, mu_minus, FT, 1, omega, margin_tol, L, ...
                                                         safe_radius_fn, eval_tol);
        n_m = n_m + n_evals - n0;
        n0 = n_evals;
        [lo_p, up_p, why_p, unres_p] = certify_direction(counted_fn, mu0, mu_plus, FT, 2, omega, margin_tol, L, ...
                                                         safe_radius_fn, eval_tol);
        n_p = n_p + n_evals - n0;
        % The refined safe ends form tighter lower bounds than the eta-based ones.
        result.M_minus = max(result.M_minus, lo_m);
        result.M_plus = max(result.M_plus, lo_p);
        result.M = min(result.M_minus, result.M_plus);
        result.M_upper_minus = up_m;
        result.M_upper_plus = up_p;
        result.M_upper = min(up_m, up_p);
        result.margin_uncertainty = result.M_upper - result.M;
        result.reason_minus = why_m;
        result.reason_plus = why_p;
        result.n_unresolved = unres_m + unres_p;
    end

    if return_diagnostics
        result.n_evals = n_evals;
        result.n_evals_minus = n_m;
        result.n_evals_plus = n_p;
        result.n_steps = steps_m + steps_p;
    end

    function y = counted_fidelity(mu)
        % Each point is evaluated once; repeats come from the cache.
        i = find(cache_x(1:n_evals) == mu, 1);
        if ~isempty(i)
            y = cache_y(i);
            return
        end
        y = fidelity_fn(mu);
        n_evals = n_evals + 1;
        if n_evals > numel(cache_x)
            cache_x(2 * n_evals) = 0;
            cache_y(2 * n_evals) = 0;
        end
        cache_x(n_evals) = mu;
        cache_y(n_evals) = y;
    end

end

function [M_refined, M_upper, reason, n_unresolved] = certify_direction(fidelity_fn, mu0, mu_end, FT, ell, omega, margin_tol, ...
                                                                       L, safe_radius_fn, eval_tol)
%CERTIFY_DIRECTION Bracket the first boundary along one direction.
%   mu_end is the endpoint of safe-radius continuation from mu0. Probes
%   outward for an unsafe point, then refines the bracket. The certified end
%   advances to a safe candidate only if |cand - mu_cert| <=
%   safe_radius_fn(F(cand)); otherwise continuation bridges the gap, so a safe
%   point beyond the first boundary is never promoted.
%   Returns [M_refined, M_upper, reason, n_unresolved]; reason as in
%   ITERATIVE_MARGIN.

    if nargin < 8 || isempty(L)
        L = 0.0;
    end
    if nargin < 9
        safe_radius_fn = [];
    end
    if isempty(safe_radius_fn) && L > 0.0
        safe_radius_fn = @(F) (F - FT) / L;
    end
    if nargin < 10 || isempty(eval_tol)
        eval_tol = 0;
    end
    n_unresolved = 0;

    sign_step = (-1)^ell;
    mu_lo = omega(1);
    mu_hi = omega(2);
    scale = max(abs(mu_end - mu0), 1e-12);

    % Geometric probe outward for an unsafe upper witness; the certified
    % end advances only under the promotion rule.
    mu_cert = mu_end;
    frontier = mu_end;
    mu_unsafe = [];
    step = max(margin_tol * scale, 1e-15);
    probe_cap = 200; % qrobustness.core.PROBE_CAP
    for i = 1:probe_cap
        cand = min(max(frontier + sign_step * step, mu_lo), mu_hi);
        if cand == frontier
            M_refined = abs(mu0 - mu_cert);
            M_upper = Inf;
            reason = 'boundary';
            return
        end
        F_cand = fidelity_fn(cand);
        if F_cand < FT - eval_tol
            mu_unsafe = cand;
            break
        end
        if eval_tol > 0 && abs(F_cand - FT) <= eval_tol
            % Neither safe nor unsafe within the evaluation band: probe further out.
            n_unresolved = n_unresolved + 1;
            frontier = cand;
            step = step * 2;
            continue
        end
        if promote_ok(fidelity_fn, cand, mu_cert, FT, safe_radius_fn, eval_tol)
            mu_cert = cand;
        else
            % A stall below a pointwise-safe sample leaves the boundary between mu_cert and cand. Keep probing for an unsafe
            % witness.
            mu_cert = continue_toward(fidelity_fn, mu_cert, cand, FT, ...
                                      sign_step, safe_radius_fn, eval_tol);
        end
        frontier = cand;
        step = step * 2;
    end
    if isempty(mu_unsafe)
        M_refined = abs(mu0 - mu_cert);
        M_upper = Inf;
        reason = 'exhausted';
        return
    end

    % Refine: unsafe midpoints tighten the upper witness; safe midpoints advance the certified end only through the promotion rule
    % or bridged continuation.
    target = max(margin_tol * max(abs(mu_cert - mu0), 1e-300), ...
                 1e-16 * max(1, abs(mu_cert)));
    for i = 1:probe_cap
        if abs(mu_unsafe - mu_cert) <= target
            break
        end
        mid = 0.5 * (mu_cert + mu_unsafe);
        if mid == mu_cert || mid == mu_unsafe
            break   % fp64 floor
        end
        F_mid = fidelity_fn(mid);
        if F_mid < FT - eval_tol
            mu_unsafe = mid;
            continue
        end
        if eval_tol > 0 && abs(F_mid - FT) <= eval_tol
            % An unresolved midpoint moves neither end: the bracket is rigorous, wider than requested.
            n_unresolved = n_unresolved + 1;
            M_refined = abs(mu0 - mu_cert);
            M_upper = abs(mu0 - mu_unsafe);
            reason = 'unresolved';
            return
        end
        if promote_ok(fidelity_fn, mid, mu_cert, FT, safe_radius_fn, eval_tol)
            mu_cert = mid;
        else
            reached = continue_toward(fidelity_fn, mu_cert, mid, FT, ...
                                      sign_step, safe_radius_fn, eval_tol);
            if reached == mu_cert
                % Continuation stalled: rigorous bracket, wider than tolerance.
                M_refined = abs(mu0 - mu_cert);
                M_upper = abs(mu0 - mu_unsafe);
                reason = 'partial';
                return
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

function ok = promote_ok(fidelity_fn, cand, cert, FT, safe_radius_fn, eval_tol)
%PROMOTE_OK Whether the certified end may advance to cand in a single step.
    F = fidelity_fn(cand);
    % eval_tol == 0 keeps F == FT safe. A positive band is closed, so
    % F == FT + eval_tol stays unresolved and is not promoted.
    if eval_tol > 0
        below = F <= FT + eval_tol;
    else
        below = F < FT;
    end
    if below
        ok = false;
        return
    end
    if isempty(safe_radius_fn)
        % No radius rule: promotion by continuation only, never across a gap.
        ok = (cand == cert);
        return
    end
    ok = abs(cand - cert) <= safe_radius_fn(F);
end

function cert = continue_toward(fidelity_fn, cert, target_pt, FT, sign_step, safe_radius_fn, eval_tol)
%CONTINUE_TOWARD Safe-radius continuation starting at cert and heading toward target_pt.
%   Returns the furthest certified point reached.
    if isempty(safe_radius_fn)
        return
    end
    for i = 1:64
        F = fidelity_fn(cert);
        if eval_tol > 0
            below = F <= FT + eval_tol;
        else
            below = F < FT;
        end
        if below
            return
        end
        step_r = safe_radius_fn(F);
        if step_r <= abs(target_pt - cert) * 1e-15 + 1e-300
            return   % stalled at the band
        end
        nxt = cert + sign_step * min(step_r, abs(target_pt - cert));
        if sign_step * (nxt - target_pt) >= 0
            Ft = fidelity_fn(target_pt);
            if (eval_tol > 0 && Ft > FT + eval_tol) || (eval_tol == 0 && Ft >= FT)
                cert = target_pt;
            end
            return
        end
        cert = nxt;
    end
end

function [M, converged, mu_end, n_steps, status, guard] = dispatch_one_direction( ...
        fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, method, root_solver, zeta_fn, safe_radius_fn, strict)
% Run one direction under the chosen method; ell = 1 walks the negative side, ell = 2 the positive one.
    guard = false;
    switch method
        case 'algorithm1'
            [M, converged, mu_end, n_steps, status, guard] = one_direction_lipschitz( ...
                fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, 'bisection', safe_radius_fn, strict);
        case 'lipschitz_brent'
            [M, converged, mu_end, n_steps, status, guard] = one_direction_lipschitz( ...
                fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, 'brent', safe_radius_fn, strict);
        case 'lipschitz_toms748'
            % MATLAB has no TOMS748. fzero (Brent-like) is used under the same API name.
            [M, converged, mu_end, n_steps, status, guard] = one_direction_lipschitz( ...
                fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, 'toms748', safe_radius_fn, strict);
        case 'doubling'
            rs = root_solver;
            if strcmp(rs, 'bisection'); rs = 'toms748'; end
            [M, converged, mu_end, n_steps, status, guard] = one_direction_doubling( ...
                fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, rs, safe_radius_fn, strict);
        case 'newton_probe'
            rs = root_solver;
            if strcmp(rs, 'bisection'); rs = 'toms748'; end
            [M, converged, mu_end, n_steps, status, guard] = one_direction_newton_probe( ...
                fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, rs, zeta_fn, safe_radius_fn, strict);
        otherwise
            error('qrobustness:margin:method', 'Unknown method=%s.', method);
    end
end

function [M, converged, mu_end, n_steps, status, guard] = one_direction_lipschitz( ...
        fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, root_solver, safe_radius_fn, strict)
% Algorithm 1 along one direction: advance by the certified safe radius, so every point between mu0 and the endpoint is certified.
    sign_step = (-1)^ell;
    mu_lo = omega(1);
    mu_hi = omega(2);
    k = 1;  % tallies evaluated trial points, so k_max of them are permitted
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
        if ~is_safe(F_next, FT, strict)
            guard = true;
            [mu_next, F_next] = bracket_root_safe( ...
                fidelity_fn, mu, mu_next, FT, eta, root_solver, strict);
        end
        [done, converged, M, status] = stop_one_direction( ...
            mu0, mu_next, F_next, FT, eta, mu_lo, mu_hi, k, k_max, strict);
        if done
            mu_end = mu_next;
            return
        end
        if sign_step * (mu_next - mu) <= 4 * eps * max(1, abs(mu))
            % No progress beyond floating-point resolution: further steps would repeat it.
            M = abs(mu0 - mu);
            converged = false;
            mu_end = mu;
            status = 'stalled';
            return
        end
        k = k + 1;
        n_steps = n_steps + 1;
        mu = mu_next;
        Fmu = F_next;
    end
end

function [M, converged, mu_end, n_steps, status, guard] = one_direction_doubling( ...
        fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, root_solver, safe_radius_fn, strict)
% Geometric search in one direction: cheaper, yet only the endpoint is certified, since the doubled step may leap over an unsafe
% gap.
    sign_step = (-1)^ell;
    mu_lo = omega(1);
    mu_hi = omega(2);
    n_steps = 0;
    mu_safe = mu0;
    F_safe = fidelity_fn(mu_safe);
    step = max(safe_radius_fn(F_safe), eta / max(L, 1e-30));
    mu_probe = min(max(mu_safe + sign_step * step, mu_lo), mu_hi);
    F_probe = fidelity_fn(mu_probe);
    k = 1;  % tallies evaluated trial points, so k_max of them are permitted
    guard = false;

    while is_safe(F_probe, FT, strict)
        if on_boundary(mu_probe, mu_lo, mu_hi)
            [~, converged, M, status] = stop_one_direction( ...
                mu0, mu_probe, F_probe, FT, eta, mu_lo, mu_hi, k, k_max, strict);
            mu_end = mu_probe;
            return
        end
        if is_safe(F_probe, FT, strict) && (F_probe - FT < eta)
            M = abs(mu0 - mu_probe);
            converged = true;
            mu_end = mu_probe;
            status = 'eta_band';
            return
        end
        if k >= k_max
            M = abs(mu0 - mu_probe);
            converged = false;
            mu_end = mu_probe;
            status = 'iteration_limit';
            return
        end
        mu_safe = mu_probe;
        F_safe = F_probe; %#ok<NASGU>
        step = 2 * step;
        mu_probe = min(max(mu_safe + sign_step * step, mu_lo), mu_hi);
        if abs(mu_probe - mu_safe) <= 0
            % The doubled probe cannot leave mu_safe: the domain edge (or
            % the fp64 floor) is reached while still safe.
            M = abs(mu0 - mu_safe);
            converged = true;
            mu_end = mu_safe;
            status = 'domain_truncated';
            return
        end
        F_probe = fidelity_fn(mu_probe);
        k = k + 1;
        n_steps = n_steps + 1;
    end

    [mu_end, ~] = bracket_root_safe( ...
        fidelity_fn, mu_safe, mu_probe, FT, eta, root_solver, strict);
    M = abs(mu0 - mu_end);
    converged = true;
    status = 'eta_band';
    guard = true;
end

function [M, converged, mu_end, n_steps, status, guard] = one_direction_newton_probe( ...
        fidelity_fn, L, FT, mu0, eta, omega, k_max, ell, root_solver, zeta_fn, safe_radius_fn, strict)
% Slope-guided search in one direction, using zeta_fn to aim the next trial point. Endpoint-certified, as in the doubling search.
    sign_step = (-1)^ell;
    mu_lo = omega(1);
    mu_hi = omega(2);
    k = 1;  % tallies evaluated trial points, so k_max of them are permitted
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
        if ~is_safe(F_next, FT, strict)
            [mu_next, F_next] = bracket_root_safe( ...
                fidelity_fn, mu, mu_next, FT, eta, root_solver, strict);
            M = abs(mu0 - mu_next);
            converged = true;
            mu_end = mu_next;
            status = 'eta_band';
            guard = true;
            return
        end
        [done, converged, M, status] = stop_one_direction( ...
            mu0, mu_next, F_next, FT, eta, mu_lo, mu_hi, k, k_max, strict);
        if done
            mu_end = mu_next;
            return
        end
        if step > lip_step * (1 + 1e-12) && (F_next - FT) >= eta
            step2 = 2 * step;
            mu_probe = min(max(mu_next + sign_step * step2, mu_lo), mu_hi);
            F_probe = fidelity_fn(mu_probe);
            n_steps = n_steps + 1;
            if ~is_safe(F_probe, FT, strict)
                [mu_next, F_next] = bracket_root_safe( ...
                    fidelity_fn, mu_next, mu_probe, FT, eta, root_solver, strict);
                M = abs(mu0 - mu_next);
                converged = true;
                mu_end = mu_next;
                status = 'eta_band';
                guard = true;
                return
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
        mu0, mu_next, F_next, FT, eta, mu_lo, mu_hi, k, k_max, strict)
% The stopping rules, tested in this order: the domain edge, the eta band, and the iteration limit. Returns which one fired, so
% a truncated result is never reported as a resolved one.
    if on_boundary(mu_next, mu_lo, mu_hi) && (F_next - FT >= eta)
        done = true; converged = true; M = abs(mu0 - mu_next);
        status = 'domain_truncated'; return
    end
    if is_safe(F_next, FT, strict) && (F_next - FT < eta)
        done = true; converged = true; M = abs(mu0 - mu_next);
        status = 'eta_band'; return
    end
    if k >= k_max
        done = true; converged = false; M = abs(mu0 - mu_next);
        status = 'iteration_limit'; return
    end
    done = false; converged = true; M = abs(mu0 - mu_next);
    status = 'running';
end

function tf = on_boundary(mu, mu_lo, mu_hi)
% Whether mu lies on a finite domain edge, up to floating-point slack.
    tf = false;
    if isfinite(mu_lo) && abs(mu - mu_lo) <= max(1e-15, 10 * eps * abs(mu_lo))
        tf = true;
    end
    if isfinite(mu_hi) && abs(mu - mu_hi) <= max(1e-15, 10 * eps * abs(mu_hi))
        tf = true;
    end
end

function [mu_safe, F_safe] = bracket_root_safe( ...
        fidelity_fn, mu_safe0, mu_bad, FT, eta, root_solver, strict)
% Step back from an unsafe point to a safe one inside the bracket. The safeguard fires when an under-estimated L let a
% certified step overshoot the threshold.
    if strcmp(root_solver, 'bisection')
        [mu_safe, F_safe] = bisect_safe(fidelity_fn, mu_safe0, mu_bad, FT, eta, strict);
        return
    end

    g = @(mu) fidelity_fn(mu) - FT;
    if ~is_safe(fidelity_fn(mu_safe0), FT, strict)
        error('qrobustness:margin:bracket', 'mu_safe must lie on the safe side of FT');
    end
    if is_safe(fidelity_fn(mu_bad), FT, strict)
        mu_safe = mu_safe0;
        F_safe = fidelity_fn(mu_safe0);
        return
    end

    if ~(g(mu_safe0) > 0 && g(mu_bad) < 0)
        [mu_safe, F_safe] = bisect_safe(fidelity_fn, mu_safe0, mu_bad, FT, eta, strict);
        return
    end

    xtol = max(eta / 10, 1e-14 * max([1, abs(mu_safe0), abs(mu_bad)]));
    a = min(mu_safe0, mu_bad);
    b = max(mu_safe0, mu_bad);
    % fzero serves both 'brent' and 'toms748'. TolX matches the Python xtol (eta/10) so both engines stop at the same tolerance.
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
    if is_safe(F_try, FT, strict)
        mu_safe = mu_try;
        F_safe = F_try;
        return
    end
    [mu_safe, F_safe] = bisect_safe(fidelity_fn, mu_safe0, mu_bad, FT, eta, strict);
end

function [mu_safe, F_safe] = bisect_safe(fidelity_fn, mu_safe0, mu_bad, FT, eta, strict)
    % mu_safe0 has F >= FT (caller invariant); mu_bad has F < FT
    a = mu_safe0;
    b = mu_bad;
    Fa = fidelity_fn(a);
    for it = 1:60
        mid = 0.5 * (a + b);
        Fm = fidelity_fn(mid);
        if is_safe(Fm, FT, strict)
            a = mid;
            Fa = Fm;
            if (Fa - FT) < eta
                break
            end
        else
            b = mid;
        end
    end
    mu_safe = a;
    F_safe = Fa;
end

function tf = is_safe(F, FT, strict)
% Safe side of FT. A positive band (strict) leaves equality unresolved.
    if strict
        tf = F > FT;
    else
        tf = F >= FT;
    end
end

function r = lower_radius(raw_radius, F, FT, eval_tol)
    % Safe radius at F - eval_tol; zero unless F - eval_tol exceeds FT.
    if F - eval_tol > FT
        r = raw_radius(F - eval_tol);
    else
        r = 0;
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later

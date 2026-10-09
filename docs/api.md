# API reference

Python is the reference implementation (`import qrobustness`). MATLAB and
Octave provide the peer package `+qrobustness` (`help qrobustness`) with the
same names, inputs and outputs. Where Python returns a dataclass, the peer
returns a struct. The mathematics, the experiments and the results are in
the papers: QRM (scalar margin, Algorithm 1) and xQRM (joint, trajectory,
open-system and state certificates, Algorithm 2). The symbols below are the
papers'.

## Conventions

- Units with hbar = 1; dense complex matrices; Hamiltonians Hermitian.
- Piecewise-constant controls on tau intervals of length Delta = t_f/tau;
  `H_list` is the list of interval Hamiltonians H^(k), `dt` is Delta.
- Propagator U(t_f) = U^(tau) ... U^(1), U^(k) = exp(-i Delta H^(k)).
- Gate fidelity F = |Tr(U_f^dagger U)|/N (trace amplitude); `FT` is the
  threshold F_T. Process fidelity F^pro = F^2 for unitaries.
- A structure is a list `Hhat_list` (or `dH_list`) of interval
  perturbations \hat H^(k); the perturbed generator is H^(k) + mu \hat H^(k).
- Open systems use column-stacked superoperators.

## Defaults

| Symbol | Argument | Default |
| :--- | :--- | :--- |
| F_T | `FT` | none (drivers use 0.999) |
| eta (continuation band) | `eta` | `1e-6` |
| eps (relative bracket tolerance) | `margin_tol` | `None` (no bracket) |
| eps_num (evaluation band) | `eval_tol` | `0` |
| K_max (continuation steps per ray) | `k_max` | `10000` |
| Omega (admissible ray interval) | `omega` | `(-inf, inf)` |

## Margins: `iterative_margin`

`iterative_margin(fidelity_fn, L, FT, mu0=0, eta=1e-6, omega=(-inf, inf),
k_max=10000, method="algorithm1", root_solver="toms748", zeta_fn=None,
return_diagnostics=False, margin_tol=None, safe_radius_fn=None,
eval_tol=0)` returns a `MarginResult`.

Along each ray from `mu0` the continuation steps by the safe radius.
The default radius is (F - F_T)/L. `safe_radius_fn` supplies another, for
example the angular radius. The walk stops when the fidelity surplus is
below `eta`. With `margin_tol` it then searches outward for an unsafe point
and refines the bracket [M, M_upper]. With `eval_tol = 0`, a point is safe when F >= F_T. With `eval_tol > 0`, a
point is safe only when F > F_T + eval_tol. The continuation and the bracket
use that same test. A point is unsafe only if F < F_T - eval_tol. A point
between those bounds is unresolved. The safe radius uses the computed fidelity.

| `method` | step | overshoot polish | certificate |
| :--- | :--- | :--- | :--- |
| `algorithm1` (default) | safe radius | bisection | segment |
| `lipschitz_brent`, `lipschitz_toms748` | safe radius | Brent / TOMS748 | segment |
| `doubling` | geometric probe | `root_solver` | endpoint |
| `newton_probe` | Newton probe via `zeta_fn` | `root_solver` | endpoint |

`MarginResult` fields:

| field | meaning |
| :--- | :--- |
| `M_minus`, `M_plus`, `M` | certified margins per ray and their minimum (lower bounds) |
| `mu_minus`, `mu_plus` | end points of the continuation |
| `status_minus`, `status_plus` | continuation stop: `eta_band`, `domain_truncated` (margin is at least the distance to the edge of `omega`), `iteration_limit` |
| `M_upper_minus`, `M_upper_plus`, `M_upper` | evaluated unsafe witnesses (`inf` if none) |
| `margin_uncertainty` | `M_upper - M` |
| `reason_minus`, `reason_plus` | bracket outcome: `bracketed` (width <= eps), `partial` (valid bracket, wider), `unresolved` (an in-band probe stopped refinement), `boundary` (edge of `omega` reached while safe), `exhausted` (no unsafe point found), `zero_gauge` (direction leaves the dynamics unchanged) |
| `n_unresolved` | probes inside the evaluation band |
| `certificate` | `segment` or `endpoint` |
| `converged_*`, `safeguard_*` | continuation converged; overshoot polish was needed |
| `n_evals`, `n_evals_minus`, `n_evals_plus`, `n_steps` | with `return_diagnostics`: evaluations in total (both rays plus `mu0`), per ray, and continuation steps |

The paper's statuses correspond as follows: resolved = `bracketed`;
lower certificate only = `boundary`, `exhausted`, `zero_gauge`;
unresolved = `unresolved`; `partial` is a valid bracket wider than eps.

## Modules

Functions marked (s) are reached through their submodule
(`qrobustness.states`, `qrobustness.openstates`, `qrobustness.kosut`,
`qrobustness.berberich`); all others are also exported at package level.

### `core`

| function | in | out |
| :--- | :--- | :--- |
| `propagator(H_list, dt)` | interval Hamiltonians | U(t_f) |
| `gate_fidelity(U, Uf)` | unitaries | F |
| `lipschitz_constant(FT, N, C_H)` | F_T, dimension N, C_{\hat H} | L = B_T C_{\hat H}, B_T = sqrt((1 - F_T^2)/N) |
| `traceless(Hhat)` | matrix | traceless part |
| `structure_constant(kind, Hhat, dt, tau, controls=None)` | `kind` "drift" or "control" | C_{\hat H} of the centred structure |
| `perturbed_hamiltonians`, `dH_structure`, `make_fidelity_fn` | QRM case-study model (H0, H1, H2, u1, u2) | perturbed H_list, structure list, F(mu) |
| `differential_sensitivity(H_list, dH_list, dt, Uf, n_quad=32, method="exact")` | structure at the evaluation point | zeta = dF/dmu; `method="quadrature"` uses `n_quad` Gauss-Legendre nodes |
| `segment_eig`, `segment_propagator`, `dU_dmu_exact`, `dU_dmu_integral`, `gauss_legendre_01` | interval data | per-interval propagator and derivative |
| `iterative_margin(...)` | see above | `MarginResult` |
| `fidelity_vs_delta(fidelity_fn, delta_grid)` | grid | (grid, F) for plots |
| `load_problem(path)`, `load_controllers(path, max_error=1e-4)` | MAT file, controller CSV | dict `H0, H1, H2, Uf, n_qubits, dim`; list of controllers with eps_0 <= `max_error` |

### `multiparam`

| function | out |
| :--- | :--- |
| `structure_constants(specs, dt, tau, FT, N)` | (C_j, L_j) for specs `("drift", H)` / `("control", H, u)` |
| `safe_polytope(centre, L, F, FT)` | `SafePolytope` sum_j L_j \|x_j\| <= F - F_T (`axis_radii`, `inradius_l2`, `inradius_linf`, `contains`, `boundary_point`) |
| `joint_gauge(Hhat_lists, dt)` | `JointGauge`: C_joint(x) = Delta sum_k sqrt(x^T P^(k) x) (`C`, `L_dir`, `contains`, `boundary_radius`, `inradius_certified`) |
| `angular_gauge(Hhat_lists, dt)` | `AngularGauge`: C^stat_FS(x) = Delta sum_k sqrt(x^T Q^(k) x), budget arccos F_T - arccos F (`C`, `budget`, `contains`, `boundary_radius`, `inradius_certified`) |
| `directional_margin(fidelity_fn, L, FT, d, mu0=None, L_dir=None, angular_gauge=None, **kw)` | `MarginResult` along mu0 + s d; with `angular_gauge` it steps by the angular radius |
| `make_multiparam_fidelity_fn`, `make_ray_fn` | F(mu) on the joint model, F(s) on a ray |
| `axis_directions(p)`, `diagonal_directions(p)`, `sphere_directions(p, n, seed)` | direction designs |
| `SafeUnion` | union of certified polytopes |

### `lengthspace`

`interval_grams(Hhat_lists, make_traceless=False, normalise=False)` (P^(k);
Q^(k) with both flags), `angle_budget(F0, FT)` (arccos F_T - arccos F_0),
`margin_from(budget, speed)` (budget/speed, `inf` at zero speed),
`refine(H_list, Hhat_list, q)` (q-fold interval refinement), `PathGauge`,
`traceless`.

### `timevarying`

| function | out |
| :--- | :--- |
| `uniform_margin(L, F, FT)` | r_0 = (F - F_T)/sum L_j |
| `fs_margin(Hhat_list, dt, F0, FT, r0=0)` | `FSMargin`: speed s, theta_0, r_FS |
| `fs_margin_joint(Hhat_lists, dt, F0, FT)` | trajectory box gauge (vertex and separable) |
| `tv_fidelity_and_gradient(H_list, Hhat_list, delta, dt, Uf)` | F[delta] and its gradient for a piecewise-constant trajectory |
| `adversarial_fidelity(H_list, Hhat_list, dt, Uf, m, ...)` | smallest fidelity found, the trajectory, and the number of fidelity evaluations |
| `toggling_frame_integral(H_list, dHhat_list, dt)` | toggling-frame integral of a structure |
| `adversarial_upper_bound(H_list, Hhat_list, dt, Uf, FT, r0, m_hi, rel_tol=0.01, seed=None, n_starts=4, starts="legacy", maxiter=200)` | `TVBracket` [r_0, m_adv]; `n_evals` counts fidelity evaluations; m_adv is a found violating budget (heuristic search), not a certificate |

### `lindblad`

Builders `hamiltonian_superop`, `dissipator`, `generator(H, Vs, gammas)`,
`channel(G_list, dt)`, `unitary_superop`; `process_fidelity`,
`average_gate_fidelity`; `frechet_derivative(G, E, dt)`; Choi
`choi_matrix`, `superop_from_choi`, `choi_roundtrip_exact`; diamond norms
`diamond_norm(S, solver)` (SDP, needs `qrobustness[open]`),
`diamond_norm_free(S)` (no solver), `common_rate_local_dnorm(n)` (2n for
common-rate local dephasing and amplitude damping),
`hamiltonian_part(S)` / `hamiltonian_dnorm(S)` (Rump-verified
lambda_max - lambda_min for -i[B, .]), `local_ops`, `local_dephasing_ops`.
Each diamond-norm function returns a `DiamondNorm` whose
`value_certified` is a verified upper bound; a failed verification raises
`VerificationFailure`. Margins: `open_structure_constants(G_structs, dt)`
(C^op_j, L^op_j; `dt` is a scalar or one length per interval), `make_open_fidelity_fn`, `open_margin(fidelity_fn, L,
FT_pro, omega=(0, inf))` (F_T^pro = F_T^2). Coherence times
`dephasing_time`, `relaxation_time`, `coherence_time`, `rates_from_times`.

### `states` (s) and `openstates` (s)

`half_spread(H)` (||X||_c), `state_speed(Hhat_list, dt)` (C^st),
`state_speed_joint(Hhat_lists, dt, x)`, `propagate_state`, `state_fidelity`
(f = \|<chi\|psi>\|), `fs_angle`, `state_lipschitz_constant(FT, C)`,
`state_angular_margin(fidelity_fn, C, FT, angular=True, **kw)`,
`trajectory_radius(budget, speed)`; preparation:
`nondegenerate_eigenvector(H, index=0)` (state, eigenvalue, gap gamma),
`preparation_speed(dH, psi0, gap, bound="sigma")` (sigma/gamma or
\|\|dH\|\|_c/gamma, a bound on C_prep), `gapped_preparation_radius(budget, gap,
prep_spread, evolution_speed=0)`. Open: `open_speed(Ghat_list, dt, norm=None,
exact_hamiltonian=False)` (D), `evolve_density`, `trace_distance`,
`open_state_fidelity_margin(fidelity_fn, D, FT, omega)`.

### Comparison bounds: `kosut` (s) and `berberich` (s)

`kosut.uncertainty_rates(H_list, dH_list, dt, ...)` (`UncertaintyRates`:
w_unc, w_avg, w_dev per unit delta, with the w_dev error-control fields;
`n_quad` is accepted and ignored), `time_bandwidth`, `fidelity_bound`,
`fidelity_bound_at`, `effective_threshold(FT, nominal_error, absorption)`,
`threshold_time_bandwidth`, `margin(rates, FT, nominal_error=0,
absorption="angular", uncertainty="constant")` (M^K; `"trajectory"` gives
M^{K,tri}_tv; package-level alias `kosut_margin`). `absorption="additive"`
is kept for the QRM tables and is not a sufficient condition.
`berberich.margin(H_list, dH_list, dt, FT, nominal_error=0,
uncertainty="independent", rates=None)` (`BerberichMargin`: M^B_tv for
`"independent"`, M^B for `"systematic"`; `"systematic"` raises when the
Magnus condition fails).

### `synthesis`, `optimize`, `verify`, `plotting`

`synthesis.grape`, `grape_ensemble`, `grape_robust` (L-BFGS-B GRAPE with
seeded standard-normal initialisation; `GrapeResult`),
`fidelity_and_control_gradient`. `optimize.optimize_controller`,
`fidelity_and_gradient`, `pack_controls`, `unpack_controls` (QRM workflow;
MATLAB uses `fminunc`). `verify.check_*` (numerical checks of each
certified inequality; `CheckReport` with probes, minimum slack, tolerance,
arg-min), `unitarity_defect`, `fidelity_cross_check`. `plotting`: the QRM
figures (`plot_margins_vs_index`, `plot_margins_vs_sensitivity`,
`plot_fidelity_error_sweeps`, `log10_axis`, `apply_plot_style`,
`save_fig`; needs `qrobustness[plot]`). `ylim` is optional. Omitted, the
historical limits stay when every positive point lies inside them, and
widen when a point falls outside.

## Accuracy classes

| quantity | class |
| :--- | :--- |
| propagators, fidelities, structure constants, Grams, gauges, r_0, r_FS, w_unc, w_avg | exact to roundoff (closed form) |
| zeta (`method="exact"`) | exact to roundoff (divided difference) |
| M | certified lower bound; without `margin_tol` its relative error is of order eta/\|zeta\| |
| [M, M_upper] | bracket refined to `margin_tol` when reason is `bracketed` |
| w_dev | sampled supremum with two certificates; residual error optimistic |
| diamond norms (`value_certified`) | verified upper bound of the represented superoperator |
| m_adv, adversarial minima | numerically validated witnesses of a heuristic search |

## Paper symbols and code names

| paper | code |
| :--- | :--- |
| F_T, F_T^pro | `FT`, `FT_pro` |
| C_{\hat H}, B_T, L = B_T C | `structure_constant`, inside `lipschitz_constant`, `lipschitz_constant` |
| C_j, L_j | `structure_constants` |
| C_joint, L_dir, P^(k) | `JointGauge.C`, `JointGauge.L_dir`, `interval_grams(make_traceless=True)` |
| C^stat_FS, Q^(k) | `AngularGauge.C`, `interval_grams(make_traceless=True, normalise=True)` |
| s_j, theta_0, r_0, r_FS | `FSMargin.speed`, `theta_0`, `r0`, `r_fs` |
| M, M_upper, M_const, m_adv | `MarginResult.M`, `M_upper`, CSV `M_const`, `TVBracket.m_adv` |
| eps, eta, eps_num | `margin_tol`, `eta`, `eval_tol` |
| C^op_j = L^op_j, D | `open_structure_constants`, `open_speed` |
| \|\|X\|\|_c, C^st, C_prep bound, gamma (gap) | `half_spread`, `state_speed`, `preparation_speed`, `nondegenerate_eigenvector` |
| w_unc, w_avg, w_dev, \bar w | `UncertaintyRates.w_unc`, `w_avg`, `w_dev`, `w_avg_traj` |
| M^K, M^{K,tri}_tv | `kosut.margin(uncertainty="constant")`, `uncertainty="trajectory"` |
| M^B, M^B_tv | `berberich.margin(uncertainty="systematic")`, `uncertainty="independent"` |

## MATLAB and Octave

The peer covers `core`, `multiparam`, `lengthspace`, `timevarying`,
`lindblad` (diamond norm solver-free under the plain name `diamond_norm`;
Python's SDP `diamond_norm` has no peer), `states`, `openstates`, `kosut`,
`berberich`, `optimize` and `plotting`. Python-only: `synthesis` (GRAPE),
`verify`, and the adversarial search in `timevarying`. Peers are struct
based rather than `classdef`, so they run unchanged under both engines.

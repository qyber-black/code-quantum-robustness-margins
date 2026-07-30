# Shared API contract (Python <-> MATLAB)

For *what* these functions compute -- the mathematics, the paper cross-references, and the accuracy of each quantity -- see [theory.md](theory.md).

Both languages expose the same conceptual API. Python is the reference implementation and uses the module `qrobustness`; MATLAB uses the package `qrobustness.*`.

## Conventions

- Units with \(\hbar = 1\).
- Complex dense matrices; Hamiltonians are Hermitian.
- Piecewise-constant controls on \(\tau\) intervals of length \(\Delta = t_f/\tau\).
- Ordered product: \(U(t_f) = U^{(\tau)}\cdots U^{(1)}\) with \(U^{(k)}=\exp(-\mathrm{i} H^{(k)}\Delta)\).
- Gate fidelity: \(\mathcal{F} = \frac{1}{N}\lvert\mathrm{Tr}(U_f^\dagger U)\rvert\).
- Perturbation parameter \(\delta = \mu - \mu_0\) with \(\mu_0 = 0\) by default in the case study (multiplicative structure on \(H_j\)).

## Numerical defaults

| Symbol | Name | Default |
|--------|------|---------|
| \(\mathcal{F}_T\) | `fidelity_threshold` | `0.999` (case study) |
| \(\eta\) | `eta` | `1e-6` |
| \(K_{\max}\) | `k_max` | `10000` |
| \(\Omega_\mu\) | `omega` | `[-Inf, Inf]` unless set |
| Consistency tol | `rtol` / `atol` | `1e-8` / `1e-10` (tests) |

## Functions

### `propagator(H_list, dt)`

- **In:** cell/list of Hermitian matrices \(H^{(k)}\); step `dt`.
- **Out:** unitary \(U(t_f)\).

### `gate_fidelity(U, Uf)`

- **Out:** scalar \(\mathcal{F}\in[0,1]\).

### `lipschitz_constant(FT, N, C_H)`

- \(B_T = \sqrt{(1-\mathcal{F}_T^2)/N}\), \(L = B_T\, C_{\hat{H}}\).
- Helpers build \(C_{\hat{H}}\):
  - drift: \(C = t_f \|H_0\|_F\)
  - control \(m\): \(C = \Delta \|f_m\|_{\ell^1}\|H_m\|_F\)

### `structure_constant(kind, Hhat, dt, tau, controls=None)`

- `kind`: `"drift"` or `"control"`.
- Returns \(C_{\hat{H}}\).

### `differential_sensitivity(H_list, dH_list, dt, Uf, n_quad=32, method="exact")`

- Implements the paper's \(\zeta\) via the product-derivative form, inserting \(\partial U^{(k)}/\partial\mu\) into the ordered product.
- `dH_list[k] = \partial\tilde{H}^{(k)}/\partial\mu` at the evaluation point.
- `method="exact"` (default) evaluates \(\partial U^{(k)}/\partial\mu\) in closed form in the eigenbasis of \(H^{(k)}\). Since the controls are piecewise constant, \(H^{(k)}\) is constant on the interval and the defining integral is a divided difference -- so this is exact to roundoff, not an approximation.
- `method="quadrature"` uses Gauss-Legendre with `n_quad` nodes. Retained as an alternative and as a cross-check on the closed form; it is not needed for accuracy.
- `n_quad` applies only to `method="quadrature"`; under the default it is accepted and unused. A positional form, `differential_sensitivity(..., Uf, 32)`, is accepted.
- The same `method` / `n_quad` options are accepted by `fidelity_and_gradient` and threaded through `optimize_controller`. MATLAB takes them as name-value pairs; `qrobustness.DU_METHODS` lists the valid values.
- On the exact path one eigendecomposition per interval serves both the propagator and the derivatives, so \(U^{(k)}\) and \(\partial U^{(k)}/\partial\mu\) are exactly consistent. `propagator()` uses `expm`, so fidelities computed the two ways can differ at ~1e-15.

### `iterative_margin(fidelity_fn, L, FT, mu0=0, eta=1e-6, omega=(-inf,inf), k_max=10000, method="algorithm1", root_solver="toms748", zeta_fn=None, return_diagnostics=False, margin_tol=None)`

The default `method="algorithm1"` is Algorithm 1 of the paper, used by the paper drivers and the goldens:

- For each direction \(\ell\in\{1,2\}\) (decrease / increase):
  - step \(\mu \leftarrow \mu + (-1)^\ell (\mathcal{F}_\mu-\mathcal{F}_T)/L\)
  - clamp to \(\Omega_\mu\)
  - if overshoot \(\mathcal{F}<\mathcal{F}_T\), bisect back to a safe point
  - stop on tolerance, domain boundary, or \(K_{\max}\)

Selectable `method` (default stays Algorithm 1 for paper reproduction):

| `method` | Advance | Overshoot polish | Certificate |
|----------|---------|------------------|-------------|
| `algorithm1` | Lipschitz \((F-F_T)/L\) | in-place bisection | Full (paper; preferred certified default) |
| `lipschitz_brent` | same Lipschitz | Brent (`scipy.optimize.brentq` / MATLAB `fzero`) | Full (optional polish; little speed gain vs bisection) |
| `lipschitz_toms748` | same Lipschitz | TOMS748 (Python); MATLAB maps to `fzero` | Full (same as Brent polish) |
| `doubling` | geometric probe beyond Lip radius | Brent/TOMS748 | Endpoint-only unless monotone |
| `newton_probe` | Newton-sized probe via `zeta_fn` | Brent/TOMS748 | Same as `doubling` when probing |

Lipschitz + bisection is the recommended certified path; Brent/TOMS748 are optional polish only. See [margin-solvers-notes.md](margin-solvers-notes.md).

- `root_solver`: `toms748` (default), `brent`, or `bisection` -- used by non-`algorithm1` bracket polish; ignored for `algorithm1`.
- `zeta_fn(mu)`: required for `newton_probe`.
- `return_diagnostics=True`: also sets `n_evals`, `n_steps`, `method` on the result.
- `margin_tol`: if set, certify the margin to this relative precision. The safe/unsafe bracket is refined until `(M_upper - M)/M <= margin_tol`, and `M` itself is tightened. See below.
- **Returns:** `M_minus`, `M_plus`, `M = min(M_minus, M_plus)`, `converged_minus`, `converged_plus`, `mu_minus`, `mu_plus`, `certificate`; and with `margin_tol`: `M_upper_minus`, `M_upper_plus`, `M_upper`, `margin_uncertainty`, `reason_minus`, `reason_plus`.

#### Accuracy of the margin

`M` is always the distance to a point at which `F >= FT` was evaluated, so it is a lower bound: conservative, never optimistic.

`eta` is a fidelity band, not a margin band. The induced uncertainty in `mu` is `~eta/|zeta|`, unbounded as `zeta -> 0` (i.e. for the flattest, most robust controllers). On the case study the default `eta=1e-6` leaves about 5e-4 relative error in `M`, so `margins_table_0.999.csv` carries ~3-4 significant figures of margin despite printing 6. Pass `margin_tol` to fix this: `margin_tol=1e-10` reaches 1e-10 in about 90 extra fidelity evaluations. Paper drivers do not pass it, so published tables reproduce exactly.

`certificate` is `'segment'` when every point between `mu0` and the endpoint is covered by a safe-radius certificate (`algorithm1`, `lipschitz_*`), and `'endpoint'` when only the endpoint is verified (`doubling`, `newton_probe` probe beyond the Lipschitz radius). `reason_*` is `'bracketed'`, `'boundary'` (domain edge reached while still safe -- the margin is a domain truncation, `M_upper = inf`), or `'exhausted'`.

`fidelity_fn(mu)` must return \(\mathcal{F}_\mu\). See [margin-solvers-notes.md](margin-solvers-notes.md) for guarantees and the bench harness.

### `fidelity_vs_delta(fidelity_fn, delta_grid)`

- Dense sweep for plotting only (not the certificate).

### Plotting (Python: `qrobustness.plotting`; optional `matplotlib`)

Mirrors MATLAB `+qrobustness` helpers. Decade axes use \(\log_{10}\) of values on linear axes, not log scales.

| Function | Role |
|----------|------|
| `plot_margins_vs_index(err, M0, M1, M2)` | Controllers sorted by \(\varepsilon_0\); \(\log_{10}\) of error/margins on linear \(y\) |
| `plot_margins_vs_sensitivity(\|z0\|,\|z1\|,\|z2\|,M0,M1,M2)` | Two-panel \(M\) vs \(\log_{10}\lvert\zeta\rvert\) |
| `plot_fidelity_error_sweeps(X_list, Y_list, FT)` | Spaghetti fidelity-error vs \(\delta\); \(\log_{10}\) error on linear \(y\) |
| `log10_axis(ax, which, raw_lim)` | Decade tick labels for pre-transformed data |
| `apply_plot_style(fig)` | White background / black axes |

Install: `pip install 'qrobustness[plot]'`. Full paper runs:

- `make lipschitz-margin-matlab` -> `results/lipschitz-margin-matlab/`
- `make lipschitz-margin-python` -> `results/lipschitz-margin-python/`
- `make lipschitz-margin-octave` -> `results/lipschitz-margin-octave/` (optional peer; not in `check-margins`)
- `make verify-paper-matlab` / `make verify-paper-python` / `make verify-paper-octave` -> `verify_paper.md` in each tree
- `make compare-octave` -> MATLAB vs Octave margin tables
- `make check-margins` -> `lipschitz-margin-matlab` + `lipschitz-margin-python` + `compare-full` + `verify-paper` (release gate)
- `make synth-matlab` / `make synth-python` -> `results/synth-*/` (does not touch paper controllers)
- `make analyse-synth-matlab` / `make analyse-synth-python` -> margin trees for synth sets
- `make sync-paper-matlab` / `make sync-paper-python` / `make sync-paper-octave` -> `../figures/`

### Case-study helpers

- `load_problem(path_mat)` -> `H0, H1, H2, Uf, n_qubits, dim`. Note `problem.N` in the MAT file is the number of qubits; `dim = 2**n_qubits` is the Hilbert space dimension and is what `lipschitz_constant` expects. Both are returned under unambiguous names rather than a single overloaded `N`.
- `load_controllers(path_csv, max_error=1e-4)` -> list of `{fid, error, u1, u2, tf, tau}`
- `perturbed_hamiltonians(H0, H1, H2, u1, u2, structure, delta)` -> `H_list` for structure in `{H0,H1,H2}`

### Synthesis (fidelity maximisation)

Paper-aligned defaults: \(t_f=15\), \(\tau=32\), Gaussian init \(\mathcal{N}(0,1)\), GRAPE gradient of \(\mathcal{F}\).

| Function | Role |
|----------|------|
| `fidelity_and_gradient(H0,H1,H2,u1,u2,Uf,dt)` | \(\mathcal{F}\) and \(\partial\mathcal{F}/\partial u_m\) |
| `optimize_controller(...)` | Single-run quasi-Newton / L-BFGS-B maximize \(\mathcal{F}\) |

MATLAB uses `fminunc` (quasi-Newton); Python uses `scipy.optimize.minimize(..., method="L-BFGS-B")`. Final errors need not match bit-for-bit across languages; hard consistency remains on the analysis API.

Ensemble drivers write `results/synth-*/{controllers.csv,meta.json,problem9.mat}`. Analysis drivers accept an arbitrary controller directory (`controller_dir` / `--controller-dir`).

### Kosut et al. time-bandwidth bound (`qrobustness.kosut`)

Supplementary and experimental; outside the reproduction gate `make check-margins`.
See [time-bandwidth-bound.md](time-bandwidth-bound.md) for the specialisation,
the caveats and the results.

```
uncertainty_rates(H_list, dH_list, dt, n_quad=None, n_dev=17, dev_tol=1e-9,
                  n_dev_max=4097, adaptive_dev=True, dev_samples_per_cycle=16)
```

Returns `w_unc`, `w_avg`, `w_dev`, `T` and the error-control fields
`w_dev_certified`, `w_dev_bracket_lo/hi`, `w_dev_refinement`, `n_dev_used`,
`dev_converged`, `dev_cycles_max`, `dev_samples_per_cycle`, `dev_resolved` --
fourteen fields, identical in both languages (Python returns the
`UncertaintyRates` dataclass, MATLAB a struct).

Accuracy: `w_unc` and `w_avg` are exact to roundoff, the latter because the time
average is a closed-form divided difference rather than a quadrature, so
`n_quad` is accepted and unused. `w_dev` is a supremum recovered by sampling at
a density derived from the Bohr bandwidth, then polished; it carries two
rigorous certificates. Sampling under-estimates a supremum and a smaller `w_dev`
gives a larger margin, so the residual error is biased optimistic.

| Function | Role |
|----------|------|
| `uncertainty_rates(H_list, dH_list, dt, ...)` | Eq. 28 measures per unit `delta` |
| `time_bandwidth(rates, delta)` | `T*Omega_bnd` (Eq. 29) |
| `fidelity_bound(T_omega_bnd)` / `fidelity_bound_at(rates, delta)` | `F_lb` (Eq. 30) |
| `threshold_time_bandwidth(FT, nominal_error=0)` | Closed-form inverse of `F_lb` |
| `margin(rates, FT, nominal_error=0)` | Implied margin `M^K` (Python re-export: `kosut_margin`) |
| `t_omega_max()` / `T_OMEGA_MAX` | `2*sqrt(log(1+sqrt(2))) = 1.8776`; bound vacuous beyond this |

MATLAB lives in `matlab/+qrobustness/+kosut/`, Python in
`python/src/qrobustness/kosut.py` (re-exported at package level). Drivers:
`scripts/run_time_bandwidth_bound_comparison.py` and
`matlab/examples/run_time_bandwidth_bound_comparison.m`; both write the CSV
columns fixed by `CSV_HEADERS` (Python) and
`qrobustness.compat.kosut_csv_headers` (MATLAB). Compare with
`scripts/compare_time_bandwidth_bound.py` (`make compare-time-bandwidth-bound`).


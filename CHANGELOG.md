# Changelog

> SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>\
> SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>\
> SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>\
> SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>\
> SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>\
>
> SPDX-License-Identifier: AGPL-3.0-or-later

Notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Version numbers follow
[semantic versioning](https://semver.org/).

## [1.1.0]

Not released. Changes since 1.0.2.

### Added

- `multiparam_<FT>_angular.csv` stores `rang_*`, the one-step angular radius
  per direction. The xQRM figures read it. Paper macros include the quoted
  settings, the band-on audit percentages and the directional gain over the
  cross-polytope radius.
- Certificate layers `lengthspace`, `multiparam`, `timevarying`, `lindblad`,
  `synthesis`, `berberich` and `verify`, exported with the QRM API.
  `berberich_margin` sits beside `kosut_margin`. `states` and `openstates`
  are submodules only.
- `lindblad.rate_lipschitz`, `local_ops`, `local_dephasing_ops`,
  `adversarial_fidelity` and `toggling_frame_integral`.
- MATLAB/Octave peers, including the Kosut trajectory class and the state
  certificates (`states_parity.json`).
- `hamiltonian_part` and `hamiltonian_dnorm`: a Rump-verified bound. The
  diamond norm of `-1j[B, \cdot]` is `lambda_max(B) - lambda_min(B)`, with no
  SDP. `open_speed(..., exact_hamiltonian=True)` uses it.
- xQRM drivers for state examples, continuation checks and the bracket
  audit. `run_robust_vs_nominal` records per-structure upper brackets, pulse
  areas and angle budgets.
- `MarginResult.n_evals_minus` and `n_evals_plus`.
- `make install`, `test`, `run`, `verify` and `sync` replace `venv`,
  `paper-*`, `reproduce-*` and `check-*`. `run` does not copy artefacts into
  the paper. `clean`, `distclean` and `maintainer-clean` remain. `test`
  lints, then runs the unit suite, synthesis smoke and parity. `ENGINE=`
  selects one engine.
- `make verify-xQRM-synth`: the certificate harness on a freshly synthesised
  ensemble.
- A `Contents.m` in each MATLAB package. Tests check every documented `make`
  target, every document link, every relative link, and the SPDX header on
  every source file.

### Changed

- An adversarial witness is `m_adv` (`madv_*`). The certified upper end is
  `M_upper` (`Mupper_*`, `M_gamma_upper`). `TVBracket` fields are `m_adv`
  and `F_at_adv`. In `tv_bracket_<FT>.csv`, `n_adversary_calls` is
  `n_adversary_evals` (fidelity evaluations). Existing CSVs use the new
  headers.
- Cost is a fidelity-evaluation count. The bracket audit also records
  wall-clock time on a named machine. `verify` skips only the `t_*` columns.
- A reference threshold crossing is the lower end of the final bracket: an
  evaluated point that met the threshold. A ratio quoted against it is a
  lower bound.
- BLAS threads are pinned. Closed-system drivers use `margin_tol=1e-8`.
- MATLAB `dU_dmu_quad` is `dU_dmu_integral`.
- Docs are `docs/api.md`, `docs/layout.md` and `docs/verification.md`.
- `eval_tol = 0`: safe when `F >= F_T`. `eval_tol > 0`: safe only when
  `F > F_T + eval_tol`. Continuation and the bracket use that test.
- `TVBracket.n_evals` counts fidelity evaluations.
  `adversarial_upper_bound` takes `n_starts`, `starts` and `maxiter`. Kosut
  samples per cycle are the grid left after `n_dev_max`.
  `open_structure_constants` takes one `dt` per interval. The trajectory
  check includes the perturbed propagator in its tolerance.
  `state_angular_margin` uses `atan2`. Plot ranges widen only outside the
  QRM limits. `bench_margin_solvers.py` has `--out`.

### Fixed

- `iterative_margin` evaluates each point once, on both rays and all phases
  (both engines). Counts omit repeats.
- Continuation returns `stalled` when a step makes no progress along the
  ray (both engines). That includes `eval_tol > 0` with no safe point past
  the last. The search stops. It does not keep stepping up to `k_max`.
- `run_single_qubit_example` searches up to `1.05 max(M_upper, r_FS)`,
  above the constant crossing, and records `adv_violated`.
  `tab_singlequbit` prints `m_adv` only when a witness was found.
- With a vanishing gauge, each ray of `directional_margin` is certified to
  its own end of `omega` (both engines). Both rays had used the nearer end.
- For `eval_tol > 0`, every safe radius uses `F - eval_tol` (both engines).
  A radius at the raw `F` could cover an unsafe gap between two points
  evaluated safe. `eval_tol = 0` is unchanged. Bracket-audit band rows and the
  algorithm tests are recomputed.
- Verified diamond norms (both engines). Cholesky underflow follows Rump's
  Theorem 2.3: `n 3(2n + max a_ii) 2^-1074`. The divisor `1 - alpha` is
  rounded down. Sums are at most `fl(sum)(1 + 2 gamma_{m-1})`. Complex
  partial-trace entries are bounded by their real and imaginary parts. The
  Frobenius term of `hamiltonian_dnorm` is bounded upward. Non-finite or
  overflowing values fail. The Choi round trip is checked before every SDP.
  Certified values move in the last digits.
- The bracket audit uses the fourteen `multiparam` directions. The previous
  twenty per controller counted six coordinate axes twice.
- MATLAB `iterative_margin` promotes only a certified safe point. A
  nonmonotone fidelity could otherwise give an optimistic margin. The
  shipped ensemble is unaffected.
- `eval_tol` applies to continuation and to the bracket. A zero angular
  gauge leaves the dynamics unchanged. MATLAB `iterative_margin` reports
  `unresolved` and `n_unresolved`.
- The fidelity cross-check uses the eigendecomposition route. An empty
  `CheckReport` fails. Each bracket point is evaluated once.
- Gauges use `core.traceless`. `refine` checks `q` and the list lengths.
  MATLAB `kendall_tau_b` matches SciPy. The open-system Lipschitz constant
  in MATLAB is taken per controller.
- `run_cnot_case_study` and `run_scaling_example` honour `--out`.
  `run_slice_scan` honours `--FT`. The paper CSVs have Make recipes.
- MATLAB `log10_axis` rejects an unknown axis. `berberich.margin` rejects a
  bad `nominal_error`, and a systematic margin when the Magnus condition
  fails. Three production asserts remain under `python -O`.
  `lindblad.generator` and `plot_fidelity_error_sweeps` check sequence
  lengths. Quoted `tab:tvbracket` ranges match the printed table. `__all__`
  lists `traceless` once.
- `xqAnisotropyMax` is rounded as in `tab_multiparam`. `fig_validity` puts
  the `M^K` label above the curves; `fig_open` labels every x tick with
  data. `run_bracket_audit` makes three untimed warm-up calls before each timed
  measurement and times preprocessing over batches of ten calls.

### Removed

- `docs/theory.md`, `docs/time-bandwidth-bound.md` and
  `docs/margin-solvers-notes.md`.
- The `data/legacy/` comparison and `rebase-legacy-reference`.

## [1.0.2] - 2026-08-03

Aligns `structure_constant` with the paper, records which Algorithm 1 stop
fired, and adds the Table I rank statistic. Case-study margins are
bit-for-bit unchanged.

### Changed

- `structure_constant` (both engines) centres the perturbation on
  `\overline{\hat{H}}_\mu = \hat{H}_\mu - N^{-1}(\operatorname{Tr} \hat{H}_\mu) I`
  before the Frobenius norm, as `C_{\hat{H}}` specifies, and requires a
  square Hermitian matrix. The trace part is a global phase, and the
  trace-amplitude fidelity ignores it. Earlier margins stay valid.
  Non-traceless structures tighten, for example a single-level detuning.
  `H_0`, `H_1` and `H_2` are traceless, so published numbers are unchanged.
  Adds `qrobustness.traceless` (Python and MATLAB).
- Legends in `plot_margins_vs_sensitivity` (both engines) sit at the top
  left, clear of the data.

### Added

- `status_minus` and `status_plus` (`result.status_*` in MATLAB) name the
  Algorithm 1 stop: `eta_band`, `domain_truncated` or `iteration_limit`.
  They are set on the default path, with or without `margin_tol`.
  `domain_truncated` is the distance to the edge of `omega`, not a resolved
  margin. `converged_*` cannot separate the two, and remains.
  `safeguard_*` records the bisection safeguard. `reason_*` remains the
  optional `margin_tol` outcome.
- `focal_tests_<FT>.csv` and `qrobustness.compat.kendall_tau_b`
  (MATLAB/Octave) cross-check `M_j` against `|\zeta_j|`. Table I stays
  descriptive: Pearson `r` and Spearman `\rho`, with no inferential claim.
  The CSV adds Spearman `\rho` and Kendall `\tau_b`, each with a
  Holm-corrected two-sided `p`. Either statistic gives the same reading:
  appreciable for `H_0`, weak for `H_1`, negligible for `H_2`. MATLAB,
  Octave and SciPy share one closed-form `\tau_b` p-value, so the Python
  and MATLAB CSVs are byte-identical.

### Fixed

- `iterative_margin` (both engines) counts from 1, so `k_max` is the number
  of trial points per direction, as in Algorithm 1. The limit does not bind
  on the case study (default 10 000; every direction stops in the `\eta`
  band), so reported values are unchanged.
- The paper is a sibling repository. `PAPER_ROOT` points at that checkout.
  Publishing is `sync-paper-qrm` (alias `sync-paper`) and ships the Python
  tree.
  `verify_paper_consistency` takes the paper from `--paper-source`,
  `$QRM_PAPER_SOURCE` or the sibling checkout, and skips paper checks on a
  code-only clone.
- `pip` and `pytest` run as modules, so a moved checkout does not need the
  venv console scripts.
- `optimize_controller` runs under Octave via core `optimset` and `fminunc`
  (same quasi-Newton objective and gradient). No Octave Forge package, and
  no emulation of the MATLAB Optimization Toolbox. The `output` struct is
  read without a `message` field. The Octave suite passes 15/15.

## [1.0.1] - 2026-08-02

Corrects optional `margin_tol` refinement, Kosut's target-gate composition,
and the Table I correlations. The default Algorithm 1 path, the ensemble,
the margins and the figures are unchanged.

### Changed

- Table I (`correlations_<FT>.tex`) and the MATLAB `correlations_<FT>.csv`
  use `|\zeta_j|`. `M_j` is invariant if the parameter axis is reversed.
  Signed `\zeta_j` changes sign, and that sign hid the comparison. Fig. 3
  already plotted `|\zeta|`. Pearson `\epsilon_0` against `|\zeta_0|` rises
  from 0.44 to 0.69, and `M_0` against `|\zeta_0|` from -0.35 to -0.58.
  Control-structure correlations stay weak. `margins_table_<FT>.csv` still
  stores signed `\zeta_j` and is bit-for-bit unchanged, as are the margins.

### Fixed

- `kosut.margin` and `kosut.threshold_time_bandwidth` (both engines) absorb
  `eps_0` by
  `F_{\mathrm{eff}} = \cos(\arccos \mathcal{F}_T - \arccos \mathcal{F}_0)`,
  as `kosut.effective_threshold`. Theorem 1 bounds fidelity to the achieved
  gate. The certificate is against the target. `arccos` of the fidelity is
  the angle between the Choi states, so the angles add. The old sum
  `F_T + eps_0` was not conservative. `absorption='additive'` and
  `--absorption additive` reproduce pre-1.0.1 numbers. Implied `M^K`
  shrinks. `docs/time-bandwidth-bound.md` and the README give the new
  comparison. `nominal_error` must lie in `[0, 1]` (`1 - eps_0` is a
  fidelity). Values outside that interval are rejected, not clamped. Both
  engines test the closed form and the collinear single-qubit rotation
  that saturates it. The layer is supplementary, outside
  `make reproduce-QRM-margins`, and no paper claim depends on it.
- In `kosut.margin` (both engines) the positive root of
  `a*m^2 + b*m = y^2` is `2 y^2 / (b + sqrt(b^2 + 4 a y^2))`. This avoids
  catastrophic cancellation for `a*y^2 << b^2`, where the structure nearly
  commutes with the nominal evolution, and drops the `a <= 0` branch.
- The Kosut margin is the constant structured-parameter case, not a
  supremum-norm time-varying margin. A sign-modulated trajectory inside the
  same budget can defeat the coherent averaging behind `Omega_avg`.
- With `margin_tol`, a safe sample advances the certified lower end only
  when `(F - F_T)/L_{\hat{H}}` covers the gap, or continuation bridges it.
  The bracket is the first boundary of the nominal safe component. The old
  bisection could promote a later safe island and report the inflated
  margin as `bracketed`. If continuation falls behind, `partial` is a
  rigorous bracket wider than `margin_tol`. The default path is unchanged.
  Refined case-study margins are bit-for-bit identical. Single-crossing
  rays stay `bracketed`.

## [1.0.0] - 2026-07-30

Initial release.

### Core

- Gate propagator and normalised gate fidelity. Sensitivity bound and
  Lipschitz constant `L_{\hat{H}}`. Differential sensitivity `\zeta`.
  One-dimensional robustness margin (Algorithm 1, plus optional solvers).
  GRAPE synthesis.
- Python is the reference. MATLAB and Octave are peers under `docs/api.md`,
  with cross-engine comparison and golden fixtures in `data/reference/`.
- Full reproduction of the three-qubit case study.

### Segment derivative

- `\partial U^{(k)}/\partial\mu` is a divided difference in the eigenbasis of
  piecewise-constant `H^{(k)}`, exact to roundoff. One eigendecomposition
  per interval makes `U^{(k)}` and the derivative exactly consistent.
  `propagator()` uses `expm`, so the two fidelity routes can differ at
  about `1e-15`.
- `method='quadrature'` (Gauss-Legendre) cross-checks the closed form.
  `n_quad` applies only there. `method='exact'` accepts `n_quad` and leaves
  it unused.

### Error control

As released, `docs/theory.md` classed each quantity as exact to roundoff or
certified. Two were approximated.

- **`M`.** Distance to an evaluated point with `F >= F_T`: a lower bound,
  conservative, never optimistic. `eta` is a fidelity band, not a margin
  band. Uncertainty in `\mu` is of order `\eta/|\zeta|` and grows without
  bound as `\zeta \to 0` (the flat, highly robust case). On the case study,
  `eta=1e-6` leaves about `5e-4` relative error in `M`. `margin_tol`
  refines until `(M_{\mathrm{upper}} - M)/M <= margin_tol`, about `1e-10`
  after 90 further fidelity evaluations. `MarginResult` carries `M_upper*`,
  `margin_uncertainty`, `reason_*` and `certificate` (`segment` for
  `algorithm1` and `lipschitz_*`; `endpoint` for `doubling` and
  `newton_probe`, which probe beyond the Lipschitz radius). Paper drivers
  use the default, so published tables are the `eta` values.
- **Kosut `w_dev`.** `\Omega_{\mathrm{avg}}^{\mathrm{dev}}` is a supremum
  over `t`. Sampling can only fall short, and a smaller `w_dev` makes the
  implied margin larger, so the residual is optimistic. The grid follows
  the Bohr bandwidth of the interaction-picture operator. Brent polishes
  candidate maxima. Sweeps stop at `dev_tol`. Certificates: a Lipschitz
  bound from `d\tilde{H}/ds = i[H, \tilde{H}]`, and an isospectral bracket
  `\|\tilde{H}(t)\| = \|\hat{H}^{(k)}\|`. The average
  `\langle\tilde{H}\rangle` uses the same divided difference, so `w_avg`
  is exact. `uncertainty_rates` accepts `n_quad` and leaves it unused.

### Conventions

- Result trees are named for the method (`results/lipschitz-margin-*`,
  `results/time-bandwidth-bound-*`). The only paper-aware part of the build
  is the `PAPER_*` block in the `Makefile`, with the `sync-paper-*` and
  `verify-paper-*` targets.
- `load_problem` returns `n_qubits` and `dim` (`2**n_qubits`) separately.
  `lipschitz_constant` expects the Hilbert-space dimension.
- Sources and docs are ASCII, with LaTeX-like maths. British spelling in
  prose. Identifiers stay as written. All three engines write CSV with LF.
- `data/controllers/` is a frozen ensemble from earlier optimisation runs.
  `optimize_controller` repeats the workflow, not that run. See `README.md`.

### Supplementary

- Kosut-Lidar-Rabitz time-bandwidth bound
  ([arXiv:2507.01215](https://arxiv.org/abs/2507.01215)), for a closed
  system with a purely coherent scalar perturbation:
  `python/src/qrobustness/kosut.py`, `matlab/+qrobustness/+kosut/`,
  `run_time_bandwidth_bound_comparison`,
  `scripts/compare_time_bandwidth_bound.py` and the
  `time-bandwidth-bound*` targets. Experimental, and outside
  `make reproduce-QRM-margins`. No paper claim depends on it. Agreement of
  the three engines is a consistency check, not an accuracy check.
  `docs/time-bandwidth-bound.md` records the specialisation, the caveats
  and the numerical accuracy.

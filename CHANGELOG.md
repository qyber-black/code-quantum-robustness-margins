# Changelog

> SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>\
> SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>\
> SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>\
> SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>\
> SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>\
>
> SPDX-License-Identifier: AGPL-3.0-or-later

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [1.0.2] - 2026-08-03

Patch release aligning the toolbox with the paper's structure constant and
making the Algorithm 1 stopping rule observable, plus the rank statistic used
for Table I. The case-study margins are bit-for-bit unchanged.

### Changed

- `structure_constant` (both engines) now centres the perturbation structure to
  its traceless part, \(\overline{\hat{H}}_\mu = \hat{H}_\mu - N^{-1}(\operatorname{Tr}
  \hat{H}_\mu) I\), before taking the Frobenius norm, as the paper's
  \(C_{\hat{H}}\) specifies. The trace part contributes only a global phase to
  the propagator, which the trace-amplitude fidelity ignores, so this is a
  *tightening*: previously reported margins remain valid but were unnecessarily
  conservative for non-traceless user structures (e.g. a single-level detuning).
  The case-study structures \(H_0, H_1, H_2\) are traceless, so every published
  number is unchanged. The structure is now also validated as square and
  Hermitian. New `qrobustness.traceless` / `qrobustness.traceless` (MATLAB).
- `plot_margins_vs_sensitivity` (both engines): legends moved to the top left,
  where they no longer sit over the data.

### Added

- `MarginResult.status_minus` / `status_plus` (MATLAB: `result.status_*`) report
  *which* Algorithm 1 stopping rule fired -- `eta_band`, `domain_truncated` or
  `iteration_limit` -- and are populated on the default path, independently of
  `margin_tol`. A domain-truncated result certifies only that the margin is at
  least the distance to the edge of `omega` and must not be read as a resolved
  margin; the pre-existing `converged_*` flag cannot distinguish the two and is
  retained for backward compatibility. `safeguard_*` records whether the
  bisection safeguard fired. `reason_*` keeps its distinct meaning: the outcome
  of the optional `margin_tol` bracket refinement.
- `focal_tests_<FT>.csv` and `qrobustness.compat.kendall_tau_b` (MATLAB /
  Octave): a rank-statistic cross-check for the three margin-versus-sensitivity
  comparisons \(M_j\) versus \(|\zeta_j|\). Table I stays descriptive --
  Pearson \(r\) and Spearman \(\rho\), unchanged -- and the paper makes no
  inferential claim; the CSV records Spearman \(\rho\) and Kendall
  \(\tau_b\) with Holm-corrected two-sided \(p\) for each, confirming that
  the descriptive reading (appreciable for \(H_0\), weak for \(H_1\),
  negligible for \(H_2\)) does not depend on the choice of rank statistic.
  Both engines share a closed-form asymptotic \(\tau_b\) p-value, so MATLAB,
  Octave and the SciPy reference agree exactly and the Python and MATLAB CSVs
  are byte-identical.

### Fixed

- `iterative_margin` (both engines): the step counter now starts at 1 rather
  than 0, so `k_max` is exactly the number of evaluated trial points per
  direction instead of one fewer than the number permitted. The paper's
  Algorithm 1 and both engines now share this convention. Only reachable when
  the limit actually binds, which no case-study run does (the default is
  10 000 and every direction terminates in the \(\eta\) band), so no reported
  value changes.
- The paper now lives in a sibling repository rather than containing this one,
  so `PAPER_ROOT` no longer points at the parent directory. The publishing
  target is named after the paper (`sync-paper-qrm`, aliased `sync-paper`)
  rather than the engine, and publishes the Python tree. `verify_paper_consistency`
  resolves the paper via `--paper-source` / `$QRM_PAPER_SOURCE` / the sibling
  checkout and skips its paper checks on a code-only clone instead of failing.
- `pip` and `pytest` are invoked as modules, so a moved checkout no longer
  breaks the venv console scripts.
- `optimize_controller` now runs under Octave. It called `optimoptions`, which
  is MATLAB-only, so GRAPE synthesis and `test_optimize_controller` failed on
  the Octave peer with a misleading "install the optim package" message. Octave
  ships `fminunc` and `optimset` in core, so the option struct is now built with
  `optimset` there -- same quasi-Newton objective-and-gradient path, no Octave
  Forge package and no Optimization Toolbox emulation required. The `output`
  struct is also read defensively, since Octave supplies no `message` field.
  The full Octave suite (15/15) now passes.

## [1.0.1] - 2026-08-02

Patch release correcting the optional `margin_tol` refinement, the target-gate
composition in the supplementary Kosut layer, and the correlation analysis used
for Table I. The default Algorithm 1 path, the controller ensemble, the
robustness-margin values and the figures are unchanged.

### Changed

- `correlations_<FT>.tex` (Table I of the paper) and the MATLAB
  `correlations_<FT>.csv` now correlate against the sensitivity *magnitudes*
  \(|\zeta_j|\) rather than the signed \(\zeta_j\). The margin
  \(M_j = \min\{M_{j,-}, M_{j,+}\}\) is invariant under reversal of the
  parameter coordinate, while \(\zeta_j\) changes sign, so \(|\zeta_j|\) is the
  orientation-invariant local comparator; mixing signs was masking the
  relationship, and
  `robustness_margins_sensitivity.png` (Fig. 3) already plotted \(|\zeta|\).
  The drift-structure entries change materially -- \(\varepsilon_0\) against
  \(|\zeta_0|\) rises from 0.44 to 0.69 (Pearson) and \(M_0\) against
  \(|\zeta_0|\) from -0.35 to -0.58 -- while the control-structure relations
  remain weak. `margins_table_<FT>.csv` still records the signed \(\zeta_j\)
  and is bit-for-bit unchanged, as are the margins themselves.

### Fixed

- `kosut.margin` / `kosut.threshold_time_bandwidth` (both engines): the nominal
  fidelity deficit `eps_0` is now absorbed into the threshold through the
  *angular* relation
  \(F_{\mathrm{eff}} = \cos(\arccos \mathcal{F}_T - \arccos \mathcal{F}_0)\),
  exposed as the new `kosut.effective_threshold`. Their Theorem 1 bounds the
  fidelity to the *achieved* nominal gate, whereas the certificate is stated
  against the *target*; since `arccos` of the gate fidelity is the angle
  between the corresponding Choi states, the angles -- not the fidelity
  deficits -- add. The previous additive form `F_T + eps_0` is looser than the
  sufficient condition and so was **not** conservative; it remains selectable
  as `absorption='additive'` (and `--absorption additive` in the drivers) to
  reproduce pre-1.0.1 numbers. The implied margins `M^K` shrink accordingly,
  and `docs/time-bandwidth-bound.md` and the README report the recomputed
  comparison. `effective_threshold` also validates `0 <= nominal_error <= 1`
  (`1 - eps_0` is a fidelity) rather than clamping impossible inputs, and both
  engines regression-test the closed form together with the collinear
  single-qubit rotation that saturates it. The Kosut layer is supplementary and
  experimental, outside `make check-margins`, and no paper claim depends on it.
- `kosut.margin` (both engines): the positive root of `a m^2 + b m = y^2` is
  now evaluated in the rationalised form `2 y^2 / (b + sqrt(b^2 + 4 a y^2))`,
  which avoids catastrophic cancellation when `a y^2 << b^2` (structures nearly
  commuting with the nominal evolution) and removes the `a <= 0` special case.
- `kosut` documentation now states the scope explicitly: the margin is the
  *constant structured-parameter* specialisation and is **not** a
  supremum-norm time-varying margin -- a sign-modulated trajectory within the
  same budget can defeat the coherent averaging behind `Omega_avg`.
- `iterative_margin(margin_tol=...)`: the optional safe/unsafe bracket
  refinement now applies a certified-promotion rule -- a pointwise-safe
  sample advances the certified lower endpoint only when the gap from the
  current certified end is covered by that sample's safe radius
  \((F - F_T)/L_{\hat{H}}\) or bridged by safe-radius continuation.
  Previously, with a nonmonotone fidelity, the plain bisection could promote
  a sample from a disconnected safe island beyond the first threshold
  crossing and report an inflated margin as `'bracketed'`. The bracket now
  always encloses the *first* boundary of the nominal safe component; when
  continuation cannot keep pace with bisection the new reason `'partial'`
  reports a rigorous bracket whose width exceeds `margin_tol`. The default
  Algorithm 1 path (no `margin_tol`) is unchanged, and the refined case-study
  margins are bit-for-bit identical (single-crossing rays remain
  `'bracketed'`); only the general nonmonotone claim needed the fix.

## [1.0.0] - 2026-07-30

Initial release.

### Core

- Gate propagator and normalised gate fidelity; sensitivity bound and Lipschitz
  constant \(L_{\hat{H}}\); differential sensitivity \(\zeta\); iterative
  one-dimensional robustness margin (Algorithm 1, plus optional solvers);
  GRAPE controller synthesis.
- Python is the reference implementation; MATLAB and Octave are peers held to it
  by a shared API contract (`docs/api.md`), cross-engine comparison and golden
  fixtures in `data/reference/`.
- Full reproduction of the three-qubit case study of the accompanying paper.

### Segment derivative

- \(\partial U^{(k)}/\partial\mu\) is evaluated in exact closed form in the
  eigenbasis of \(H^{(k)}\). The controls are piecewise constant, so
  \(H^{(k)}\) is constant on each interval and the defining integral is a
  divided difference, exact to roundoff.
- Gauss-Legendre quadrature is selectable as `method='quadrature'` and serves as
  a cross-check on the closed form in the test suite. `n_quad` applies only to
  that path; under the default `method='exact'` it is accepted and unused.
- On the exact path a single eigendecomposition per interval serves both the
  propagator and the derivatives, so \(U^{(k)}\) and \(\partial
  U^{(k)}/\partial\mu\) are exactly consistent. `propagator()` uses `expm`, so
  fidelities computed the two ways may differ at ~1e-15.

### Error control

`docs/theory.md` classifies every reported quantity as exact to roundoff or
approximated with an explicit certificate. Two are approximated:

- **Margin \(M\).** \(M\) is the distance to a point at which
  \(F \ge F_T\) was evaluated, hence a lower bound on the true margin:
  conservative, never optimistic. `eta` is a fidelity band rather than a margin
  band, and the induced uncertainty in \(\mu\) is \(\sim\eta/|\zeta|\),
  which grows without bound as \(\zeta \to 0\), that is for the flat, highly
  robust controllers of interest; on the case study the default `eta=1e-6`
  leaves about 5e-4 relative error in \(M\). `margin_tol` refines the
  safe/unsafe bracket until \((M_{\mathrm{upper}} - M)/M \le\) `margin_tol`
  and tightens \(M\), reaching 1e-10 in about 90 extra fidelity evaluations.
  `MarginResult` carries `M_upper*`, `margin_uncertainty`, `reason_*` and
  `certificate` (`'segment'` for `algorithm1` and `lipschitz_*`, `'endpoint'`
  for `doubling` and `newton_probe`, which probe beyond the Lipschitz radius).
  The paper drivers use the default, so the published tables are the `eta`-based
  values.
- **Kosut `w_dev`.** \(\Omega_{\mathrm{avg}}^{\mathrm{dev}}\) is a supremum
  over \(t\). Sampling can only under-estimate a supremum, and a smaller
  `w_dev` yields a larger implied margin, so the residual error is biased in the
  optimistic direction. The sampling density is derived from the exactly-known
  Bohr bandwidth of the interaction-picture operator, candidate maxima are
  polished with Brent, and the grid is refined until successive sweeps agree to
  `dev_tol`. Two rigorous certificates are returned: a Lipschitz bound from
  \(d\tilde{H}/ds = i[H,\tilde{H}]\) and an isospectral bracket from
  \(\|\tilde{H}(t)\| = \|\hat{H}^{(k)}\|\).

The Kosut time average \(\langle\tilde{H}\rangle\) is closed form, using the
same divided-difference identity as \(\partial U/\partial\mu\), so `w_avg` is
exact and `uncertainty_rates` accepts `n_quad` without using it.

### Conventions

- Result trees are named after the method (`results/lipschitz-margin-*`,
  `results/time-bandwidth-bound-*`). The only paper-aware part of the build is
  the `PAPER_*` block at the top of the `Makefile` and the `sync-paper-*` and
  `verify-paper-*` targets that read it.
- `load_problem` returns `n_qubits` and `dim` (= `2**n_qubits`) as separate
  fields; `lipschitz_constant` expects the Hilbert space dimension.
- Sources and documentation are ASCII, with mathematical symbols in LaTeX-like
  notation. British spelling in prose; identifiers as written. CSV writers emit
  LF in all three engines.
- The controller ensemble in `data/controllers/` comes from earlier optimisation
  runs and is shipped frozen; `optimize_controller` reproduces the synthesis
  workflow but not that specific run. See the provenance note in `README.md`.

### Supplementary (experimental)

- A comparison with the Kosut-Lidar-Rabitz time-bandwidth bound
  ([arXiv:2507.01215](https://arxiv.org/abs/2507.01215)), specialised to the
  closed-system, purely coherent, scalar structured perturbation model of this
  toolbox: `python/src/qrobustness/kosut.py`,
  `matlab/+qrobustness/+kosut/`, the `run_time_bandwidth_bound_comparison`
  drivers, `scripts/compare_time_bandwidth_bound.py` and the
  `time-bandwidth-bound*` make targets.
- It is experimental and sits outside the reproduction gate
  `make check-margins`; no claim in the paper depends on it. Agreement between
  the three engines is a consistency check rather than an accuracy check.
  `docs/time-bandwidth-bound.md` documents the specialisation, the caveats and
  the numerical accuracy.

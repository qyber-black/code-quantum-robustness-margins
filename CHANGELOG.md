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

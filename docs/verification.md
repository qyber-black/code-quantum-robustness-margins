# Tests and verification

> SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>\
> SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>\
> SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>\
> SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>\
> SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>\
>
> SPDX-License-Identifier: AGPL-3.0-or-later

## `make test`

Runs every stage for every engine. `ENGINE=...` restricts the run to one
engine. Each stage runs even if an earlier stage failed. Each engine ends
with a tally.

| stage | what it runs |
| :--- | :--- |
| `test-lint` | `lint`: ruff on `python/` and `scripts/`, miss_hit on `matlab/` |
| `test-unit` | `python/tests` (pytest) or `matlab/tests/run_all_tests` (MATLAB, Octave) |
| `test-synth` | synthesis smoke on a few unseen controllers, written to `build/` |
| `test-parity` | for MATLAB and Octave: the engine's committed result tables against Python's (no-op for Python) |

The MATLAB/Octave state and open-state tests read the parity fixture
`matlab/tests/fixtures/states_parity.json`, written by
`scripts/gen_states_parity.py`; `python/tests/test_states_parity.py` checks
that the committed fixture is current.

## `make verify`

| part | what it checks |
| :--- | :--- |
| `verify-PAPER-reproduce` | recomputes every result of the paper into a separate tree and compares it numerically with the committed tree; regenerates the paper artefacts and compares them with the paper's copies; every generated macro and table must be cited by the paper and every cited one generated |
| `verify-QRM-consistency` | every number of the QRM manuscript against the results tree (`verify_paper.md`) |
| `verify-xQRM-theorems` | the certified inequalities over the shipped ensemble (`scripts/run_theorem_verification.py`) |
| `verify-xQRM-synth` | the same checks on a freshly synthesised ensemble |

## Theorem checks (`qrobustness.verify`)

Each check evaluates a certified inequality as a slack. The slack is the
certified quantity minus its bound, and it is non-negative when the
inequality holds. The check fails only when the slack is negative by more
than a tolerance. The tolerance comes from the computation. Fidelities are
evaluated by two propagator routes, a per-interval eigendecomposition and
`expm`, and their discrepancy plus the unitarity defect set the tolerance.
Where a certificate stops on a band (eta), that band is the tolerance.
Probes sit where a certificate is most likely to fail: on the certified
boundary, on sign-modulated and sparse trajectories, under sub-interval
refinement, and under multi-start adversaries.

| check | inequality |
| :--- | :--- |
| `check_metric_triangle` | arccos F is a metric (triangle inequality) |
| `check_absorption` | angular nominal-error absorption implies F >= F_T on the target |
| `check_constant_margin` | F(mu) >= F_T on a dense grid of [-M, M] |
| `check_polytope` | F >= F_T in the certified cross-polytope |
| `check_lipschitz_pairs` | trajectory Lipschitz bound on pairs inside the safe set |
| `check_tv_slope` | trajectory slope bound along homotopies |
| `check_fs_angle` | theta(U_S, U(delta)) <= m s on refined trajectories |
| `check_trajectory_certificate` | adversaries at r_0, r_FS, M^{K,tri}_tv cannot break F >= F_T |

Each returns a `CheckReport` (number of probes, minimum slack, tolerance,
the probe that attained the minimum), so a failure can be reproduced in
isolation. A failure is a theorem violation, an implementation error or an
under-estimated tolerance; reproduce the arg-min probe before changing a
tolerance.

## MATLAB and Octave coverage

| module | peer | test |
| :--- | :--- | :--- |
| `core` (incl. `iterative_margin`), `optimize`, `plotting` | `+qrobustness` | `test_*` core tests, `test_evaluation_band.m`; parity on the QRM tables |
| `kosut` | `+qrobustness/+kosut` | `test_kosut_bound.m`; parity on the Kosut tables |
| `lengthspace` | `+qrobustness/+lengthspace` | `test_lengthspace.m` |
| `multiparam` | `+qrobustness/+multiparam` | `test_multiparam.m` |
| `timevarying` (certificates) | `+qrobustness/+timevarying` | `test_timevarying.m` |
| `lindblad` | `+qrobustness/+lindblad` | `test_lindblad.m` |
| `berberich` | `+qrobustness/+berberich` | no MATLAB/Octave test yet |
| `states`, `openstates` | `+qrobustness/+states`, `+openstates` | `test_states.m`, `test_openstates.m` (parity fixture) |
| `synthesis`, `verify`, adversarial search | none | Python only |

`lindblad.diamond_norm` in MATLAB/Octave is the solver-free upper bound
(Python `diamond_norm_free`); Python's SDP `diamond_norm` has no peer.

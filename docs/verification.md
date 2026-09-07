# Numerical verification of the theorems

> SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>\
> SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>\
> SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>\
> SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>\
> SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>\
>
> SPDX-License-Identifier: AGPL-3.0-or-later

Every certified inequality shipped with the package can be re-verified
numerically. `python/src/qrobustness/verify.py` provides the harness.
`python/tests/test_theorems.py` runs fast seeded instances of every
check, one test per theorem or lemma of the paper.
`scripts/run_theorem_verification.py` sweeps the shipped three-qubit
ensemble and exits nonzero on any violation.

## Principles

1. **Slack, not equality.** Each check evaluates a certified inequality
   as a slack (certified quantity minus its bound, nonnegative when
   satisfied) and fails only when the slack is negative beyond a
   numerical allowance.
2. **Derived tolerances.** The allowance is derived from the
   computation, not assumed: fidelities are evaluated through two
   independent propagator routes (per-interval eigendecomposition and
   `scipy.linalg.expm`), and the observed cross-route discrepancy plus
   the unitarity defect of the propagator product set the tolerance.
   Where the certificate itself terminates on a band (the fidelity band
   `eta` of Algorithm 1), that band is the allowance.
3. **Adversarial probes.** Certificates are probed where they are most
   likely to fail: at the certified boundary, with sign-modulated and
   sparse trajectories, with sub-interval refinement (the failure mode
   that disproved the sup-norm reading of the constant-class
   time-bandwidth margin; see docs/time-bandwidth-bound.md), and with
   multi-start exact-gradient adversaries -- not only with uniform
   random samples.

## Checks

| Check | Statement verified |
|-------|--------------------|
| `check_metric_triangle` | The gate-fidelity angle is a metric (triangle inequality on random unitary triples) |
| `check_absorption` | Angular nominal-error absorption: achieved-gate fidelity at the angular threshold implies target fidelity at `F_T` |
| `check_constant_margin` | Algorithm 1 margin: `F(mu) >= F_T` on a dense grid of `[-M, M]` |
| `check_polytope` | Safe-polytope theorem: `F >= F_T` at random boundary and interior points of the certified cross-polytope |
| `check_lipschitz_pairs` | Trajectory Lipschitz lemma on random trajectory pairs inside the safe set |
| `check_fs_angle` | Fubini-Study certificate, inner inequality: `theta(U_S, U(delta)) <= m * s` on refined random trajectories |
| `check_trajectory_certificate` | Uniform trajectory margins (`r_0`, `r_FS`, `M^K_tv`): adversaries at the certified budget cannot break `F >= F_T` |

Each check returns a `CheckReport` with the number of probes, the
minimum observed slack, the tolerance in force, and the probe that
attained the minimum, so a failure is immediately reproducible.

## Interpreting a failure

A failing check means one of three things. It may be a genuine theorem
violation, as with the additive absorption and the constant-class
time-bandwidth margin, both found by exactly this kind of probing. It
may be an implementation bug, as with the cancellation in the
quadratic-root inversion, caught by the single-qubit commuting case.
Otherwise the tolerance is under-estimated. The
`argmin` field pinpoints the probe; reproduce it in isolation before
changing any tolerance.

## MATLAB and Octave parity

The paper-1 core (`core`, `kosut`, `optimize`, `plotting`) has full
three-engine parity, held to Python by `make test-parity`, which compares
the committed result tables of the two engines directly. The certificate layers have peers as follows:

| module | MATLAB/Octave peer | notes |
| :--- | :--- | :--- |
| `lengthspace` | `+qrobustness/+lengthspace` | `test_lengthspace.m` |
| `lindblad` | `+qrobustness/+lindblad` | superoperators, the diamond norm, `channel`, `open_margin`; `test_lindblad.m` |
| `multiparam` | `+qrobustness/+multiparam` | polytope, both gauges, directional margin; `test_multiparam.m` |
| `timevarying` | `+qrobustness/+timevarying` | `uniform_margin`, `fs_margin`, `fs_margin_joint`; `test_timevarying.m` |
| `berberich` | `+qrobustness/+berberich` | `margin`, both uncertainty classes |
| `verify` | -- | Python only |
| `synthesis` | -- | Python only |

Python-only, and why: `verify` is the numerical theorem harness, a testing
tool rather than toolbox API; `synthesis` is ensemble-robust GRAPE, which
needs an optimiser whose MATLAB and Octave behaviour differ (`fminunc`
against the core Octave one) enough that a peer would not reproduce
Python's controllers bit for bit. Neither blocks a user computing margins.
The adversarial search behind the upper witnesses is Python-only for the
same reason: it is evidence, not a certificate.

`iterative_margin` takes a `safe_radius_fn` hook in both engines, which is
what lets the multi-parameter layer step with the Choi-angular rule. Its
default is the Lipschitz surplus rule.

`PathGauge` is a struct plus functions rather than a `classdef`: the `.m`
sources are meant to run unchanged under both engines and Octave's
`classdef` support is not complete enough to rely on.

### The diamond norm needs no SDP solver

The Watrous program is a **minimisation**: every feasible point is already
an upper bound, and an upper bound is exactly what a robustness constant
needs. A solver is only required to make the bound tight. Neither CVX
(MATLAB) nor SDPT3 (Octave) is therefore required, and neither ports
cleanly.

`qrobustness.lindblad.diamond_norm` (MATLAB/Octave) and
`lindblad.diamond_norm_free` (Python) therefore start from a closed-form
feasible point built from the polar factors of the Choi matrix, then
tighten it by subgradient steps, restoring feasibility by projection
onto the positive semidefinite cone. A final shift repairs the iterate
to exact feasibility, so the answer is an upper bound whatever the
iteration did.

Accuracy, measured rather than assumed:

- exact on the dephasing and amplitude-damping families of the case
  studies, where the analytic value is `2n` -- and in fact closer to it
  than the cvxpy path, which returns `4 + 6e-8` where this returns
  `4 + 1.5e-10`;
- agreeing with cvxpy to about 1e-4 relative on random dissipative and
  mixed generators;
- up to about 3% conservative for a pure Hamiltonian superoperator,
  where the subgradient stalls on a degenerate spectrum. Conservative is
  the safe direction. `python/tests/test_diamond_free.py` asserts this
  bound so the limitation cannot regress silently.

So `cvxpy` is off the critical path: the open-system layer runs with numpy
and scipy alone, and the `[open]` extra is needed only to cross-check
against the SDP.

`make test ENGINE=octave` runs the unit suite under Octave, not only the
case-study drivers.

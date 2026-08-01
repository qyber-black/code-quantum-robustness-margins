# What this package computes

> SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>\
> SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>\
> SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>\
> SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>\
> SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>\
>
> SPDX-License-Identifier: AGPL-3.0-or-later

Every quantity the package reports, what it means, which result of the paper
*Fidelity-Based Robustness Margins for Finite-Time Quantum Control* it comes
from, and how accurately it is computed. Paper cross-references are given as
`(eq:...)`, `Lemma ...`, `Theorem ...`, matching the labels in `main.tex`.

Companion documents: [api.md](api.md) for the call signatures,
[margin-solvers-notes.md](margin-solvers-notes.md) for the solver variants.

---

## 1. Model

Units with `hbar = 1`. Controls are **piecewise constant** on `tau` intervals of
length `Delta = t_f/tau` -- this assumption is what makes most of the package
exact rather than approximate, and it is used throughout.

Nominal dynamics (`eq:schrodinger_gate`), with the ordered product

```
U(t_f) = U^(tau) ... U^(1),        U^(k) = exp(-i H^(k) Delta)
H^(k)  = H_0 + u_1(k) H_1 + u_2(k) H_2
```

Structured scalar perturbation (`eq:perturbed_hamiltonian`): a **single real
parameter** `mu` multiplies a known structure matrix. In the case study `mu`
enters as a relative amplitude error on one of `H_0, H_1, H_2`:

```
Htil^(k)(mu) = H^(k) + (mu - mu_0) Hhat^(k)
```

with `mu_0 = 0` nominal. `Hhat^(k)` is what `dH_structure` returns:

| structure | `Hhat^(k)` | interpretation |
|---|---|---|
| `H0` | `H_0` | drift amplitude error |
| `H1` | `u_1(k) H_1` | *relative* error on control 1 |
| `H2` | `u_2(k) H_2` | *relative* error on control 2 |

Note the control cases carry the control amplitude, so `Hhat` is
`k`-dependent. This is why `Omega_unc` is `max_k` rather than a single norm.

**Gate fidelity** -- phase-invariant, normalised:

```
F_mu = |Tr(U_f^dag Util(mu))| / N          (`gate_fidelity`, N = dim)
```

`eps_0 = 1 - F_{mu_0}` is the nominal error. `F_T` is the threshold
(`0.999` in the case study).

---

## 2. Differential sensitivity `zeta`

`zeta_mu0 = dF_mu/dmu` at `mu_0` (`eq:sens`), computed via the
product-derivative form (`eq:product_derivative`): the derivative of the
ordered product is a sum over intervals, each term inserting
`dU^(k)/dmu` into the product at position `k`.

```
zeta = (1/N) sum_k Re Tr[ U_f^dag D^(k) e^{-i phi} ]
D^(k) = U^(tau) ... dU^(k)/dmu ... U^(1)
```

with the global phase `e^{i phi}` from `eq:phase`.

The per-interval derivative is (`eq:Xb`)

```
dU^(k)/dmu = -i Delta int_0^1 e^{-i Delta H^(k)(1-s)} Hhat^(k) e^{-i Delta H^(k) s} ds
```

This integral is not evaluated numerically. Because the control is
piecewise constant, `H^(k)` is *constant* on the interval, and in its
eigenbasis the integral is a divided difference. With `H^(k) = V diag(lam) V^dag`
and `X_mn = Delta (lam_n - lam_m)/2`:

```
dU^(k)/dmu = -i Delta * V [ (V^dag Hhat^(k) V) .* Phi ] V^dag
Phi_mn     = exp(-i Delta (lam_m + lam_n)/2) * sin(X_mn)/X_mn
```

The identity used is `(e^a - 1)/a = e^{a/2} sin(X)/X` with
`a = -i Delta (lam_n - lam_m)`. Since `a` is purely imaginary there is no
cancellation and no magnitude threshold to tune; only the literal `X == 0`
entries (the diagonal and exact degeneracies) are masked. See
`core._dU_dmu_exact` / `qrobustness.dU_dmu_exact`.

Sign convention: `zeta` is signed, as in `eq:sens`. Table I of the paper
correlates the signed values.

---

## 3. Lipschitz constant and the sensitivity bound

`Theorem thm:sens_bound` bounds the sensitivity by a sum of Frobenius norms of
the per-interval derivatives. Bounding those (`Lemma lemma:dU_bound`) and
maximising over the region where `F_mu > F_T` gives the Lipschitz constant of
`Lemma lemma:lipschitz`:

```
L_Hhat = B_T * C_Hhat,        B_T = sqrt((1 - F_T^2)/N)
```

* `B_T` -- `lipschitz_constant(FT, N, C_H)`. `N` is the Hilbert space
  dimension (`dim`), not the qubit count; `load_problem` returns both
  `dim` and `n_qubits` precisely to avoid that confusion.
* `C_Hhat` -- `structure_constant(kind, Hhat, dt, tau, controls)`:

  | `kind` | `C_Hhat` | used for |
  |---|---|---|
  | `drift` | `tau * Delta * ||Hhat||_F` | `H0` |
  | `control` | `Delta * ||f||_1 * ||H_m||_F` | `H1`, `H2` (`f` = that control) |

Both are closed-form and exact given their inputs. Note that exactness of
the *formula* is not the same as `L_Hhat` being a valid Lipschitz constant for
the true `F_mu` -- that is the content of the paper's theorems, not something
the code re-verifies at runtime.

`Lemma lemma:lipschitz` states `|F_{mu_b} - F_{mu_a}| <= L_Hhat |mu_b - mu_a|`
for `mu_a, mu_b` in `I`, the connected component of `{mu : F_mu > F_T}`
containing `mu_0`. Only the *lower* half is used downstream: the margin search
must exclude the fidelity *falling* below `F_T`.

---

## 4. Robustness margin

`Theorem thm:safe_radius` (safe radius): from a safe point `nu` with
`F_nu > F_T`,

```
r_nu = (F_nu - F_T) / L_Hhat
```

and every `mu_1` with `|mu_1 - nu| <= r_nu` satisfies `F_{mu_1} >= F_T`.
Algorithm 1 (`algorithm` in the paper) iterates this: step by the current safe
radius, re-evaluate, repeat; on overshoot, bisect back to a safe point. Each
of the two directions is searched independently, and

```
M = min(M_minus, M_plus)
```

`iterative_margin` implements this as `method='algorithm1'` (the default and
the one used for the paper).

### What the returned margin certifies

* `M` is always the distance to a point at which `F >= FT` was actually
  evaluated. It is therefore a *lower* bound on the true margin: the reported
  margin is conservative, never optimistic.
* `certificate='segment'` (`algorithm1`, `lipschitz_*`) means every point
  between `mu_0` and the endpoint is covered by a safe-radius certificate.
  `certificate='endpoint'` (`doubling`, `newton_probe`) means only the
  endpoint was verified -- those methods deliberately probe beyond the Lipschitz
  radius, so a dip below `F_T` in between is not excluded. These methods are
  intended for exploration rather than certification.
* `converged_minus/plus` records only that *a stopping rule fired*. It does not
  by itself certify accuracy -- see below.

Note on terminology: *interval* denotes a constant-control time slice of length
`Delta`, and is what the `segment_*` helpers operate on. The certificate class
`'segment'` refers instead to a segment of the perturbation ray in `mu`. The two
lie on different axes.

### Accuracy of `M`

`eta` (default `1e-6`) is a fidelity band: the search stops when
`0 <= F - F_T < eta`. The induced uncertainty in `mu` is roughly `eta/|zeta|`,
which grows without bound as `zeta -> 0`, that is, for precisely the flat,
highly robust controllers of interest. On the case study the
default `eta` leaves about 5e-4 relative error in `M`, so
`margins_table_0.999.csv` carries about 3-4 significant figures of margin, not
the 6 it prints. The error is one-sided (`M` under-estimates).

Pass `margin_tol` to convert that into a *margin* statement. The safe/unsafe
bracket is refined until

```
(M_upper - M) / M <= margin_tol
```

so the true margin lies in `[M, M_upper]`, and `M` itself is tightened. The
bracket refers to the *first* boundary of the nominal safe component: the
certified end advances to a pointwise-safe sample only when the gap from the
current certified end is covered by that sample's own safe radius
`(F - F_T)/L` (or bridged by safe-radius continuation), so with a nonmonotone
fidelity a safe island beyond the first crossing cannot inflate `M`. On the
case study `margin_tol=1e-10` reaches `1e-10` in about 90 extra fidelity
evaluations per controller. `reason_minus/plus` distinguishes `'bracketed'`
(width at tolerance) from `'partial'` (the bracket is rigorous but
continuation stalled before reaching the tolerance), `'boundary'` (the
domain edge was reached while still safe -- then the margin is a *domain
truncation*, and `M_upper = inf`), and `'exhausted'` (no unsafe point
found; `M_upper = inf`).

The paper drivers do not pass `margin_tol`, so the published tables are
reproduced exactly; the option is there for anyone who needs the true value.

---

## 5. Kosut-Lidar-Rabitz time-bandwidth bound (supplementary, experimental)

A supplementary layer implements Theorem 1 of arXiv:2507.01215 specialised to
this model. It is outside the reproduction gate and no paper claim depends on
it. [time-bandwidth-bound.md](time-bandwidth-bound.md) gives the
specialisation, the caveats, the results and the accuracy analysis; only the
accuracy classification is repeated here, since section 6 indexes it.

With `Htil(t) = U_S(t)^dag H_unc(t) U_S(t)` the interaction-picture
uncertainty, the three measures of Eq. 28 of that reference are

| measure | definition | here | accuracy |
|---|---|---|---|
| `Omega_unc` | `max_t \|\|H_unc(t)\|\|` | `max_k \|\|Hhat^(k)\|\|_2` | exact; `H_unc` is piecewise constant |
| `Omega_avg` | `\|\| <Htil> \|\|` | closed-form time average | exact to roundoff |
| `Omega_avg^dev` | `max_t \|\|Htil(t) - <Htil>\|\|` | bandwidth-derived grid, Brent polish | the only approximated quantity |

All three are linear in `delta`, so the per-unit rates `w_unc, w_avg, w_dev`
are computed once and the bound inverts in closed form without a root find.

`<Htil>` is exact for the same reason `dU/dmu` is (section 2): on each interval
`int_0^Delta e^{+iHs} A e^{-iHs} ds` is the same divided difference in the
eigenbasis of `H`, with `Y_mn = Delta(lam_m - lam_n)/2` in place of `X_mn`.

`Omega_avg^dev` is a supremum over `t` and has no closed form. Sampling can only
under-estimate a supremum, and a smaller `w_dev` gives a larger implied margin,
so the residual error is biased in the optimistic direction. The sampling
density is derived from the exactly-known Bohr bandwidth of `Htil(k, .)` and the
result carries two rigorous certificates, `w_dev_certified` and
`w_dev_bracket_lo/hi`; see [time-bandwidth-bound.md](time-bandwidth-bound.md).

---

## 6. Accuracy summary

| Quantity | Function | Accuracy |
|---|---|---|
| `U(t_f)` | `propagator` | exact (backward-stable `expm`) |
| `F_mu` | `gate_fidelity` | exact |
| `B_T`, `L_Hhat` | `lipschitz_constant` | exact given inputs |
| `C_Hhat` | `structure_constant` | exact given inputs |
| `zeta` | `differential_sensitivity` | exact (`method='exact'`, default); `'quadrature'` is an alternative and test oracle |
| `dF/du` | `fidelity_and_gradient` | as `zeta` |
| `M` | `iterative_margin` | lower bound; about 5e-4 relative at the default `eta`; to `margin_tol` on request |
| `w_unc`, `w_avg` | `kosut.uncertainty_rates` | exact |
| `w_dev` | `kosut.uncertainty_rates` | bandwidth-derived grid + Brent polish + two certificates; `dev_resolved` / `dev_converged` report success |
| `T Omega_bnd`, `F_lb`, `kosut.margin` | `kosut.*` | closed form; inherit `w_dev`'s uncertainty only |

The two quantities that are not exact -- `M` and `w_dev` -- are the two that
carry explicit error control. Nothing else in the package uses a fixed
discretisation without an estimate.

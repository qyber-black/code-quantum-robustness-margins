# Comparison with the Kosut-Lidar-Rabitz time-bandwidth bound

> SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>\
> SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>\
> SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>\
> SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>\
> SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>\
>
> SPDX-License-Identifier: AGPL-3.0-or-later

Kosut, Lidar and Rabitz derive a universal worst-case performance limit for
coherent gate control ([arXiv:2507.01215](https://arxiv.org/abs/2507.01215),
referred to below as the reference). `python/src/qrobustness/kosut.py` and
`matlab/+qrobustness/+kosut/` implement Theorem 1 of the reference, specialised
to the closed-system, purely coherent, scalar structured perturbation model of
this package, so that the finite perturbation margin it implies can be placed
alongside the certified Lipschitz margin of Algorithm 1.

This layer is supplementary. It is experimental, it sits outside the
reproduction gate `make check-margins`, and no claim in the accompanying paper
depends on its values. For the accuracy of the package as a whole see
[theory.md](theory.md).

## Specialisation

The uncertain Hamiltonian of the reference (its Eqs. 3-8) is bipartite,

```
H_unc = H_S^coh (x) I_B  +  I_S (x) H_B  +  H_SB
```

where `(x)` is the tensor product and `B` labels the bath. Here the bath is
absent (`H_B = H_SB = 0`) and the coherent error is the structured scalar
perturbation of this package on interval `k`,

```
H_unc^(k) = delta * Hhat^(k)
```

With `U_S(t)` the nominal propagator, `'` the conjugate transpose,
`Htil(t) = U_S(t)' H_unc(t) U_S(t)` the interaction-picture uncertainty, and
`<A> = (1/T) int_0^T A(t) dt` the time average, Eq. 28 of the reference becomes

| Measure         | Definition                        | Here                             | Scaling        |
|-----------------|-----------------------------------|----------------------------------|----------------|
| `Omega_unc`     | `max_t norm(H_unc(t))`            | `max_k norm(Hhat^(k))`           | `~ abs(delta)` |
| `Omega_avg`     | `norm(<Htil>)`                    | interaction-picture time average | `~ abs(delta)` |
| `Omega_avg^dev` | `max_t norm(Htil(t) - <Htil>)`    | its deviation from that average  | `~ abs(delta)` |

where `norm(.)` is the induced 2-norm. A scalar perturbation enters linearly, so
the per-unit rates `w_unc, w_avg, w_dev` are properties of the controller and
the structure, computed once by `uncertainty_rates`, and

```
T*Omega_bnd(delta) = sqrt( a*delta^2 + b*abs(delta) ),
    a = T^2 * w_unc * w_dev,    b = 4 * T * w_avg

F_lb = max( 1 - 0.5 * (exp((T*Omega_bnd/2)^2) - 1)^2 , 0 )
```

(Eqs. 29-30 of the reference). `T*Omega_bnd` is monotone in `abs(delta)`, so the
implied margin `M^K`, the largest `abs(delta)` with `F_lb >= F_T`, follows in
closed form (`margin` / `kosut_margin`); no root find is required.

## Why the comparison is meaningful

`F_avg^low` (Eq. 24 of the reference) lower-bounds `abs(Tr Util(T)) / d`, the
fidelity to the *achieved* nominal gate `U_S(t_f)`. Up to the nominal-error
absorption of caveat 1 below, both quantities are certified lower bounds on the
same `F_mu`, measured in the same units of multiplicative perturbation of
`Hhat`.

## Caveats

1. Theorem 1 of the reference assumes `F_nom = 1` exactly, and it bounds the
   fidelity to the *achieved* nominal gate, not to the target. The controllers
   used here have `eps_0 <= 1e-4`, which is 10% of the `1 - F_T = 1e-3` budget,
   so the deficit cannot be ignored. Because `arccos` of the gate fidelity is
   the angle between the corresponding Choi states, it obeys the triangle
   inequality and the *angles* add, not the fidelity deficits. The sufficient
   condition on the achieved-gate fidelity is therefore

   ```
   F_eff = cos( arccos(F_T) - arccos(1 - eps_0) )
   ```

   (`effective_threshold`, `absorption='angular'`, the default since 1.0.1).
   The additive form `F_T + eps_0` used before 1.0.1 is *smaller* than `F_eff`
   whenever `eps_0 > 0` and so was **not** conservative; it remains available
   as `absorption='additive'` / `--absorption additive` to reproduce the older
   numbers. When `arccos(1 - eps_0) >= arccos(F_T)` the budget is exhausted and
   the implied margin is zero. `--literal-theorem` (Python) and
   `'literal_theorem', true` (MATLAB) evaluate the bound as stated
   (`eps_0 = 0`) instead.
2. `M^K` is the **constant structured-parameter** specialisation. `w_avg` and
   `w_dev` are computed for the fixed structure `delta * Hhat`, so the implied
   margin certifies constant perturbations `abs(delta) <= M^K`. It is *not* a
   supremum-norm time-varying margin: a sign-modulated trajectory `delta(t)`
   within the same budget can defeat the coherent averaging that makes `w_avg`
   small.
3. The bound covers strictly more uncertainty. It is worst-case over all
   uncertainty consistent with the norm bounds, including bath coupling and
   unmodelled couplings, whereas Algorithm 1 exploits the known structure
   `Hhat` and the specific controller. A larger margin here quantifies the
   value of structural knowledge; it is not evidence against the bound.
4. `F_lb` is non-trivial only for `T*Omega_bnd <= 2*sqrt(log(1+sqrt(2)))
   = 1.8776` rad (Eq. 32 of the reference); beyond that the bound is vacuous.
   `t_omega_max` / `T_OMEGA_MAX` expose this value.
5. Both are lower bounds on the true threshold-crossing radius, so the
   comparison ranks conservatism rather than accuracy.

## Numerical accuracy of the interaction-picture quantities

Extracting time averages and deviations from average of interaction-picture
operators is the delicate part of implementing the bound. Each claim below is
asserted in `test_error_control`.

`Omega_unc` is exact. `H_unc` is piecewise constant, so its supremum over `t` is
a maximum over intervals and involves no sampling.

`Omega_avg` is exact to roundoff. On each interval
`int_0^Delta e^{iHs} A e^{-iHs} ds` is a closed-form divided difference in the
eigenbasis of `H`, so no quadrature is involved. `n_quad` is accepted by
`uncertainty_rates` and unused.

`Omega_avg^dev` is a supremum over `t` and the only approximated quantity.
Sampling can only under-estimate a supremum, and a smaller `w_dev` yields a
larger implied margin, so the error is biased in the optimistic direction. Two
properties keep it controlled:

- `Htil(k, .)` is a trigonometric polynomial whose frequencies are exactly the
  Bohr frequencies `lam_m - lam_n`, so its bandwidth is known in closed form.
  The sampling density is derived from it, at `dev_samples_per_cycle` samples
  per cycle per interval (default 16), which maintains resolution for
  controllers with long intervals or wide spectra. Candidate maxima are then
  polished with Brent to near machine precision, and the grid is refined until
  successive sweeps agree to `dev_tol`.
- Two independent rigorous certificates bound the result: a Lipschitz bound
  from `d(Htil)/ds = i[H, Htil]`, giving a constant `2 norm(H) norm(Hhat)`, and
  an isospectral bracket from `norm(Htil(t)) = norm(Hhat^(k))` exactly.

`dev_resolved`, `dev_samples_per_cycle`, `dev_cycles_max` and `dev_converged`
report whether the search succeeded.

Measured on the paper controller set, which has 1.06 cycles per interval, the
polished `Omega_avg^dev` agrees with a 20001-point reference sweep to better
than 1e-10 relative. On a variant with the interval lengthened twentyfold, at
21 cycles per interval, the bandwidth-derived grid attains a relative error of
2e-9 against 3.7e-5 for a 17-point grid.

Agreement between Python, MATLAB and Octave is a consistency check rather than
an accuracy check, since all three implement the same algorithm.

## Result for the paper controller set

61 controllers, `F_T = 0.999`, `eps_0` absorbed through the angular relation of
caveat 1. `M` is the Algorithm 1 margin and `M^K` the margin implied by
Theorem 1 of the reference:

| Structure | median `M` | median `M^K` | ratio `M/M^K` (range) | Pearson |
|-----------|------------|--------------|-----------------------|---------|
| `H0`      | 5.70e-3    | 2.75e-3      | 2.12 (1.74-2.50)      | 0.83    |
| `H1`      | 8.92e-3    | 2.97e-3      | 2.92 (2.13-4.74)      | 0.78    |
| `H2`      | 9.41e-3    | 3.14e-3      | 3.03 (2.40-4.19)      | 0.82    |

The structured margin is larger for every controller and structure, by a factor
of roughly 2-3 rather than orders of magnitude, and the two rank controllers
similarly. The gap widens for the control structures `H1, H2`, where the
interaction-picture time average captured by `Omega_avg` is less effective at
exploiting the specific pulse sequence.

Before 1.0.1 the additive absorption gave larger `M^K` (median 3.07e-3, 3.44e-3
and 3.36e-3) and correspondingly smaller ratios of 1.84, 2.56 and 2.65; those
values are reproducible with `--absorption additive` but rest on a threshold
that is not sufficient for the target-gate condition.

Per-controller values, the per-unit rates `w_*`, and `T*Omega_bnd` evaluated at
the Lipschitz margin are in `kosut_comparison_0.999.csv` in each results tree;
`kosut_vs_lipschitz_0.999.png` is the corresponding log-log scatter.

## Reproducing it

```bash
make time-bandwidth-bound           # Python + MATLAB + cross-check
make time-bandwidth-bound-python    # -> results/time-bandwidth-bound-python/
make time-bandwidth-bound-matlab    # -> results/time-bandwidth-bound-matlab/
make time-bandwidth-bound-octave    # -> results/time-bandwidth-bound-octave/
make compare-time-bandwidth-bound   # Python vs MATLAB tables -> build/
```

The three implementations agree to about `1e-13` on the full 61-controller
table (`make compare-time-bandwidth-bound`, `atol=1e-10`, `rtol=1e-8`). The
drivers write identical CSV columns, fixed by `CSV_HEADERS` in
`scripts/run_time_bandwidth_bound_comparison.py` and by
`qrobustness.compat.kosut_csv_headers` in MATLAB, with a parity test in
`python/tests/test_kosut.py`.

The golden fixtures in `data/reference/` carry the `k_*` fields (`k_w_unc`,
`k_w_avg`, `k_w_dev`, `k_T`, `k_M`), so `make test-consistency` cross-checks the
bound across languages alongside the margin and sensitivity values. Unit tests
are `python/tests/test_kosut.py` and `matlab/tests/test_kosut_bound.m`; both
verify that the specialised bound lower-bounds the true perturbed fidelity and
that `M^K` never exceeds the true threshold-crossing radius.

## API

See [api.md](api.md) for the full signatures.

| Function | Role |
|----------|------|
| `uncertainty_rates(H_list, dH_list, dt, ...)` | Eq. 28 measures per unit `delta` |
| `time_bandwidth(rates, delta)` | `T*Omega_bnd` (Eq. 29) |
| `fidelity_bound(T_omega_bnd)` / `fidelity_bound_at(rates, delta)` | `F_lb` (Eq. 30) |
| `threshold_time_bandwidth(FT, nominal_error=0)` | Closed-form inverse of `F_lb` |
| `margin(rates, FT, nominal_error=0)` | Implied margin `M^K` (Python re-export: `kosut_margin`) |
| `t_omega_max()` / `T_OMEGA_MAX` | `2*sqrt(log(1+sqrt(2))) = 1.8776`; bound vacuous beyond this |

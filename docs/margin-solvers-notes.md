# Margin solvers

Selectable one-dimensional margin methods on `iterative_margin` (Python and MATLAB).
Default is paper **Algorithm 1** (`method="algorithm1"`): Lipschitz steps of size
\((F-F_T)/L\) with in-place bisection on overshoot. Paper drivers and goldens omit
`method` and stay on Algorithm 1. These variants do not affect Algorithm 1 as stated in the manuscript.

## Lipschitz constraint

From the paper safe-radius theorem: from a safe point with fidelity \(F\), only a step
of length \(\le (F-F_T)/L\) is *a priori* safe on the connected component \(\mathcal{I}\).
Bracketed root finders (bisection, Brent, TOMS748) are valid **inside** a known
\([\mu_{\mathrm{safe}}, \mu_{\mathrm{unsafe}}]\) bracket. Probes beyond the Lipschitz radius
are **not** certificate-preserving unless every intermediate point is known safe
(e.g. monotone \(F\) on the ray).

## Methods

| method | Certificate | Role |
|--------|-------------|------|
| `algorithm1` | Full | **Preferred certified default** (paper): Lipschitz + bisection |
| `lipschitz_brent` / `lipschitz_toms748` | Full | Same advance; optional polish only |
| `doubling` | Weaker (endpoint) | Geometric probes when \(L\) is conservative |
| `newton_probe` | Weaker when probing | Uses \(\zeta(\mu)\) for step size; needs `zeta_fn` |

Recommendation: Lipschitz + bisection is the best fully certified option for the
paper path. Brent/TOMS748 do not appreciably reduce fidelity evaluations when
Lipschitz steps rarely overshoot; they are selectable polish rather than the default.
`doubling` / `newton_probe` can reduce evaluations substantially under conservative \(L\), but only with
a weaker endpoint certificate unless \(F\) is monotone on the ray -- out of scope for
changing Algorithm 1 in the manuscript.

Pointwise \(B(F)=\sqrt{(1-F^2)/N}\) does **not** enlarge a certified step all the way
to \(F_T\) (the segment still approaches \(B_T\)). Newton is a probe heuristic, not a
larger certified Lipschitz step.

MATLAB maps both `brent` and `toms748` polish to `fzero` (Brent-like).

## Optional benchmark

```bash
python scripts/bench_margin_solvers.py
# optional: --controllers 5  or  --skip-case-study
```

Writes gitignored `results/bench-margin-solvers/bench_margin_solvers.csv`.

Illustrative snapshot (`--controllers 2`): with conservative \(L\), Algorithm 1 /
Lipschitz+Brent|TOMS748 used ~3679 evals while `doubling` / `newton_probe` used tens
or fewer; on case-study samples the aggressive methods were typically 0.3-0.7x Algorithm 1
evals with endpoint `cert_ok`. Lipschitz polish matched Algorithm 1 eval counts when
overshoot was rare.

# Repository layout

## Policy

| Path | Contents |
|------|----------|
| `data/controllers/<id>/` | Non-reproducible inputs (Hamiltonians + controller CSV). Paper set is frozen. |
| `data/reference/` | Golden fixtures for MATLAB<->Python consistency checks. |
| `data/legacy/` | Superseded artefacts for historical comparison only. |
| `results/lipschitz-margin-matlab/` | Full paper deliverables from MATLAB. |
| `results/lipschitz-margin-python/` | Full paper deliverables from Python. |
| `results/lipschitz-margin-octave/` | Full paper deliverables from Octave. |
| `results/synth-matlab/` | Regenerable synthesised controllers (MATLAB). |
| `results/synth-python/` | Regenerable synthesised controllers (Python). |
| `results/synth-*-margins/` | Margin analysis of a synth set (local / regenerable). |
| `results/bench-margin-solvers/` | Optional solver benchmark CSV (gitignored). |
| `results/time-bandwidth-bound-matlab/` | Supplementary Kosut et al. bound comparison (MATLAB). |
| `results/time-bandwidth-bound-python/` | Supplementary Kosut et al. bound comparison (Python). |
| `results/time-bandwidth-bound-octave/` | Supplementary Kosut et al. bound comparison (Octave peer). |
| `build/` | Regenerable scratch (compare logs, smoke). Gitignored. |
| `docs/` | Theory (what is computed + accuracy), API contract, time-bandwidth bound, layout, margin-solver notes. |

All `results/lipschitz-margin-*` trees correspond to controller set `problem9_tf15_K32_quasi-newton`.
Synthesis writes new ensembles under `results/synth-*` and never overwrites `data/controllers/`.

Conventions: Python is the reference implementation and produces the manuscript
figures; MATLAB and Octave are peers, held to it by `make compare-full`,
`make compare-octave` and the golden fixtures. Sources and documentation are
ASCII, with mathematical symbols in LaTeX-like notation. Prose uses British
spelling; identifiers keep the spelling they are declared with.

Octave figures are rendered with the qt toolkit when a display is available and
with gnuplot otherwise, so Octave PNGs are not byte-identical across
environments. Only the CSV tables are compared between engines
(`make compare-octave`), so this does not affect any gate.

Result trees are named after the **method** they implement (`lipschitz-margin`,
`time-bandwidth-bound`), not after the paper that happens to publish them. The
only paper-aware part of the build is the `PAPER_*` block at the top of the
`Makefile` -- which analysis a given paper publishes, where its figures go, and
which LaTeX source `verify_paper_consistency` reads -- plus the `sync-paper-*`
and `verify-paper-*` targets that use it. A second paper is a new block there,
not a code change.

The paper lives in a **sibling repository**, not inside this one:

```
QRM/
  code-robustness-margins/          # this repository
  paper-QRM/                        # the paper (PAPER_ROOT)
```

So publishing means copying results across a repository boundary. The target is
named after the **paper**, not the language: `sync-paper-qrm` (aliased as
`sync-paper`) pushes the Python PNGs into `$(PAPER_ROOT)/figures/`. Python is
the reference implementation and the only published tree; MATLAB and Octave are
peers, compared (`make compare-full`, `make compare-octave`) but never
published. The path is a Make variable -- pass `PAPER_ROOT=` for a different
checkout. `verify_paper_consistency` locates the paper the same way
(`--paper-source`, `$QRM_PAPER_SOURCE`, then the sibling checkout) and skips its
paper checks on a code-only clone rather than failing.

## Lipschitz-margin deliverables (each of `results/lipschitz-margin-matlab/`, `results/lipschitz-margin-python/`, `results/lipschitz-margin-octave/`)

- `H0_all.png`, `H1_all.png`, `H2_all.png`
- `robustness_margins_fid_err.png`
- `robustness_margins_sensitivity.png`
- `correlations_0.999.tex` (Table I source; upper triangle Pearson \(r\), lower triangle Spearman \(\rho\), both descriptive)
- `focal_tests_0.999.csv` (rank-statistic cross-check on \(M_j\) vs \(|\zeta_j|\): Spearman \(\rho\) and Kendall \(\tau_b\), each with Holm-corrected two-sided \(p\). Not a paper claim -- it confirms the descriptive reading of Table I does not depend on the rank statistic chosen.)
- `margins_table_0.999.csv` (for `make compare-full` / `make compare-octave`)
- `verify_paper.md` (consistency report for that tree)

## Synthesis deliverables (`results/synth-matlab/`, `results/synth-python/`)

- `problem9.mat` (copy of the paper problem / \(U_f\))
- `controllers.csv` (all optimised runs; same schema as the paper CSV)
- `meta.json` (seed, \(N_{\mathrm{opt}}\), method, error stats)

## Reproduction

```bash
make lipschitz-margin-matlab           # MATLAB -> results/lipschitz-margin-matlab/
make lipschitz-margin-python           # Python -> results/lipschitz-margin-python/ (includes H*_all sweeps)
make lipschitz-margin-octave           # Octave -> results/lipschitz-margin-octave/ (optional peer)
make compare-full           # compare Python vs MATLAB margin tables
make compare-octave         # compare Python vs Octave margin tables
make sync-paper-qrm         # paper figures (Python results) -> $(PAPER_ROOT)/figures/
make export-golden          # Python + MATLAB goldens -> data/reference/
make verify-paper-matlab    # -> results/lipschitz-margin-matlab/verify_paper.md
make verify-paper-python    # -> results/lipschitz-margin-python/verify_paper.md
make verify-paper-octave    # -> results/lipschitz-margin-octave/verify_paper.md
make verify-paper           # Python + MATLAB verifiers
make check-margins            # lipschitz-margin-matlab/python + compare-full + verify-paper (release gate)
make time-bandwidth-bound       # Kosut et al. bound: Python + MATLAB + compare-time-bandwidth-bound
make compare-time-bandwidth-bound          # Python vs MATLAB Kosut tables -> build/
make synth-matlab           # optimise 100 controllers -> results/synth-matlab/
make synth-python           # optimise 100 controllers -> results/synth-python/
make synth-smoke            # 2-start smoke synthesis
make analyse-synth-matlab   # margins for synth-matlab
make analyse-synth-python   # margins for synth-python
make clean                  # remove build/
make distclean              # remove build/ and .venv/
```

# Repository layout

## Policy

| Path | Contents |
|------|----------|
| `data/controllers/<id>/` | Non-reproducible inputs (Hamiltonians + controller CSV). Paper set is frozen. |
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
| `results/multiparameter-margin-python/` | xQRM Scenario J: joint static margins, gauge regions, slice scans, trajectory brackets. |
| `results/time-bandwidth-bound-python/` | xQRM Scenario T as well: universal-bound comparison, adversarial validity sweeps, budget sweep, Berberich comparison. |
| `results/single-qubit-python/` | xQRM single-qubit case against analytic truth. |
| `results/cnot-python/` | xQRM CNOT transfer: margins, duration sweep, robustified-vs-nominal selection study. |
| `results/scaling-python/` | xQRM four-qubit scaling demonstration. |
| `results/lindblad-margin-python/` | xQRM Scenario D: Lindblad rate margins, amplitude damping, threshold sweep, mixed coherent-dissipative, coherent-through-open comparison. |
| `results/verification-python/` | xQRM certificate verification harness: every theorem probed over the ensemble. |
| `results/paper-xqrm/` | Generated xQRM paper artefacts (tables, figures, macros), copied to the paper repository by `make sync-xQRM`. |
| `build/` | Regenerable scratch (compare logs, smoke). Gitignored. |
| `docs/` | Theory (what is computed + accuracy), API contract, time-bandwidth bound, layout, margin-solver notes. |

All `results/lipschitz-margin-*` trees correspond to controller set `problem9_tf15_K32_quasi-newton`.
The xQRM trees above are Python-only: the peers implement the paper-1
certificates, not the joint, Choi-trajectory and Lindblad extensions. The one
exception is the universal-bound comparison. The peers cover all three of
its invocations: the angular default, `absorption='additive'`, and
`uncertainty='trajectory'` for the sup-norm trajectory class M^K_tv. Both
peer trees therefore carry the same three CSVs as the Python tree, and
`make test-parity` compares them. Which driver
writes each file is recorded in `scripts/_paper.py` and checked against the
Makefile by `test_paper_driver_map_matches_the_makefile`.
Synthesis writes new ensembles under `results/synth-*` and never overwrites `data/controllers/`.

Conventions: Python is the reference implementation and produces the manuscript
figures; MATLAB and Octave are peers, held to it by `make test-parity
ENGINE=matlab` / `ENGINE=octave` and by each engine's consistency test,
which computes live and compares against the other engine's committed
results table. Sources and documentation are
ASCII, with mathematical symbols in LaTeX-like notation. Prose uses British
spelling; identifiers keep the spelling they are declared with.

Octave figures are rendered with the qt toolkit when a display is available and
with gnuplot otherwise, so Octave PNGs are not byte-identical across
environments. Only the CSV tables are compared between engines
(`make test-parity ENGINE=octave`), so this does not affect any gate.

Result trees are named after the **method** they implement (`lipschitz-margin`,
`time-bandwidth-bound`), not after the paper that happens to publish them. The
only paper-aware part of the build is the `PAPER_*` block at the top of the
`Makefile`: which analysis a given paper publishes, where its figures go, and
which LaTeX source `verify_paper_consistency` reads. The `sync-*` and
`check-*` targets use it. A second paper is a new block there, not a code
change.

Papers live in **sibling repositories**, not inside this one:

```
QRM/
  code-robustness-margins/          # this repository
  paper-QRM/                        # paper 1 (PAPER_ROOT)
  paper-xQRM/                       # paper 2 (XPAPER_ROOT)
```

So publishing means copying results across a repository boundary. The targets
are named after the **paper**, not the language: `sync-QRM` pushes the Python
PNGs into `$(PAPER_ROOT)/figures/`, and `sync-xQRM` regenerates paper 2's
tables and figures from this tree into `$(XPAPER_ROOT)/`. Python is the
reference implementation and the only published tree;
MATLAB and Octave are peers, compared (`make test-parity ENGINE=matlab` /
`ENGINE=octave`) but never published. Both paths are Make variables -- pass `PAPER_ROOT=` or
`XPAPER_ROOT=` for a different checkout.

The flow is one-way. `make check` reads nothing outside this repository:
publication is a copy out of it, so a paper checkout can be behind these
results, and a check against a stale manuscript would fail on the sync rather
than on a wrong number.

## Lipschitz-margin deliverables (each of `results/lipschitz-margin-matlab/`, `results/lipschitz-margin-python/`, `results/lipschitz-margin-octave/`)

- `H0_all.png`, `H1_all.png`, `H2_all.png`
- `robustness_margins_fid_err.png`
- `robustness_margins_sensitivity.png`
- `correlations_0.999.tex` (Table I source; upper triangle Pearson \(r\), lower triangle Spearman \(\rho\), both descriptive)
- `focal_tests_0.999.csv` (rank-statistic cross-check on \(M_j\) vs \(|\zeta_j|\): Spearman \(\rho\) and Kendall \(\tau_b\), each with Holm-corrected two-sided \(p\). Not a paper claim -- it confirms the descriptive reading of Table I does not depend on the rank statistic chosen.)
- `margins_table_0.999.csv` (compared by `make test-parity`)
- `verify_paper.md` (consistency report for that tree)

## Synthesis deliverables (`results/synth-matlab/`, `results/synth-python/`)

- `problem9.mat` (copy of the paper problem / \(U_f\))
- `controllers.csv` (all optimised runs; same schema as the paper CSV)
- `meta.json` (seed, \(N_{\mathrm{opt}}\), method, error stats)

## Reproduction

`make help` lists every target with its current name. The ones that decide
what lands where, with `PAPER` one of `QRM`, `xQRM` and `ENGINE` one of
`python` (the reference implementation), `matlab`, `octave`:

```bash
make paper-PAPER ENGINE=E   # compute that paper's results from that engine
make reproduce-PAPER        # recompute into a scratch tree and compare
make check-PAPER            # falsifiable property checks on the results
make test ENGINE=E          # lint, unit tests, synthesis smoke, parity
make sync-PAPER             # copy the generated artefacts into the paper repo
make clean | distclean      # remove build/ | build/ and .venv/
make maintainer-clean       # also every generated result, to start over
```

There are no golden fixtures. The committed `results/` trees are the
reference: each engine's consistency test computes live and compares
against the *other* engine's committed table, and `git diff` reports any
numeric change across a regeneration. See `docs/theory.md`.

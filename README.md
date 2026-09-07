# Fidelity-Based Quantum Robustness Margins

> SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>\
> SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>\
> SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>\
> SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>\
> SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
>
> SPDX-License-Identifier: AGPL-3.0-or-later

Toolbox for **analysing** (and optionally **synthesising**) piecewise-constant coherent gate controllers under structured Hamiltonian and Lindbladian uncertainty. It carries the code and results behind two papers:

- **QRM** -- *Fidelity-Based Robustness Margins for Finite-Time Quantum Control*, [arXiv:2608.03698](https://arxiv.org/abs/2608.03698). The scalar-perturbation margin: the gate-fidelity sensitivity bound, the Lipschitz constant it induces, and Algorithm 1.
- **xQRM** -- *Certified robustness regions under joint, time-dependent, and Lindbladian uncertainty*. In preparation; not yet on arXiv. Joint static regions over several parameters, the Choi-Fubini-Study trajectory certificate, Lindblad rate margins, and the comparison bounds.

`PAPER` selects between them throughout the build: `make paper-QRM`, `make check-xQRM`, and so on.

> **Scope:** Manuscript figures use the frozen paper controller set under `data/controllers/`. Synthesis writes new ensembles to `results/synth-*` and never overwrites that set. The paper reproduction path uses Algorithm 1 (Lipschitz + bisection); other margin solvers are optional -- see [docs/margin-solvers-notes.md](docs/margin-solvers-notes.md).

> **What is computed, and how accurately:** see [docs/theory.md](docs/theory.md) -- every reported quantity, the paper result it comes from, and its accuracy. Two quantities are not exact to roundoff and carry explicit error control: the margin `M` (a *lower* bound; ~5e-4 relative at the default `eta`, refinable to a requested precision via `margin_tol`) and the Kosut `w_dev`.

> **Provenance of the controllers.** The margin, sensitivity and figure results under `results/` are produced by this code. The **controller ensemble** in `data/controllers/`, however, is not: it comes from earlier optimisation runs and is shipped frozen rather than regenerated here. The `optimize_controller` routine in this toolbox reproduces that synthesis *workflow*, but not necessarily the algorithm, parameters or constraints used for the original run -- and the MATLAB (`fminunc`, quasi-Newton) and Python (`L-BFGS-B`) implementations here already differ from each other. Re-synthesising will therefore give a different, equally valid ensemble; it will not reproduce `data/controllers/` element by element.

## Features

- Gate propagator and normalised gate fidelity
- Lipschitz constant \(L_{\hat{H}}\) from the paper's sensitivity bound
- Differential sensitivity \(\zeta\), with the segment derivative in exact closed form (Gauss-Legendre quadrature selectable via `method='quadrature'`)
- Iterative one-dimensional robustness margin (Algorithm 1 default; selectable solvers)
- Dual MATLAB and Python APIs over the certificate layers, with matching
  paper-style plots (see below for what the peer does not cover)
- Paper case-study reproduction via `make paper-QRM-margins` in any engine
- Supplementary comparison with the Kosut et al. fundamental bound (arXiv:2507.01215) -- experimental, see below
- Fidelity-maximising controller synthesis (GRAPE + quasi-Newton), smoke-tested
  by `make test-synth`; run `scripts/run_synthesize_controllers.py` (or its
  MATLAB peer) directly to generate a full ensemble. Synthesis is a separate
  experiment: the frozen ensemble remains the sole paper input.

## Requirements

- **Python** 3.10+ with NumPy, SciPy, pytest; matplotlib for plots (`qrobustness[plot]`). Reference implementation, and the source of the manuscript figures.
- **MATLAB** R2020b+ (peer; required by `make test-parity ENGINE=matlab`). GRAPE synthesis (`optimize_controller`, `make test-synth ENGINE=matlab`) additionally needs the Optimization Toolbox for `fminunc`.
- **Octave** 7+ (peer: any target with `ENGINE=octave`). No Octave Forge packages are required: `fminunc` and `optimset` ship in core Octave, and the rank statistics are computed in-package.
- Git LFS for large `.mat` / `.png` artefacts
- The open-system certificates additionally need `cvxpy` (`qrobustness[open]`) and a semidefinite solver. The toolbox names the solver rather than letting `cvxpy` pick one, because the choice moves certified diamond norms by orders of magnitude; see [docs/theory.md](docs/theory.md).
- `make venv` installs the pinned set in [`python/requirements-repro.txt`](python/requirements-repro.txt), the environment the published results were computed in. Pass `PINS=` to build against current releases instead.

## Quick start

### Python

```bash
cd python
pip install -e ".[dev]"
pytest
```

### MATLAB

```matlab
addpath('matlab');
help qrobustness.iterative_margin
```

```bash
make test ENGINE=matlab
```

## Reproduce paper results

Inputs: [`data/controllers/problem9_tf15_K32_quasi-newton/`](data/controllers/problem9_tf15_K32_quasi-newton/)

```bash
make test                 # lint, unit tests, synthesis smoke, parity against
                          # Python; every stage runs, then one tally. The stages
                          # are targets too: test-lint, test-unit, test-synth,
                          # test-parity (a no-op when ENGINE=python).
make paper-PAPER          # produce one paper's results (PAPER = QRM | xQRM)
make reproduce-PAPER      # recompute separately and compare against the tree
make check-PAPER          # falsifiable property checks on that paper's results
make sync-PAPER           # copy the generated artefacts into the paper repo
```

`ENGINE` selects the implementation for every target and defaults to
`python`, the reference: `make paper-xQRM-multiparam ENGINE=octave`. A
target with no implementation for the chosen engine fails rather than
falling back to Python, so a pass always means the work actually ran.
Add `-EXPNAME` to `paper-` or `check-` for a single experiment, and
`make help` lists the names.

The paper repositories must be sibling directories; pass `PAPER_ROOT` or
`XPAPER_ROOT` to override their locations. Python is the reference
implementation. MATLAB and Octave are peers, held to it by cross-engine
comparisons against each other's committed result tables. `make help` lists
the lower-level analysis, comparison, synthesis, and cleanup targets.

The peer covers the certificates, not the whole library. `+qrobustness`
implements the core margin machinery, the multiparameter and trajectory
gauges, the length-space layer, the Lindblad certificates, and both
comparison bounds (Kosut and Berberich), and the cross-engine checks run on
exactly those. Three things are Python-only and have no MATLAB peer: the
**adversarial search** (`timevarying.adversarial_fidelity` and the upper
witnesses built on it), **GRAPE synthesis** (`synthesis.grape*`;
`optimize_controller` is the peer of the older workflow, not of these), and
the **certificate harness** (`qrobustness.verify`). Results that depend on
those are produced by the Python drivers alone.

One name means different things in the two engines, deliberately.
`lindblad.diamond_norm` is the cvxpy SDP in Python and the solver-free
upper bound in MATLAB, whose Python peer is `diamond_norm_free`: MATLAB and
Octave have no portable SDP solver, so the solver-free method is the only
one there and takes the plain name.

## Comparison with the Kosut-Lidar-Rabitz time-bandwidth bound

A supplementary layer implements Theorem 1 of
[arXiv:2507.01215](https://arxiv.org/abs/2507.01215), specialised to the
closed-system scalar structured perturbation model used here, so that the
margin it implies can be placed alongside the certified Lipschitz margin.
Supplementary to the first paper, which makes no claim resting on it, and
load-bearing for the second, whose time-variation table, figures and macros
compare against it. The structured margin is larger by a median factor of
2.1 to 3.0 across the three perturbation structures. The implied margin is
the constant structured-parameter specialisation, with the nominal error
absorbed through the angular relation (see the caveats).

See [docs/time-bandwidth-bound.md](docs/time-bandwidth-bound.md) for the
specialisation, the caveats, the numerical accuracy of the interaction-picture
quantities, the full results and `make paper-QRM-time-bandwidth`.

## CI

GitLab CI ([`.gitlab-ci.yml`](.gitlab-ci.yml)) on push/MR runs **Python** `pytest` and an sdist/wheel build (`python -m build`). It pulls Git LFS so the frozen controller ensembles in `data/controllers/` are available. MATLAB tests and the full paper gate (`make check`) stay local. Pipelines: <https://qyber.black/lw1660/code-robustness-margins/-/pipelines>.

Each of [`results/lipschitz-margin-matlab/`](results/lipschitz-margin-matlab/), [`results/lipschitz-margin-python/`](results/lipschitz-margin-python/), and [`results/lipschitz-margin-octave/`](results/lipschitz-margin-octave/) contains:

| Artefact | Role |
|----------|------|
| `H0_all.png` ... `H2_all.png` | Fidelity error vs \(\delta_j\) |
| `robustness_margins_fid_err.png` | Margins vs controller index |
| `robustness_margins_sensitivity.png` | Margins vs \(\lvert\zeta\rvert\) |
| `correlations_0.999.tex` | Table I |
| `margins_table_0.999.csv` | Numeric table compared across engines |
| `verify_paper.md` | Consistency report for that tree |

## Documentation

| Document | Covers |
|----------|--------|
| [docs/theory.md](docs/theory.md) | What each quantity is, the paper result it comes from, and how accurately it is computed |
| [docs/api.md](docs/api.md) | The shared API contract: which call in each engine computes which quantity |
| [docs/verification.md](docs/verification.md) | How each certified inequality is re-verified numerically, and what the peers cover |
| [docs/layout.md](docs/layout.md) | Where inputs, results and generated artefacts live, and which target writes them |
| [docs/time-bandwidth-bound.md](docs/time-bandwidth-bound.md) | The Kosut-Lidar-Rabitz comparison: the specialisation, its caveats and the results |
| [docs/margin-solvers-notes.md](docs/margin-solvers-notes.md) | The selectable margin solvers and what each certifies |

## Repository layout

```text
matlab/+qrobustness/     MATLAB toolbox
matlab/examples/         Paper case-study / synthesis drivers
matlab/tests/            MATLAB unit + consistency tests
python/src/qrobustness/  Python package (+ plotting)
python/tests/            pytest suite
data/controllers/        Frozen paper controller set
results/lipschitz-margin-matlab/    Lipschitz-margin results (MATLAB)
results/lipschitz-margin-python/    Lipschitz-margin results (Python)
results/lipschitz-margin-octave/    Lipschitz-margin results (Octave)
results/synth-*/         Regenerable synthesis (+ optional margins; local)
results/time-bandwidth-bound-*/ Kosut et al. time-bandwidth bound (experimental)
build/                   Scratch (gitignored)
CITATION.cff             Citation metadata (GitHub "Cite this repository")
.zenodo.json             Zenodo deposition metadata (applied on release)
CHANGELOG.md             Release history
docs/                    Theory, API, verification, layout, comparisons
scripts/                 Reproduction / compare / verify / bench
```

## Releases

Development happens on [qyber.black](https://qyber.black/spinnet/code-quantum-robustness-margins);
the repository is mirrored to
[GitHub](https://github.com/qyber-black/code-quantum-robustness-margins), and
tagged releases are archived on Zenodo, which mints a DOI per release from
[`.zenodo.json`](.zenodo.json). See [`CHANGELOG.md`](CHANGELOG.md) for the
release history.

Release checklist:

1. `make reproduce` (both papers recomputed and compared), `make check`
   (property checks) and `make test` pass.
2. `make paper-QRM-time-bandwidth` passes (supplementary; Python vs MATLAB cross-check).
3. Version agrees across `CITATION.cff`, `.zenodo.json`,
   `python/pyproject.toml`, `python/src/qrobustness/__init__.py` and
   and the paper repository's `refs.bib`; `CHANGELOG.md` has an entry with
   the release date.
4. `reuse lint` is clean and `cffconvert --validate` passes (both run in CI).
5. Tag the release; the GitHub mirror triggers the Zenodo deposition.

## Citation

If you use this code, please cite the software and the accompanying preprint.

**Software**

F. C. Langbein, S. P. O'Neil, S. Schirmer, C. A. Weidner, E. A. Jonckheere.
**Fidelity-Based Quantum Robustness Margins**. Version 1.1.0. Software, 2026.
<https://qyber.black/spinnet/code-quantum-robustness-margins>
(GitHub mirror: <https://github.com/qyber-black/code-quantum-robustness-margins>)

```bibtex
@software{langbein2026qrobustness,
  title   = {Fidelity-Based Quantum Robustness Margins},
  author  = {Langbein, F. C. and O'Neil, S. P. and Schirmer, S.
             and Weidner, C. A. and Jonckheere, E. A.},
  version = {1.1.0},
  year    = {2026},
  url     = {https://qyber.black/spinnet/code-quantum-robustness-margins},
}
```

**Paper (QRM)**

S. P. O'Neil, F. C. Langbein, C. A. Weidner, E. A. Jonckheere, and S. Schirmer.
**Fidelity-Based Robustness Margins for Finite-Time Quantum Control**.
Preprint, 2026. [arXiv:2608.03698](https://arxiv.org/abs/2608.03698)

```bibtex
@misc{oneil2026fidelity_margins,
  title         = {Fidelity-Based Robustness Margins for Finite-Time Quantum Control},
  author        = {O'Neil, S. P. and Langbein, F. C. and Weidner, C. A.
                   and Jonckheere, E. A. and Schirmer, S.},
  year          = {2026},
  eprint        = {2608.03698},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  url           = {https://arxiv.org/abs/2608.03698},
}
```

**Paper (xQRM)**

*Certified robustness regions under joint, time-dependent, and Lindbladian
uncertainty*. In preparation; no preprint identifier yet. Cite the software
and the QRM preprint until one exists.

Machine-readable entries are in [`CITATION.cff`](CITATION.cff) (also rendered by
GitHub's "Cite this repository") and [`.zenodo.json`](.zenodo.json) (used for
the Zenodo deposition on release).

## License

[AGPL-3.0-or-later](LICENSES/AGPL-3.0-or-later.txt). This repository is [REUSE](https://reuse.software/) compliant (`reuse lint`).

## Future Work

Not required for this toolbox release or the accompanying paper reproduction:

- Robust synthesis using the margin as an objective
- Multi-parameter structured uncertainty
- Open-system / dissipative dynamics

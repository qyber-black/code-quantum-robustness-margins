# Fidelity-Based Quantum Robustness Margins

> SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>\
> SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>\
> SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>\
> SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>\
> SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>\
>
> SPDX-License-Identifier: AGPL-3.0-or-later

Toolbox for **analysing** (and optionally **synthesising**) piecewise-constant coherent gate controllers under scalar structured Hamiltonian perturbations, accompanying the paper *Fidelity-Based Robustness Margins for Finite-Time Quantum Control*.

> **Scope:** Manuscript figures use the frozen paper controller set under `data/controllers/`. Synthesis writes new ensembles to `results/synth-*` and never overwrites that set. The paper reproduction path uses Algorithm 1 (Lipschitz + bisection); other margin solvers are optional -- see [docs/margin-solvers-notes.md](docs/margin-solvers-notes.md).

> **What is computed, and how accurately:** see [docs/theory.md](docs/theory.md) -- every reported quantity, the paper result it comes from, and its accuracy. Two quantities are not exact to roundoff and carry explicit error control: the margin `M` (a *lower* bound; ~5e-4 relative at the default `eta`, refinable to a requested precision via `margin_tol`) and the Kosut `w_dev`.

> **Provenance of the controllers.** The margin, sensitivity and figure results under `results/` are produced by this code. The **controller ensemble** in `data/controllers/`, however, is not: it comes from earlier optimization runs and is shipped frozen rather than regenerated here. The `optimize_controller` routine in this toolbox reproduces that synthesis *workflow*, but not necessarily the algorithm, parameters or constraints used for the original run -- and the MATLAB (`fminunc`, quasi-Newton) and Python (`L-BFGS-B`) implementations here already differ from each other. Re-synthesizing will therefore give a different, equally valid ensemble; it will not reproduce `data/controllers/` element by element.

## Features

- Gate propagator and normalized gate fidelity
- Lipschitz constant \(L_{\hat{H}}\) from the paper's sensitivity bound
- Differential sensitivity \(\zeta\), with the segment derivative in exact closed form (Gauss-Legendre quadrature selectable via `method='quadrature'`)
- Iterative one-dimensional robustness margin (Algorithm 1 default; selectable solvers)
- Dual MATLAB and Python APIs with matching paper-style plots
- Paper case-study reproduction via `make lipschitz-margin-matlab` / `make lipschitz-margin-python` / `make lipschitz-margin-octave`
- Supplementary comparison with the Kosut et al. fundamental bound (arXiv:2507.01215) -- experimental, see below
- Fidelity-maximising controller synthesis (GRAPE + quasi-Newton) via `make synth-matlab` / `make synth-python` / `make synth-octave`

## Requirements

- **Python** 3.10+ with NumPy, SciPy, pytest; matplotlib for plots (`qrobustness[plot]`). Reference implementation, and the source of the manuscript figures.
- **MATLAB** R2020b+ (peer; required by `make check-margins`). GRAPE synthesis (`optimize_controller`, `make synth-matlab`) additionally needs the Optimization Toolbox for `fminunc`.
- **Octave** 7+ (optional peer: `make lipschitz-margin-octave`, `make time-bandwidth-bound-octave`, `make synth-octave`). No Octave Forge packages are required: `fminunc` and `optimset` ship in core Octave, and the rank statistics are computed in-package.
- Git LFS for large `.mat` / `.png` artefacts

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
make test-matlab
```

## Reproduce paper results

Inputs: [`data/controllers/problem9_tf15_K32_quasi-newton/`](data/controllers/problem9_tf15_K32_quasi-newton/)

```bash
make lipschitz-margin-matlab           # -> results/lipschitz-margin-matlab/
make lipschitz-margin-python           # -> results/lipschitz-margin-python/
make lipschitz-margin-octave           # -> results/lipschitz-margin-octave/ (optional peer)
make compare-full           # Python vs MATLAB margin tables
make compare-octave         # MATLAB vs Octave margin tables
make sync-paper-qrm         # publish paper figures into ../paper-QRM/figures/
make export-golden          # refresh Python + MATLAB goldens
make verify-paper-matlab    # -> results/lipschitz-margin-matlab/verify_paper.md
make verify-paper-python    # -> results/lipschitz-margin-python/verify_paper.md
make verify-paper-octave    # -> results/lipschitz-margin-octave/verify_paper.md
make verify-paper           # Python + MATLAB verifiers
make check-margins            # lipschitz-margin-matlab/python + compare-full + verify-paper (release gate)
make synth-matlab           # new controllers -> results/synth-matlab/
make synth-python           # new controllers -> results/synth-python/
make analyse-synth-matlab   # margins for a synth set
make analyse-synth-python
make test
make clean                  # remove build/
make distclean              # remove build/ and venv
```

Python is the reference implementation and produces the manuscript figures. MATLAB and Octave are peers, held to it by `make compare-full`, `make compare-octave` and the golden fixtures in `data/reference/`. The release gate is `make check-margins` (Python and MATLAB); Octave is optional and not required by it.

## Comparison with the Kosut-Lidar-Rabitz time-bandwidth bound

A supplementary layer implements Theorem 1 of
[arXiv:2507.01215](https://arxiv.org/abs/2507.01215), specialised to the
closed-system scalar structured perturbation model used here, so that the
margin it implies can be placed alongside the certified Lipschitz margin. It is
experimental, sits outside the reproduction gate `make check-margins`, and no
claim in the paper depends on it. The structured margin is larger by a median
factor of 2.1 to 3.0 across the three perturbation structures. The implied
margin is the constant structured-parameter specialisation, with the nominal
error absorbed through the angular relation (see the caveats).

See [docs/time-bandwidth-bound.md](docs/time-bandwidth-bound.md) for the
specialisation, the caveats, the numerical accuracy of the interaction-picture
quantities, the full results and the `make time-bandwidth-bound*` targets.

## CI

GitLab CI ([`.gitlab-ci.yml`](.gitlab-ci.yml)) on push/MR runs **Python** `pytest` and an sdist/wheel build (`python -m build`). It pulls Git LFS so case-study `.mat` fixtures are available. MATLAB tests and the full paper gate (`make check-margins`) stay local. Pipelines: <https://qyber.black/lw1660/code-robustness-margins/-/pipelines>.

Each of [`results/lipschitz-margin-matlab/`](results/lipschitz-margin-matlab/), [`results/lipschitz-margin-python/`](results/lipschitz-margin-python/), and [`results/lipschitz-margin-octave/`](results/lipschitz-margin-octave/) contains:

| Artefact | Role |
|----------|------|
| `H0_all.png` ... `H2_all.png` | Fidelity error vs \(\delta_j\) |
| `robustness_margins_fid_err.png` | Margins vs controller index |
| `robustness_margins_sensitivity.png` | Margins vs \(\lvert\zeta\rvert\) |
| `correlations_0.999.tex` | Table I |
| `margins_table_0.999.csv` | Numeric table for compare-full |
| `verify_paper.md` | Consistency report for that tree |

See [docs/layout.md](docs/layout.md).

## Repository layout

```text
matlab/+qrobustness/     MATLAB toolbox
matlab/examples/         Paper case-study / synthesis drivers
matlab/tests/            MATLAB unit + consistency tests
python/src/qrobustness/  Python package (+ plotting)
python/tests/            pytest suite
data/controllers/        Frozen paper controller set
data/reference/          Golden fixtures
data/legacy/             Superseded artefacts (historical only)
results/lipschitz-margin-matlab/    Lipschitz-margin results (MATLAB)
results/lipschitz-margin-python/    Lipschitz-margin results (Python)
results/lipschitz-margin-octave/    Lipschitz-margin results (Octave)
results/synth-*/         Regenerable synthesis (+ optional margins; local)
results/time-bandwidth-bound-*/ Kosut et al. time-bandwidth bound (experimental)
build/                   Scratch (gitignored)
CITATION.cff             Citation metadata (GitHub "Cite this repository")
.zenodo.json             Zenodo deposition metadata (applied on release)
CHANGELOG.md             Release history
docs/                    API, layout, margin-solver notes
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

1. `make check-margins` (Python + MATLAB reproduction gate) and `make test` pass.
2. `make time-bandwidth-bound` passes (supplementary; Python vs MATLAB cross-check).
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
**Fidelity-Based Quantum Robustness Margins**. Version 1.0.2. Software, 2026.
<https://qyber.black/spinnet/code-quantum-robustness-margins>
(GitHub mirror: <https://github.com/qyber-black/code-quantum-robustness-margins>)

```bibtex
@software{langbein2026qrobustness,
  title   = {Fidelity-Based Quantum Robustness Margins},
  author  = {Langbein, F. C. and O'Neil, S. P. and Schirmer, S.
             and Weidner, C. A. and Jonckheere, E. A.},
  version = {1.0.2},
  year    = {2026},
  url     = {https://qyber.black/spinnet/code-quantum-robustness-margins},
}
```

**Paper**

S. P. O'Neil, F. C. Langbein, C. A. Weidner, E. A. Jonckheere, and S. Schirmer.
**Fidelity-Based Robustness Margins for Finite-Time Quantum Control**.
Preprint, 2026.

```bibtex
@misc{oneil2026fidelity_margins,
  title  = {Fidelity-Based Robustness Margins for Finite-Time Quantum Control},
  author = {O'Neil, S. P. and Langbein, F. C. and Weidner, C. A.
            and Jonckheere, E. A. and Schirmer, S.},
  year   = {2026},
  note   = {Preprint},
}
```

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

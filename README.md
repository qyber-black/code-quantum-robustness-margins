# Fidelity-Based Quantum Robustness Margins

> SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>\
> SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>\
> SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>\
> SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>\
> SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
>
> SPDX-License-Identifier: AGPL-3.0-or-later

Use this repository to analyse piecewise-constant coherent gate controllers under structured Hamiltonian and Lindbladian uncertainty, and to synthesise them when you need a new ensemble. It holds the code and the numerical results for two papers:

- **QRM** -- *Fidelity-Based Robustness Margins for Finite-Time Quantum Control*, [arXiv:2608.03698](https://arxiv.org/abs/2608.03698). The scalar-perturbation margin: the gate-fidelity sensitivity bound, the Lipschitz constant it induces, and Algorithm 1.
- **xQRM** -- *Certified robustness regions under joint, time-dependent, and Lindbladian uncertainty*. In preparation; not yet on arXiv. Joint static regions over several parameters, the Choi-Fubini-Study trajectory certificate, Lindblad rate margins, and the comparison bounds.

`PAPER` selects between them throughout the build: `make run-QRM`, `make verify-xQRM`, and so on.

> **Scope:** The paper results use the frozen controller sets under `data/controllers/`. Synthesis writes new ensembles to `results/synth-*` and does not overwrite `data/controllers/`. The theory, the experiments and the results are in the papers. This repository documents the code ([docs/api.md](docs/api.md)), the results files ([docs/layout.md](docs/layout.md)) and the tests ([docs/verification.md](docs/verification.md)).

> **Provenance of the controllers.** The margin, sensitivity and figure results under `results/` come from this code. The **controller ensemble** in `data/controllers/` comes from earlier optimisation runs and is shipped frozen. `optimize_controller` reproduces that synthesis *workflow*. It does not necessarily reproduce the algorithm, the parameters or the constraints of the original run. The MATLAB implementation (`fminunc`, quasi-Newton) and the Python implementation (`L-BFGS-B`) already differ. A new synthesis gives a different ensemble of the same kind. It does not reproduce `data/controllers/` element by element.

## Features

- Gate propagator and normalised gate fidelity
- Lipschitz constant \(L_{\hat{H}}\) from the paper's sensitivity bound
- Differential sensitivity \(\zeta\), with the segment derivative in exact closed form (Gauss-Legendre quadrature selectable via `method='quadrature'`)
- Iterative one-dimensional robustness margin (Algorithm 1 default; selectable solvers)
- MATLAB and Python APIs for the same certificate layers, with matching
  paper-style plots (see below for what the peer leaves out)
- Paper case-study reproduction via `make run-QRM-margins` in any engine
- Joint static regions, trajectory certificates, Lindblad rate margins and state-fidelity certificates (xQRM)
- Comparison bounds: the Kosut-Lidar-Rabitz time-bandwidth bound (arXiv:2507.01215) and the Berberich et al. gate-level bound, specialised to the structured model
- Fidelity-maximising controller synthesis (GRAPE + quasi-Newton), smoke-tested
  by `make test-synth`; run `scripts/run_synthesize_controllers.py` (or its
  MATLAB peer) directly to generate a full ensemble. Synthesis is a separate
  experiment: the frozen ensemble remains the sole paper input.

## Requirements

- **Python** 3.10+ with NumPy, SciPy, pytest; matplotlib for plots (`qrobustness[plot]`). Reference implementation, and the source of the manuscript figures.
- **MATLAB** R2020b+ (peer; required by `make test-parity ENGINE=matlab`). GRAPE synthesis (`optimize_controller`, `make test-synth ENGINE=matlab`) additionally needs the Optimization Toolbox for `fminunc`.
- **Octave** 7+ (peer: any target with `ENGINE=octave`). No Octave Forge packages are required: `fminunc` and `optimset` ship in core Octave, and the rank statistics are computed in-package.
- Git LFS for large `.mat` / `.png` artefacts
- The open-system certificates run with NumPy and SciPy (solver-free diamond norm); the SDP diamond norm `lindblad.diamond_norm` additionally needs `cvxpy` (`qrobustness[open]`), with the solver named by `lindblad.DEFAULT_SDP_SOLVER`.
- `make install` installs the pinned set in [`python/requirements-repro.txt`](python/requirements-repro.txt), the environment the published results were computed in. Pass `PINS=` to build against current releases instead.

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
make install              # the environment, from the pinned set
make test                 # lint, then unit tests, synthesis smoke and parity
                          # against Python for every engine (ENGINE=... for one);
                          # every stage runs, then one tally per engine
make run                  # every experiment of both papers (run-PAPER for one)
make verify               # recompute separately and compare, then the
                          # falsifiable property checks (verify-PAPER for one)
make sync-PAPER           # copy the generated artefacts into the paper repo
make sync                 # both papers
make clean                # build/; distclean also the environment,
                          # maintainer-clean also every generated result
```

`ENGINE` selects the implementation for `run` and `test`. It defaults to
`python`, the reference: `make run-xQRM-multiparam ENGINE=octave`.
`verify` and `sync` are Python only. A `run` or `test` target with no
implementation for the engine you chose fails. It does not fall back to
Python, so a pass means that engine did the work.
Add `-EXPNAME` to `run-` or `-ID` to `verify-` for one part.
`make help` lists those names. Every target runs all of its parts.

The paper repositories must be sibling directories; pass `PAPER_ROOT` or
`XPAPER_ROOT` to override their locations. Python is the reference
implementation. MATLAB and Octave are peers, held to it by cross-engine
comparisons against each other's committed result tables. `make help` lists
the `run` and `verify` part names and the cleanup targets.

The peer implements the certificates. It does not implement the whole library.
`+qrobustness` has the core margin machinery, the multiparameter and trajectory
gauges, the length-space layer, the Lindblad certificates, and both
comparison bounds (Kosut and Berberich). The cross-engine checks run on
those. Three parts are Python-only: the **adversarial search**
(`timevarying.adversarial_fidelity` and the upper witnesses built on it),
**GRAPE synthesis** (`synthesis.grape*`; `optimize_controller` is the peer
of the older workflow), and the **certificate harness**
(`qrobustness.verify`). Results that depend on those come from the Python
drivers alone.

The same name can mean a different routine in each engine.
`lindblad.diamond_norm` is the cvxpy SDP in Python. In MATLAB it is the
solver-free upper bound, whose Python peer is `diamond_norm_free`. MATLAB and
Octave have no portable SDP solver, so the solver-free method is the one
under the plain name.

## CI

GitLab CI ([`.gitlab-ci.yml`](.gitlab-ci.yml)) runs on push and on a merge request. It runs **Python** `pytest` and an sdist/wheel build (`python -m build`). It pulls Git LFS so the frozen controller ensembles in `data/controllers/` are present. MATLAB tests and the full paper gate (`make verify`) stay on your machine. Pipelines: <https://qyber.black/spinnet/code-quantum-robustness-margins/-/pipelines>.

The results files and their columns are listed in [docs/layout.md](docs/layout.md).

## Documentation

| Document | Covers |
|----------|--------|
| [docs/api.md](docs/api.md) | Modules and functions, inputs and outputs, result fields and statuses, accuracy classes, and the paper symbols they correspond to |
| [docs/layout.md](docs/layout.md) | Repository and results trees, every results file and its columns, publication to the papers |
| [docs/verification.md](docs/verification.md) | What `make test` and `make verify` check, the theorem checks, and MATLAB/Octave coverage |

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
results/time-bandwidth-bound-*/ Kosut et al. and Berberich et al. comparisons
results/*-python/        xQRM results (see docs/layout.md)
results/paper-xqrm/      Generated xQRM tables, figures, macros
build/                   Scratch (gitignored)
CITATION.cff             Citation metadata (GitHub "Cite this repository")
.zenodo.json             Zenodo deposition metadata (applied on release)
CHANGELOG.md             Release history
docs/                    API, layout and results files, verification
scripts/                 Reproduction / compare / verify / bench
```

## Releases

Development is on [qyber.black](https://qyber.black/spinnet/code-quantum-robustness-margins).
The repository is mirrored to
[GitHub](https://github.com/qyber-black/code-quantum-robustness-margins).
Tagged releases are archived on Zenodo, which mints a DOI for each release from
[`.zenodo.json`](.zenodo.json). The release history is in [`CHANGELOG.md`](CHANGELOG.md).

Release checklist:

1. `make verify` (both papers recomputed and compared, then the property
   checks) and `make test` pass.
2. Version agrees across `CITATION.cff`, `.zenodo.json`,
   `python/pyproject.toml`, `python/src/qrobustness/__init__.py` and
   the paper repository's `refs.bib`; `CHANGELOG.md` has an entry with
   the release date.
3. `reuse lint` is clean and `cffconvert --validate` passes (both run in CI).
4. Tag the release; the GitHub mirror triggers the Zenodo deposition.

## Citation

If you use this code, cite the software and the accompanying preprint.

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

Machine-readable entries are in [`CITATION.cff`](CITATION.cff), which GitHub
also renders as "Cite this repository", and in [`.zenodo.json`](.zenodo.json),
which Zenodo reads when it deposits a release.

## License

[AGPL-3.0-or-later](LICENSES/AGPL-3.0-or-later.txt). This repository is [REUSE](https://reuse.software/) compliant (`reuse lint`).

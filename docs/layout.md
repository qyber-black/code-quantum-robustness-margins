# Repository layout and results files

## Tree

| Path | Contents |
| :--- | :--- |
| `python/src/qrobustness/` | Python package (reference implementation) |
| `matlab/+qrobustness/` | MATLAB/Octave peer package; `matlab/examples/` drivers, `matlab/tests/` unit tests |
| `python/tests/` | Python unit tests |
| `scripts/` | experiment drivers (`run_*.py`), paper-artefact generators (`gen_paper_xqrm_*.py`), reproduction and consistency checks |
| `data/controllers/<id>/` | frozen inputs: problem definition and controller CSV (see each README) |
| `results/<method>-<engine>/` | committed results, one tree per method and engine |
| `results/paper-xqrm/` | generated xQRM tables, figures and macros, copied to the paper by `make sync-xQRM` |
| `results/synth-*/` | regenerable synthesised controller sets (not committed) |
| `build/` | scratch (smoke tests, scratch reproduction); not committed |

Result trees are named after the method. They are not named after the paper.
The paper-specific part of the build is the `PAPER_*` / `XPAPER_*` block at
the top of the `Makefile`. `scripts/_paper.py` records which driver writes
each results file, and the tests check that record against the Makefile.
Driver flags are defined once, in `scripts/_invocations.py`.

## Papers and publication

The papers are sibling repositories. By default they are `../paper-QRM`
(QRM, `PAPER_ROOT`) and `../paper-xQRM` (xQRM, `XPAPER_ROOT`), relative to
this repository. `make sync-QRM` copies the QRM figures. `make sync-xQRM`
regenerates the xQRM tables, figures and macros from `results/` and copies
them. Only the Python trees are published. The MATLAB and Octave trees are
compared with them in the parity stage of `make test`. `make verify`
reports a paper whose copies differ from the generated artefacts. It does
not change the paper.

## Results files (Python trees; columns by group)

`<FT>` is the threshold, `0.999` throughout. Margins are in units of the
perturbation parameter (relative errors for multiplicative structures).

**`lipschitz-margin-*` (QRM; also MATLAB and Octave)**

- `margins_table_<FT>.csv`: per controller `fid`, `err` (eps_0), and per
  structure `H0`, `H1`, `H2`: `M_*`, `Mm_*`, `Mp_*` (margin and the two
  rays), `zeta_*` (sensitivity).
- `focal_tests_<FT>.csv`: rank statistics of M_j against \|zeta_j\| (Spearman,
  Kendall, Holm-corrected p).
- `correlations_<FT>.tex`, `H*_all.png`, `robustness_margins_*.png`: QRM
  table and figures; `verify_paper.md`: consistency report of the tree.

**`time-bandwidth-bound-*` (Kosut comparison; also MATLAB and Octave)**

- `kosut_comparison_<FT>.csv` (additive absorption),
  `..._angular.csv` (angular absorption, the default),
  `..._angular_tv.csv` (trajectory class): per structure `M_*`, `KM_*`
  (M^K or M^{K,tri}_tv), `ratio_*`, `KTOb_*` (T Omega_bnd), `Kflb_*` (F_lb),
  `wunc_*`, `wavg_*`, `wdev_*`.
- `validity_<FT>.csv`, `validity_<FT>_tv.csv`, `validity_witness_<FT>.csv`:
  adversarial minimum fidelity at budgets m and 1.05 m on the control grid
  and its x4, x16 refinements (`Fmin_*`), `violated`, trajectory measures
  `omega_*`.
- `fs_validity_<FT>.csv`: the same attack at r_FS (`speed` = s).
- `budget_sweep_ctrl16_H1.csv`: minimum fidelity against budget m.
- `berberich_comparison_<FT>.csv`: `MB_ind` (M^B_tv), `MB_sys` (M^B),
  `gamma_*`, `magnus_ok`, `Fmin_attack`.
- `kosut_vs_lipschitz_*.png`: comparison figures.

**`multiparameter-margin-python`**

- `multiparam_<FT>.csv` (Lipschitz step) and `..._angular.csv` (angular
  step): `L_H*`, polytope radii `poly_r_H*`, inradii, `r0_*`, and per
  direction (`+e0` ... `diagppp`) `M_*`, `Mupper_*`, `nev_*` (evaluations,
  both rays of the direction).
- `joint_gauge_<FT>.csv`: diagonal and inradius gains of the joint and
  angular gauges over the cross-polytope, trajectory box gains.
- `tv_bracket_<FT>.csv`: per controller and structure `r0`, `r_fs`,
  `M_const`, `M_const_upper`, `m_adv`, `F_at_adv`, `adv_violated`, the
  constancy-gap interval `gap_lower`, `gap_upper` (bounds on M_const -
  M_tv), and `n_adversary_evals` (fidelity evaluations of the adversarial
  searches).
- `slice_ctrl1_<FT>.npz`: fidelity scan of a two-parameter slice.

**`single-qubit-python`**: `single_qubit_<FT>.csv`: per structure `M`,
`M_upper`, analytic `delta_star`, `r0`, `r_fs`, `m_adv`, `KM`, `KM_tv`.

**`cnot-python`**

- `cnot_margins_<FT>.csv`: per structure `M_*`, `r0_*`, `rfs_*`, `KM_*`,
  `KMtv_*`; inradii; dephasing `M_gamma`, `r0_gamma`, `gamma_star`.
- `duration_sweep_<FT>.csv`: the same against `tf`.
- `robust_vs_nominal_<FT>.csv`: per controller `kind` (nominal/robust),
  `seed`, angle budget `budget`, pulse areas `area_X*`, per structure `M_*`,
  `Mupper_*`, `rfs_*`, adversarial `madv_*` with re-evaluated fidelity
  `madvF_*`, toggling-frame integral norms `cancel_*`, `inradius_gauge`.

**`scaling-python`**: `scaling4q_margins_<FT>.csv`: as `cnot_margins` for
five structures, with evaluation and step counts.

**`lindblad-margin-python`**

- `open_margins_<FT>.csv` (dephasing), `open_amp_<FT>.csv` (amplitude
  damping and the joint diagonal), `open_coherent_<FT>.csv` (coherent
  structure through the open-system constant): F^pro_0, constants `L_*`,
  margins, one-step radii, bisected reference `*_star` (a safe endpoint of
  the final bracket), ratios, evaluations.
- `open_threshold_sweep.csv`, `open_threshold_cohort.csv`: ratios against
  F_T and the controllers eligible at each threshold.
- `mixed_ctrl1.npz`: mixed coherent-dissipative region data.
- `dnorm_certificates.csv`: per generator the solver optimum `raw`, the
  verified `certified` value, `gap`, `rel_inflation`, `feas_shift`, solver
  status and name, and for the closed-form families the deviation from 2n.

**`verification-python`**: `verification_<FT>.csv`: per check `n` probes,
`min_slack`, `tol`, `passed`, `max_fraction_of_bound`.

**`state-examples-python`** (xQRM state appendix)

- `ghz_detuning_<FT>.csv`: per n, state and Choi speeds of the hold, the
  one-step state radius and its analytic crossing, the fidelity there, the
  gate radius and gate crossing, and the margin with detuning during
  preparation and hold (`M_all`, `M_upper_all`, `n_evals_all`).
- `tfim_preparation_<FT>.csv`: per chain length `L` and field `h0`, the gap,
  C_prep by finite differences, sigma/gamma, half-spread/gamma, the one-step
  radius, and per direction the margin, evaluations and resolved crossing.
- `ghz_dephasing_<FT>.csv`: D/2, one-step radius, margin, analytic crossing.
- `state_variance_<FT>.csv`: integrated sigma_t against C^st and the Choi
  speed per controller, structure and initial state.
- `closed_limit.csv`: exact spread, verified closed form, SDP and
  solver-free diamond norms of the chain structures.

**`algorithm-tests-python`**

- `crosstalk_<FT>.csv`: per controller and coupling `kappa`, the Gram
  correlation, radii and gains of the joint and angular gauges along the
  cancelling diagonal, the smallest fidelity on the certified boundaries,
  and the iterated diagonal margin.
- `rays_<FT>.csv`: per test ray the first unsafe point, `M`, `M_upper`,
  `reason`, `n_unresolved`, `n_evals`, and checks `prefix_ok`,
  `witness_ok`.

**`bracket-audit-python`**

- `brackets_<FT>.csv`: per controller, direction, rule (`angular` or the
  scalar `precursor`) and evaluation band: `M`, `M_upper`, `rel_width`,
  `reason`, `n_unresolved`, `n_evals` (the `+` ray).
- `timing_<FT>.csv`, `environment.json`: wall-clock medians and
  interquartile ranges (`t_*`) of preprocessing, one fidelity evaluation and
  one directional run (both rays), and the machine and library versions.
  The `t_*` columns are the only ones the reproduction check does not
  compare.

**`paper-xqrm`**: `tables/*.tex`, `figures/*.pdf`, `macros.tex`.

## Synthesis output (`results/synth-*`)

`problem9.mat` (problem copy), `controllers.csv` (same schema as the
frozen set), `meta.json` (seed, number of runs, method, error statistics).

## Regression reference

The committed `results/` trees are the reference. A regeneration that
changes a number shows in `git status`. `make verify` recomputes into a
separate tree and compares the numbers. Generated figures are
byte-deterministic. The SDP solver is named
(`lindblad.DEFAULT_SDP_SOLVER`), so the open-system numbers do not follow
whichever solver happens to be installed. The environment is pinned in
`python/requirements-repro.txt`. `make install` installs that pin.
`PINS=` installs current releases instead.

Sources and documentation are ASCII. Mathematics is written in LaTeX-like
notation.

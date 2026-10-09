# Controller set: problem9_tf15_K32_quasi-newton

These are the frozen inputs of the three-qubit Heisenberg-chain case study
in the QRM paper. The xQRM paper uses the same ensemble. This code does not
regenerate them.

| File | Role |
|------|------|
| `problem9.mat` | drift and control Hamiltonians H_0, H_1, H_2 and target gate U_f |
| `controllers.csv` | optimised piecewise-constant controllers (t_f = 15, tau = 32): `fid`, `error`, `u1_*`, `u2_*` |

Paper filter: nominal error eps_0 <= 1e-4 (`load_controllers(..., max_error=1e-4)`);
threshold F_T = 0.999.

Results: `results/lipschitz-margin-{python,matlab,octave}/` and the xQRM
trees listed in `docs/layout.md`.

Leave this directory as it is. New controllers for the same problem come
from `scripts/run_synthesize_controllers.py`, or from the MATLAB/Octave
driver `matlab/examples/run_synthesize_controllers.m`, and are written to
`results/synth-*`.

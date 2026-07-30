# Controller set: problem9_tf15_K32_quasi-newton

Non-reproducible inputs for the three-qubit Heisenberg gate-control case study in the paper *Fidelity-Based Robustness Margins for Finite-Time Quantum Control*.

| File | Role |
|------|------|
| `problem9.mat` | Drift/interaction Hamiltonians \(H_0,H_1,H_2\) and target gate \(U_f\) |
| `controllers.csv` | 100 optimised piecewise-constant controllers (\(t_f=15\), \(\tau=32\)) |

**Paper filter:** keep controllers with nominal fidelity error \(\varepsilon_0\le 10^{-4}\) (61 controllers). Robustness threshold \(\mathcal{F}_T=0.999\).

**Results:** [`results/lipschitz-margin-matlab/`](../../../results/lipschitz-margin-matlab/) and [`results/lipschitz-margin-python/`](../../../results/lipschitz-margin-python/).

**Do not overwrite this directory.** To generate new controllers for the same problem, use `make synth-matlab` / `make synth-python` -> `results/synth-*`.

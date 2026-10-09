# Controller set: cnot_tf4_K20_lbfgs

This is the two-qubit CNOT ensemble of the xQRM paper. It was synthesised by
`scripts/run_cnot_case_study.py` with `qrobustness.synthesis.grape_ensemble`
(L-BFGS-B GRAPE, seeds 20260801 + i, 60 attempts, eps_0 <= 1e-4, the first 50
kept).

| File | Role |
|------|------|
| `problem.npz` | `H0` = pi Z_1 Z_2 + 0.1 pi (Z_1 - Z_2), controls `X1`, `X2`, target `Uf` (CNOT), `tf` = 4, `tau` = 20 |
| `controllers.csv` | `seed`, `fid`, `err`, `u1_*`, `u2_*` |

Results: `results/cnot-python/`.

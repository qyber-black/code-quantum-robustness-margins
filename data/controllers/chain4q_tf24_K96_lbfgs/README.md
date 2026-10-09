# Controller set: chain4q_tf24_K96_lbfgs

This is the four-qubit Ising-chain ensemble of the xQRM paper. It was
synthesised by `scripts/run_scaling_example.py` with
`qrobustness.synthesis.grape_ensemble` (L-BFGS-B GRAPE, at most 3000
iterations, seeds 20260802 + i, 20 attempts, eps_0 <= 1e-4, the first 10
kept).

| File | Role |
|------|------|
| `problem.npz` | `H0` = 2 pi J sum_q Z_q Z_{q+1} + 2 pi sum_q d_q Z_q (J = 0.5, d = (0.11, -0.07, 0.05, -0.13)), controls `X1` ... `X4`, Haar-random target `Uf` (seed 20260801), `tf` = 24, `tau` = 96 |
| `controllers.csv` | `seed`, `fid`, `err`, controls per interval |

Results: `results/scaling-python/`.

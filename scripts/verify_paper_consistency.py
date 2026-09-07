#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Falsifiable checks against a paper results tree (Python peer of the MATLAB verifier).

Writes results/<results_id>/verify_paper.md
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from qrobustness import (
    dH_structure,
    differential_sensitivity,
    gate_fidelity,
    iterative_margin,
    lipschitz_constant,
    load_controllers,
    load_problem,
    make_fidelity_fn,
    perturbed_hamiltonians,
    propagator,
    structure_constant,
)
from scipy.stats import pearsonr, spearmanr

from _drivers import DEFAULT_ETA, DEFAULT_FT, DEFAULT_MAX_ERROR, VIOLATION_TOL

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
#: Safe-radius continuation step, matched to the other drivers.
ETA = DEFAULT_ETA
FT = DEFAULT_FT
STRUCTURES = ("H0", "H1", "H2")

#: Agreement demanded of quantities that should be equal up to rounding
#: (a recomputation, a legacy table, a closed form) and of ones that are
#: only equal up to an iteration's own tolerance. Named as the library's
#: verifier names them, since these are the same two kinds of claim.
TOL_EXACT = 1e-12
TOL_TIGHT = 1e-10

#: Bracket refinement, matched to the drivers. Check [11] recomputes the
#: table's margin and demands equality, so it has to certify to the same
#: bracket the driver does; without it the recomputation stops at the last
#: safe continuation step, ~4e-4 relative below the table.
MARGIN_TOL = 1e-8

#: Central-difference step for the finite-difference check of zeta.
FD_STEP = 1e-7

#: Variables in Table I; must match the vars_ list built by
#: run_lipschitz_margin_case_study.write_correlation_tex.
N_CORR_VARS = 7

NEED = (
    "H0_all.png",
    "H1_all.png",
    "H2_all.png",
    "robustness_margins_fid_err.png",
    "robustness_margins_sensitivity.png",
    "correlations_0.999.tex",
    "margins_table_0.999.csv",
)


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.ok = True

    def log(self, msg: str = "") -> None:
        self.lines.append(msg)
        print(msg, end="" if msg.endswith("\n") else "\n")

    def check(self, cond: bool) -> None:
        if cond:
            self.log("  PASS")
        else:
            self.log("  FAIL")
            self.ok = False


def eye_pow(n: int) -> np.ndarray:
    """Identity on ``n`` qubits, or the 1x1 identity for ``n <= 0``."""
    if n <= 0:
        return np.array([[1.0]])
    return np.eye(2**n)


def heisenberg_refs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three-spin Heisenberg model the paper's problem file should hold."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    def op(A: np.ndarray, ell: int) -> np.ndarray:
        return np.kron(np.kron(eye_pow(ell - 1), A), eye_pow(3 - ell))

    H0 = 0.5 * (
        op(sx, 1) @ op(sx, 2)
        + op(sy, 1) @ op(sy, 2)
        + op(sz, 1) @ op(sz, 2)
        + op(sx, 2) @ op(sx, 3)
        + op(sy, 2) @ op(sy, 3)
        + op(sz, 2) @ op(sz, 3)
    )
    H1 = 2 * op(sx, 1)
    H2 = 2 * op(sy, 1)
    return H0, H1, H2


def parse_corr_tex(path: Path) -> np.ndarray:
    """Read Table I back out of the generated tex."""
    rows = []
    for line in path.read_text().splitlines():
        nums = re.findall(r"\$([+-]?\d+\.\d+)\$", line)
        if len(nums) >= N_CORR_VARS:
            rows.append([float(x) for x in nums[:N_CORR_VARS]])
    C = np.asarray(rows, dtype=float)
    if C.shape != (N_CORR_VARS, N_CORR_VARS):
        raise ValueError(f"parse_corr_tex: size {C.shape}")
    return C


def load_margins_csv(path: Path) -> dict[str, np.ndarray]:
    """Numeric columns of a margins table, keyed by column name."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    cols = reader.fieldnames or []
    out: dict[str, np.ndarray] = {}
    for c in cols:
        try:
            out[c] = np.array([float(r[c]) for r in rows], dtype=float)
        except ValueError:
            continue
    return out


def corr_matrix(X: np.ndarray) -> np.ndarray:
    """Upper triangle Pearson r, lower triangle Spearman rho (Table I)."""
    n = X.shape[1]
    C = np.eye(n)
    for i in range(n):
        for j in range(n):
            if j > i:
                C[i, j] = pearsonr(X[:, i], X[:, j]).statistic
            elif j < i:
                C[i, j] = spearmanr(X[:, i], X[:, j]).statistic
    return C


def main() -> int:
    """Run every falsifiable check and write the report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-id", default="lipschitz-margin-python")
    args = ap.parse_args()
    results_id = args.results_id
    results_dir = ROOT / "results" / results_id
    results_dir.mkdir(parents=True, exist_ok=True)
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    report_txt = build / f"verify_paper_{results_id}.txt"
    report_md = results_dir / "verify_paper.md"

    R = Report()
    R.log(f"=== Paper consistency verification ({results_id}) ===")
    R.log(f"root={ROOT}")
    R.log(f"results={results_dir}")
    R.log()

    problem = load_problem(CTRL / "problem9.mat")
    ctrls = load_controllers(CTRL / "controllers.csv", DEFAULT_MAX_ERROR)
    nC = len(ctrls)
    R.log(f"[1] Controllers with eps0<={DEFAULT_MAX_ERROR:g}: {nC} (paper: 61)")
    R.check(nC == 61)

    tf = ctrls[0]["tf"]
    tau = ctrls[0]["tau"]
    dt = tf / tau
    R.log(f"[1b] tf={tf:g} tau={tau} (paper: 15, 32)")
    R.check(abs(tf - 15) < TOL_EXACT and tau == 32)

    errs = np.array([c["error"] for c in ctrls])
    R.log(f"[1c] eps0 in [{errs.min():.3e}, {errs.max():.3e}]")
    R.check(np.all(errs <= DEFAULT_MAX_ERROR + 1e-15))

    N = problem["dim"]
    H0_ref, H1_ref, H2_ref = heisenberg_refs()
    dH = [
        np.linalg.norm(problem["H0"] - H0_ref, "fro"),
        np.linalg.norm(problem["H1"] - H1_ref, "fro"),
        np.linalg.norm(problem["H2"] - H2_ref, "fro"),
    ]
    R.log(f"[2] ||Hj-paper||_F = {[float(x) for x in dH]}, N={N}")
    R.check(max(dH) < TOL_EXACT and N == 8)

    max_fid_mismatch = 0.0
    for c in ctrls:
        H_list = perturbed_hamiltonians(
            problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], "H0", 0.0
        )
        F = gate_fidelity(propagator(H_list, dt), problem["Uf"])
        max_fid_mismatch = max(max_fid_mismatch, abs((1.0 - F) - c["error"]))
    R.log(f"[3] max |eps_recomputed - eps_csv| = {max_fid_mismatch:.3e}")
    R.check(max_fid_mismatch < TOL_TIGHT)

    csv_path = results_dir / "margins_table_0.999.csv"
    T: dict[str, np.ndarray] | None = None
    if not csv_path.is_file():
        # The results id is the make target name.
        R.log(f"[4] MISSING {csv_path} (run make {results_id})")
        R.check(False)
    else:
        T = load_margins_csv(csv_path)
        # The margin table is compared against nothing here on purpose. A
        # legacy .mat used to serve as the reference, but its margin array
        # was regenerated from this very table, so the check compared the
        # results with a re-exported copy of themselves; git does that
        # comparison on the committed CSV, exactly and without a tolerance.
        # What remains worth asserting is the relation between the columns,
        # which check [11] does on a recomputation.
        for tag in STRUCTURES:
            Mm, Mp, M = T[f"Mm_{tag}"], T[f"Mp_{tag}"], T[f"M_{tag}"]
            dM = float(np.max(np.abs(M - np.minimum(np.abs(Mm), np.abs(Mp)))))
            R.log(f"[4] {tag} max|M - min(|Mm|,|Mp|)| = {dM:.3e}")
            R.check(dM < TOL_EXACT)

    if T is not None:
        spot = sorted(set(list(range(5)) + list(range(nC - 3, nC))))
        n_bad = 0
        max_overshoot = 0.0
        min_slack = np.inf
        for n in spot:
            c = ctrls[n]
            for tag in STRUCTURES:
                M = float(T[f"M_{tag}"][n])
                fid_fn = make_fidelity_fn(
                    problem["H0"],
                    problem["H1"],
                    problem["H2"],
                    c["u1"],
                    c["u2"],
                    problem["Uf"],
                    dt,
                    tag,
                )
                for sgn in (-1.0, 1.0):
                    F = fid_fn(sgn * M)
                    slack = F - FT
                    min_slack = min(min_slack, slack)
                    if F < FT - VIOLATION_TOL:
                        n_bad += 1
                        max_overshoot = max(max_overshoot, FT - F)
        R.log(
            f"[5] F(+/-M) spot: min(F-FT)={min_slack:.3e} bad={n_bad} "
            f"max_overshoot={max_overshoot:.3e}"
        )
        R.check(n_bad == 0)
    else:
        R.log("[5] SKIP (no margins table)")

    if T is not None:
        max_rel_fd = 0.0
        max_table_dz = 0.0
        for n in range(5):
            c = ctrls[n]
            for tag in STRUCTURES:
                H_list = perturbed_hamiltonians(
                    problem["H0"],
                    problem["H1"],
                    problem["H2"],
                    c["u1"],
                    c["u2"],
                    tag,
                    0.0,
                )
                dHs = dH_structure(
                    problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag
                )
                z = differential_sensitivity(H_list, dHs, dt, problem["Uf"], 32)
                fid_fn = make_fidelity_fn(
                    problem["H0"],
                    problem["H1"],
                    problem["H2"],
                    c["u1"],
                    c["u2"],
                    problem["Uf"],
                    dt,
                    tag,
                )
                h = FD_STEP
                z_fd = (fid_fn(h) - fid_fn(-h)) / (2 * h)
                max_rel_fd = max(max_rel_fd, abs(z - z_fd) / max(TOL_EXACT, abs(z_fd)))
                max_table_dz = max(max_table_dz, abs(T[f"zeta_{tag}"][n] - z))
        R.log(
            f"[6] max rel|zeta-FD|(h={FD_STEP:g})={max_rel_fd:.3e}  "
            f"max|table-recompute|={max_table_dz:.3e}"
        )
        # FD vs analytic \zeta is a soft spot-check (engine-dependent); table recompute is hard.
        if max_rel_fd >= 2e-4:
            R.log("  NOTE soft FD check exceeded 2e-4 (not failing)")
        R.check(max_table_dz < TOL_TIGHT)
    else:
        R.log("[6] SKIP")

    # The generated table, not the manuscript. A paper repository is synced
    # from these results and can be behind them, so comparing against it
    # would fail on a stale checkout rather than on a wrong number. Check [8]
    # is what makes the table falsifiable: it recomputes the matrix from the
    # results CSV and demands the printed table agree exactly.
    corr_tex = results_dir / "correlations_0.999.tex"
    C_code = None
    if not corr_tex.is_file():
        R.log(f"[7] MISSING {corr_tex}")
        R.check(False)
    else:
        C_code = parse_corr_tex(corr_tex)
        R.log(f"[7] {corr_tex.name}: {C_code.shape[0]}x{C_code.shape[1]} parsed")

    if T is not None and C_code is not None:
        # Table I correlates against the sensitivity magnitudes: min(M-, M+)
        # is invariant under reversal of the parameter coordinate while zeta
        # changes sign, so |zeta| is the orientation-invariant comparator.
        vars_ = ["err", "M_H0", "M_H1", "M_H2", "zeta_H0", "zeta_H1", "zeta_H2"]
        absolute = {"zeta_H0", "zeta_H1", "zeta_H2"}
        X = np.column_stack([np.abs(T[v]) if v in absolute else T[v] for v in vars_])
        C_re = corr_matrix(X)
        dCre = float(np.max(np.abs(np.round(C_re, 2) - C_code)))
        R.log(f"[8] max |round(recomputed,2) - correlations.tex| = {dCre:.3e}")
        R.check(dCre == 0.0)
    else:
        R.log("[8] SKIP")

    c0 = ctrls[0]
    C0 = structure_constant("drift", problem["H0"], dt, c0["tau"])
    C1 = structure_constant("control", problem["H1"], dt, c0["tau"], c0["u1"])
    B_T = np.sqrt((1 - FT**2) / N)
    L0 = lipschitz_constant(FT, N, C0)
    C0_ref = tf * np.linalg.norm(problem["H0"], "fro")
    R.log(f"[9] B_T={B_T:.6g} C0={C0:.6g} L0={L0:.6g}; C0_ref=tf*||H0||F={C0_ref:.6g}")
    R.check(abs(C0 - C0_ref) < TOL_EXACT)
    R.check(
        abs(
            C1
            - dt
            * np.linalg.norm(c0["u1"].ravel(), 1)
            * np.linalg.norm(problem["H1"], "fro")
        )
        < TOL_EXACT
    )

    ok_pub = True
    for name in NEED:
        f = results_dir / name
        if not f.is_file():
            R.log(f"[10] MISSING {f}")
            ok_pub = False
    R.log(f"[10] {results_id} deliverables")
    R.check(ok_pub)

    if T is not None:
        c = ctrls[0]
        C = structure_constant("drift", problem["H0"], dt, c["tau"])
        L = lipschitz_constant(FT, N, C)
        fid_fn = make_fidelity_fn(
            problem["H0"],
            problem["H1"],
            problem["H2"],
            c["u1"],
            c["u2"],
            problem["Uf"],
            dt,
            "H0",
        )
        res = iterative_margin(fid_fn, L, FT, mu0=0.0, eta=ETA, margin_tol=MARGIN_TOL)
        R.log(
            f"[11] ctrl1 H0: M-={res.M_minus:.6g} M+={res.M_plus:.6g} M={res.M:.6g} "
            f"tableM={T['M_H0'][0]:.6g} conv=[{int(res.converged_minus)} {int(res.converged_plus)}]"
        )
        R.check(abs(res.M - T["M_H0"][0]) < TOL_EXACT)
        R.check(res.M == min(res.M_minus, res.M_plus))
        R.check(res.mu_minus < 0 and res.mu_plus > 0)
    else:
        R.log("[11] SKIP")

    verdict = "PASS" if R.ok else "FAIL"
    R.log()
    R.log(f"=== VERDICT: {verdict} ===")

    body = "\n".join(R.lines) + "\n"
    report_txt.write_text(body)
    report_md.write_text(
        f"# Paper consistency verification ({results_id})\n\n"
        f"Generated by `scripts/verify_paper_consistency.py`.\n\n"
        f"```\n{body}```\n"
    )
    print(f"Wrote {report_txt} and {report_md}")
    return 0 if R.ok else 1


if __name__ == "__main__":
    sys.exit(main())

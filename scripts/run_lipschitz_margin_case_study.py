#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The single-parameter case study on the main ensemble.

Computes, per controller and structure H0, H1, H2, the iterated margin M
(and its arms M^- and M^+) and the differential sensitivity zeta_j, the
correlation table of nominal error, M_j and |zeta_j|, and rank tests of M_j
against |zeta_j| with a Holm correction.

Options: --FT, --out, --sweep (fidelity-versus-delta sweeps and H*_all.png;
the Makefile always passes it), --no-plots, --controller-dir (problem9.mat
and controllers.csv), --max-error.

Writes results/lipschitz-margin-python/ (or --out):
    margins_table_<FT>.csv (also build/margins_table_<FT>_python.csv):
        controller, fid, err: instance and nominal fidelity / error.
        M_<s>, Mm_<s>, Mp_<s>, zeta_<s>: M, M^-, M^+, zeta for s in H0, H1, H2.
    correlations_<FT>.tex: tabular, Pearson above and Spearman below the
        diagonal, over err, M_j and |zeta_j|.
    focal_tests_<FT>.csv: comparison, n, spearman_rho, p_two_sided, p_holm,
        kendall_tau_b, kendall_p_two_sided, kendall_p_holm.
    robustness_margins_fid_err.png, robustness_margins_sensitivity.png, and
        with --sweep H0_all.png, H1_all.png, H2_all.png.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr
from qrobustness import (
    dH_structure,
    differential_sensitivity,
    fidelity_vs_delta,
    iterative_margin,
    lipschitz_constant,
    load_controllers,
    load_problem,
    make_fidelity_fn,
    perturbed_hamiltonians,
    structure_constant,
)
from qrobustness.plotting import (
    plot_fidelity_error_sweeps,
    plot_margins_vs_index,
    plot_margins_vs_sensitivity,
)

from _drivers import (
    DEFAULT_ETA,
    DEFAULT_MAX_ERROR,
    ZETA_N_QUAD,
    base_parser,
)

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "data/controllers/problem9_tf15_K32_quasi-newton"
OUT_DIR = ROOT / "results/lipschitz-margin-python"
BUILD = ROOT / "build"

ETA = DEFAULT_ETA
K_MAX = 10000
AUDIT_MARGIN_TOL = 1e-10
STRUCTURES = ("H0", "H1", "H2")
#: Fidelity-vs-delta sweep: slightly past the wider margin arm, with a
#: fallback span when the margin is zero.
SWEEP_SPAN_FACTOR = 1.05
SWEEP_SPAN_FALLBACK = 1e-3
SWEEP_POINTS = 401

XLIMS = {
    "H0": (-8e-3, 8e-3),
    "H1": (-2e-2, 2e-2),
    "H2": (-2e-2, 2e-2),
}


#: The three M_j versus |zeta_j| comparisons tested in focal_tests.
FOCAL_PAIRS = tuple((f"M_H{j}", f"zeta_H{j}") for j in range(3))


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (monotonic, capped at 1)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def focal_tests(rows: list[dict]) -> list[dict]:
    """Spearman rho and Kendall tau_b, with two-sided and Holm-adjusted
    p-values, for each pair in FOCAL_PAIRS."""
    out = []
    for j, (mkey, zkey) in enumerate(FOCAL_PAIRS):
        M = np.array([r[mkey] for r in rows], dtype=float)
        Z = np.abs(np.array([r[zkey] for r in rows], dtype=float))
        rho = spearmanr(M, Z)
        tau, tau_p = kendalltau(M, Z, variant="b")
        out.append(
            {
                "j": j,
                "rho": float(rho.statistic),
                "p": float(rho.pvalue),
                "tau_b": float(tau),
                "tau_p": float(tau_p),
            }
        )
    for rec, padj in zip(out, holm([r["p"] for r in out]), strict=True):
        rec["p_holm"] = padj
    for rec, padj in zip(out, holm([r["tau_p"] for r in out]), strict=True):
        rec["tau_p_holm"] = padj
    return out


def write_focal_tests(rows: list[dict], path: Path) -> list[dict]:
    """Write focal_tests(rows) to path as CSV and return the records."""
    recs = focal_tests(rows)
    # LF line endings, as in every other CSV in the tree.
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            [
                "comparison",
                "n",
                "spearman_rho",
                "p_two_sided",
                "p_holm",
                "kendall_tau_b",
                "kendall_p_two_sided",
                "kendall_p_holm",
            ]
        )
        for r in recs:
            w.writerow(
                [
                    f"M_H{r['j']}_vs_abs_zeta_H{r['j']}",
                    len(rows),
                    f"{r['rho']:.6f}",
                    f"{r['p']:.6e}",
                    f"{r['p_holm']:.6e}",
                    f"{r['tau_b']:.6f}",
                    f"{r['tau_p']:.6e}",
                    f"{r['tau_p_holm']:.6e}",
                ]
            )
    return recs


def write_correlation_tex(rows: list[dict], path: Path) -> None:
    """Write the correlation tabular: Pearson above the diagonal, Spearman
    rho below it."""
    # |zeta_j|, not zeta_j: M_j is invariant under reversing the parameter
    # sign while zeta_j changes sign.
    vars_ = ["err", "M_H0", "M_H1", "M_H2", "zeta_H0", "zeta_H1", "zeta_H2"]
    absolute = {"zeta_H0", "zeta_H1", "zeta_H2"}
    labels = [
        r"$\varepsilon_0$",
        "$M_0$",
        "$M_1$",
        "$M_2$",
        r"$|\zeta_0|$",
        r"$|\zeta_1|$",
        r"$|\zeta_2|$",
    ]
    X = np.column_stack(
        [
            np.abs(np.array([r[v] for r in rows], dtype=float))
            if v in absolute
            else np.array([r[v] for r in rows], dtype=float)
            for v in vars_
        ]
    )
    # Pearson (upper) and Spearman rho (lower).
    P = np.corrcoef(X, rowvar=False)

    n_vars = len(vars_)
    S = np.eye(n_vars)
    for i in range(n_vars):
        for j in range(i):
            S[i, j] = S[j, i] = spearmanr(X[:, i], X[:, j]).statistic

    lines = [
        "% Auto-generated: upper Pearson r, lower Spearman rho; |zeta| (see above)",
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        " & " + " & ".join(labels) + r" \\",
        r"\midrule",
    ]
    for i in range(n_vars):
        cells = [labels[i]]
        for j in range(n_vars):
            if i == j:
                v = 1.0
            elif j > i:
                v = P[i, j]
            else:
                v = S[i, j]
            cells.append(f"${v:.2f}$")
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\midrule",
        r"\multicolumn{8}{@{}l@{}}{\footnotesize Cells above this table's main diagonal (1.00): Pearson $r$.} \\",
        r"\multicolumn{8}{@{}l@{}}{\footnotesize Cells below this table's main diagonal (1.00): Spearman $\rho$.} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n")


def margin_audit_record(controller: int, structure: str, result) -> dict:
    """Keep the original continuation endpoints separate from refinement."""
    minus, plus = abs(result.mu_minus), abs(result.mu_plus)
    return {
        "controller": controller,
        "structure": structure,
        "M_minus": minus,
        "M_plus": plus,
        "M": min(minus, plus),
        "M_refined_minus": result.M_minus,
        "M_refined_plus": result.M_plus,
        "M_refined": result.M,
        "M_upper_minus": result.M_upper_minus,
        "M_upper_plus": result.M_upper_plus,
        "M_upper": result.M_upper,
        "status_minus": result.status_minus,
        "status_plus": result.status_plus,
        "safeguard_minus": int(result.safeguard_minus),
        "safeguard_plus": int(result.safeguard_plus),
        "reason_minus": result.reason_minus,
        "reason_plus": result.reason_plus,
        "n_evals_continuation": result.n_evals_continuation,
        "n_evals_refinement": result.n_evals_refinement,
        "n_evals": result.n_evals,
    }


def write_margin_audit(rows: list[dict], out: Path, ft: float) -> None:
    """Write measured costs, stopping outcomes and manuscript macros."""
    with (out / f"margin_audit_{ft:g}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    widths = [(r["M_upper"] - r["M"]) / r["M"] for r in rows]
    width = max(widths)
    exponent = int(np.floor(np.log10(width)))
    mantissa = np.ceil(width / 10.0**exponent * 10) / 10
    values = {
        "Cases": len(rows),
        "Rays": 2 * len(rows),
        "KMax": K_MAX,
        "GuardRays": sum(r["safeguard_" + s] for r in rows for s in ("minus", "plus")),
        "EtaBandRays": sum(
            r["status_" + s] == "eta_band" for r in rows for s in ("minus", "plus")
        ),
        "BracketedRays": sum(
            r["reason_" + s] == "bracketed" for r in rows for s in ("minus", "plus")
        ),
    }
    for name, field in (
        ("Continuation", "n_evals_continuation"),
        ("Refinement", "n_evals_refinement"),
    ):
        data = [r[field] for r in rows]
        values.update(
            {
                name + "Min": min(data),
                name + "Median": np.median(data),
                name + "Max": max(data),
            }
        )
    lines = [
        "% Auto-generated from margin_audit CSV; evaluations are per two-direction case."
    ]
    lines += [
        rf"\newcommand{{\QRMAudit{name}}}{{{value:g}}}"
        for name, value in values.items()
    ]
    lines.append(
        rf"\newcommand{{\QRMAuditControllers}}{{{len({r['controller'] for r in rows})}}}"
    )
    for structure, name in (
        ("H0", "DriftSpread"),
        ("H1", "ControlOneSpread"),
        ("H2", "ControlTwoSpread"),
    ):
        data = [r["M"] for r in rows if r["structure"] == structure]
        lines.append(rf"\newcommand{{\QRMAudit{name}}}{{{max(data) / min(data):.1f}}}")
    coefficient, power = f"{AUDIT_MARGIN_TOL:.0e}".split("e")
    lines += [
        rf"\newcommand{{\QRMAuditRelativeWidth}}{{{mantissa:.1f}\times10^{{{exponent}}}}}",
        rf"\newcommand{{\QRMAuditTolerance}}{{{coefficient}\times10^{{{int(power)}}}}}",
    ]
    (out / f"margin_audit_{ft:g}.tex").write_text("\n".join(lines) + "\n")


def plot_results(rows: list[dict], sweeps: dict, ft: float, out: Path) -> None:
    """Render figures without changing the numerical results."""
    plot_margins_vs_index(
        [r["err"] for r in rows],
        [r["M_H0"] for r in rows],
        [r["M_H1"] for r in rows],
        [r["M_H2"] for r in rows],
        out_path=out / "robustness_margins_fid_err.png",
    )
    plot_margins_vs_sensitivity(
        np.abs([r["zeta_H0"] for r in rows]),
        np.abs([r["zeta_H1"] for r in rows]),
        np.abs([r["zeta_H2"] for r in rows]),
        [r["M_H0"] for r in rows],
        [r["M_H1"] for r in rows],
        [r["M_H2"] for r in rows],
        out_path=out / "robustness_margins_sensitivity.png",
    )
    for tag in STRUCTURES:
        if sweeps[tag]["X"]:
            plot_fidelity_error_sweeps(
                sweeps[tag]["X"],
                sweeps[tag]["Y"],
                ft,
                xlabel=rf"$\delta_{tag[-1]}$",
                xlim=XLIMS[tag],
                font_size=14,
                num_xticks=3,
                out_path=out / f"{tag}_all.png",
            )


def main() -> None:
    """Run the complete case study: margins, sensitivities, tables and plots."""
    ap = base_parser(OUT_DIR, description=__doc__)
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="Compute fidelity-vs-delta sweeps and H*_all.png",
    )
    ap.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    ap.add_argument(
        "--controller-dir",
        type=Path,
        default=CTRL,
        help="Directory with problem9.mat + controllers.csv "
        "(default: paper set; e.g. results/synth-python)",
    )
    ap.add_argument(
        "--max-error",
        type=float,
        default=DEFAULT_MAX_ERROR,
        help="Nominal error filter for load_controllers",
    )
    args = ap.parse_args()
    ft = args.FT

    ctrl_dir = args.controller_dir
    problem = load_problem(ctrl_dir / "problem9.mat")
    controllers = load_controllers(ctrl_dir / "controllers.csv", args.max_error)
    rows: list[dict] = []
    audit_rows: list[dict] = []
    sweeps = {tag: {"X": [], "Y": []} for tag in STRUCTURES}

    for i, c in enumerate(controllers):
        dt = c["tf"] / c["tau"]
        row = {
            "controller": i + 1,
            "fid": c["fid"],
            "err": c["error"],
        }
        print(f"Controller {i + 1}/{len(controllers)} fid={c['fid']:.6g}", flush=True)
        for tag in STRUCTURES:
            if tag == "H0":
                C = structure_constant("drift", problem["H0"], dt, c["tau"])
            elif tag == "H1":
                C = structure_constant("control", problem["H1"], dt, c["tau"], c["u1"])
            else:
                C = structure_constant("control", problem["H2"], dt, c["tau"], c["u2"])
            L = lipschitz_constant(ft, problem["dim"], C)
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
            margin = iterative_margin(
                fid_fn,
                L,
                ft,
                mu0=0.0,
                eta=ETA,
                omega=(-np.inf, np.inf),
                k_max=K_MAX,
                method="algorithm1",
                return_diagnostics=True,
                margin_tol=AUDIT_MARGIN_TOL,
            )
            audit = margin_audit_record(i + 1, tag, margin)
            audit_rows.append(audit)
            H_list = perturbed_hamiltonians(
                problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag, 0.0
            )
            dH = dH_structure(
                problem["H0"], problem["H1"], problem["H2"], c["u1"], c["u2"], tag
            )
            zeta = differential_sensitivity(
                H_list, dH, dt, problem["Uf"], n_quad=ZETA_N_QUAD
            )
            row[f"M_{tag}"] = float(margin.M)
            row[f"Mm_{tag}"] = float(margin.M_minus)
            row[f"Mp_{tag}"] = float(margin.M_plus)
            row[f"zeta_{tag}"] = float(zeta)

            if args.sweep:
                span = SWEEP_SPAN_FACTOR * max(margin.M_minus, margin.M_plus)
                if span <= 0:
                    span = SWEEP_SPAN_FALLBACK
                x = np.linspace(-span, span, SWEEP_POINTS)
                x, F = fidelity_vs_delta(fid_fn, x)
                sweeps[tag]["X"].append(x)
                sweeps[tag]["Y"].append(1.0 - F)
        rows.append(row)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    write_margin_audit(audit_rows, out, ft)

    csv_name = f"margins_table_{ft:g}.csv"
    fields = list(rows[0].keys())
    for dest in (out / csv_name, BUILD / f"margins_table_{ft:g}_python.csv"):
        with dest.open("w", newline="") as f:
            # LF line endings, as the MATLAB peer writes.

            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {dest}")

    tex_name = f"correlations_{ft:g}.tex"
    write_correlation_tex(rows, out / tex_name)
    focal = write_focal_tests(rows, out / f"focal_tests_{ft:g}.csv")
    for rec in focal:
        print(
            f"  focal M_{rec['j']} vs |zeta_{rec['j']}|: "
            f"rho={rec['rho']:+.3f} p_holm={rec['p_holm']:.3g} "
            f"| tau_b={rec['tau_b']:+.3f} p_holm={rec['tau_p_holm']:.3g}"
        )
    print(f"Wrote {out / tex_name}")

    if args.no_plots:
        return

    plot_results(rows, sweeps, ft, out)
    print(f"Published paper deliverables to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Emit every number the paper quotes in prose as a LaTeX macro.

Tables and figures were already generated; the numbers in running text
were not, and were maintained by hand against a post hoc consistency
check. This closes that gap: each quantity is computed here from the same
CSVs the tables come from, and the paper refers to it by name.

Writes results/paper-xqrm/macros.tex. A number whose source disappears
breaks the build rather than silently keeping its last value: LaTeX errors
on an undefined control sequence, so a macro that stops being emitted takes
the paper down with it. \\xqdef adds the guard LaTeX does not have -- it
refuses to redefine an existing name, so two generators emitting the same
macro is an error instead of a silent overwrite. The reverse direction, a
macro emitted but cited nowhere, is checked by check_reproducible.

Usage: python3 scripts/gen_paper_xqrm_macros.py [--allow-missing]
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
from qrobustness.kosut import effective_threshold

from _paper import col, configure, read

ROOT = Path(__file__).resolve().parents[1]

# Ensemble configuration. These are properties of the shipped problem and
# the driver invocations, not free text; they are emitted so the paper
# cannot drift from the runs.
THRESHOLD = 0.999

#: Joint uncertainty dimension of the main ensemble: drift plus two
#: multiplicative control errors.
N_PARAMS = 3
# The constant-margin counterexample the paper discusses. Named so a
# second one appearing does not silently change which is quoted.
WITNESS_CONTROLLER = "16"
WITNESS_STRUCTURE = "H1"
BUDGET_FACTOR = 1.05


#: Set from --allow-missing. The flag was documented and parsed but never
#: read, so a missing input aborted regardless of what the caller asked for.


# Every reported extreme rounds outward: a quoted range must contain the
# data at both ends. Rounding to nearest reported 5--11 for a spread of
# 4.81--10.52 and 1.5--2.2 for 1.48--2.21, so the printed interval
# excluded the very controllers it was summarising.
def _floor_to(x: float, places: int = 0) -> float:
    """Round down, so a quoted lower end never overstates the data."""
    f = 10.0**places
    return math.floor(x * f) / f


def _ceil_to(x: float, places: int = 0) -> float:
    """Round up, so a quoted bound stays a bound.

    An upper bound printed with ordinary rounding stops being one as soon
    as the value sits just above the last printed digit: 0.1146 formatted
    to two places reads 0.11, and "within 0.11%" is then false.
    """
    f = 10.0**places
    return math.ceil(x * f) / f


def _sci(x: float, places: int = 1, *, up: bool = True) -> str:
    """``x`` in scientific notation, with the exponent taken from ``x``.

    A hard-coded scale is a trap. These quantities are bounds whose
    magnitude moves with the code: the diamond-norm inflation was 5e-5
    under one SDP solver and 5e-9 under another, and a fixed 1e-6 scale
    rendered the second as "0.0e-6" -- a bound reported as zero, which is
    both false and the worst direction to be wrong in.

    The mantissa rounds away from zero by default, so an upper bound stays
    an upper bound at the printed precision; pass ``up=False`` for a lower
    end.
    """
    if x == 0.0:
        return "0"
    exp = math.floor(math.log10(abs(x)))
    mant = x / 10.0**exp
    f = 10.0**places
    mant = (math.ceil(mant * f) if up else math.floor(mant * f)) / f
    # Rounding up can carry the mantissa to 10.0; renormalise.
    if abs(mant) >= 10.0:
        mant /= 10.0
        exp += 1
    return f"{mant:.{places}f}\\times10^{{{exp}}}"


def _is_nonzero(raw) -> bool:
    """True when the underlying quantity is a number and not zero."""
    return isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw != 0


def _renders_as_zero(text: str) -> bool:
    """True when the leading number in a rendered macro is zero.

    Looks only at the mantissa: ``0.0\\times10^{-6}`` is zero however small
    the exponent, while ``4.8\\times10^{-9}`` is not.
    """
    m = re.match(r"\s*-?([0-9]+(?:\.[0-9]+)?)", text)
    return m is not None and float(m.group(1)) == 0.0


class Macros:
    """Collects macro definitions and writes them out.

    Values are formatted at the precision the prose needs; the raw value
    is kept in a comment so a reader can see what was rounded.
    """

    def __init__(self):
        self.items = []

    def add(self, name, value, fmt="{:.2f}", note="", raw=None):
        """Record a macro.

        ``raw`` carries the underlying number when ``value`` is already
        rendered, so the zero-render guard in write() has something to
        compare against; without it a macro whose fixed scale has gone
        stale prints a plausible zero and nothing notices.
        """
        if isinstance(value, str):
            text = value
        else:
            text = fmt.format(value)
        self.items.append((name, text, value if raw is None else raw, note))
        return text

    def write(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        out = [
            "% Auto-generated by scripts/gen_paper_xqrm_macros.py -- do not edit.",
            "% Every number quoted in the paper's prose is defined here and",
            "% computed from the CSVs in results/. Regenerate with",
            "%   make paper-xQRM",
            "%",
            "% \\xqdef defines a macro and refuses to redefine one, so a",
            "% duplicated name is an error rather than a silent overwrite.",
            r"\makeatletter",
            r"\newcommand{\xqdef}[2]{%",
            r"  \@ifundefined{#1}{\expandafter\gdef\csname #1\endcsname{#2}}%",
            r"    {\PackageError{xqrm}{Duplicate generated macro #1}{}}}",
            r"\makeatother",
            "",
        ]
        for name, text, raw, note in self.items:
            # A nonzero quantity must not render as zero. Several macros
            # are written against a fixed power of ten, which is only
            # right while the value stays near that scale; when the
            # diamond-norm inflation fell four orders under a better SDP
            # solver, its fixed 1e-6 scale printed "0.0e-6" and the paper
            # would have claimed the certification adds no inflation at
            # all. Catch it here, where every macro passes, rather than
            # relying on a reader noticing a plausible-looking zero.
            if _renders_as_zero(text) and _is_nonzero(raw):
                raise SystemExit(
                    f"ERROR: macro {name} renders as {text!r} but its value "
                    f"is {raw!r}. The printed scale no longer matches the "
                    "quantity; use _sci() or choose a scale that fits."
                )
            comment = f"  % {raw}" if not isinstance(raw, str) else ""
            if note:
                comment += f"  {note}" if comment else f"  % {note}"
            out.append(f"\\xqdef{{{name}}}{{{text}}}{comment}")
        path.write_text("\n".join(out) + "\n")
        return len(self.items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="tolerate absent inputs (development only)",
    )
    ap.add_argument(
        "--print",
        dest="show",
        action="store_true",
        help="print each macro and its value",
    )
    args = ap.parse_args()
    configure(ROOT / "results", args.allow_missing)

    res = ROOT / "results"
    m = Macros()

    mp = read(res / "multiparameter-margin-python/multiparam_0.999_angular.csv")
    jg = read(res / "multiparameter-margin-python/joint_gauge_0.999.csv")
    kos = read(res / "time-bandwidth-bound-python/kosut_comparison_0.999_angular.csv")
    kos_tv = read(
        res / "time-bandwidth-bound-python/kosut_comparison_0.999_angular_tv.csv"
    )
    sq = read(res / "single-qubit-python/single_qubit_0.999.csv")
    deph = read(res / "lindblad-margin-python/open_margins_0.999.csv")
    amp = read(res / "lindblad-margin-python/open_amp_0.999.csv")

    # -- ensemble ------------------------------------------------------
    m.add("xqNumControllers", len(mp), "{:d}")
    m.add("xqThreshold", THRESHOLD, "{:.3f}")
    m.add("xqBudgetFactor", BUDGET_FACTOR, "{:.2f}")

    # -- Scenario J: the gauge regions against the cross-polytope -------
    # Certified inradius gain of the joint Frobenius gauge over the
    # separable cross-polytope, and the estimated (nonconvex maximum)
    # gain the certified one is a conservative reading of.
    m.add("xqGaugeInradiusGain", np.median(col(jg, "inradius_gain_cert")))
    m.add("xqGaugeInradiusGainEst", np.median(col(jg, "inradius_gain_est")))
    m.add("xqGaugeDiagGain", np.median(col(jg, "diag_gain_med")))
    m.add("xqGaugeAngDiagGain", np.median(col(jg, "ang_diag_gain")))
    m.add("xqGaugeTrajGain", np.median(col(jg, "traj_gain")))
    # The gain is an l1-over-l2 ratio of the per-interval weighted
    # amplitudes, so it cannot exceed sqrt(p) whatever the system. Reporting
    # the ceiling and how much of it this ensemble reaches turns a median
    # into a statement about how much headroom the construction has left.
    m.add("xqGaugeGainCeiling", np.sqrt(N_PARAMS))
    m.add(
        "xqGaugeGainAttainedPct",
        100.0 * col(jg, "diag_gain_max").max() / np.sqrt(N_PARAMS),
        "{:.0f}",
    )

    # -- Scenario T: cost of uniformity over trajectories --------------
    # r_0/M, the price of certifying every bounded trajectory rather than
    # a constant perturbation, per structure.
    # Axis plus diagonal directions, and the total rays iterated over the
    # ensemble; both are read off the table rather than counted by hand.
    n_dirs = len([k for k in mp[0] if re.match(r"^M_[+-]|^M_diag", k)])
    m.add("xqDirectionalDirs", n_dirs, "{:d}")
    m.add("xqDirectionalRays", n_dirs * len(mp), "{:d}")

    for i, (tag, key) in enumerate((("Drift", "H0"), ("CtrlOne", "H1"))):
        m.add(
            f"xqRZeroOverM{tag}", np.median(col(mp, f"r0_{key}") / col(mp, f"M_+e{i}"))
        )

    # -- universal-bound comparison ------------------------------------
    # The paper prints this as a range, so only its two ends are needed.
    for tag, key in (("Drift", "H0"), ("CtrlTwo", "H2")):
        m.add(f"xqKosutRatio{tag}", np.median(col(kos, f"ratio_{key}")))

    # -- single qubit against analytic truth ---------------------------
    amp_sq = [r for r in sq if r["structure"] == "amplitude"][0]
    # Saturation of the path-length bound for the commuting error.
    _rel = abs(float(amp_sq["r_fs"]) - float(amp_sq["delta_star"])) / float(
        amp_sq["delta_star"]
    )
    m.add("xqSingleQubitFsRel", f"{_rel * 1e7:.0f}\\times10^{{-7}}", raw=_rel)

    det = [r for r in sq if r["structure"] == "detuning"][0]
    rfs, dstar = float(det["r_fs"]), float(det["delta_star"])
    m.add("xqSingleQubitRFs", rfs, "{:.3f}")
    m.add("xqSingleQubitCoverage", 100.0 * rfs / dstar, "{:.0f}")
    m.add("xqSingleQubitRZeroCoverage", 100.0 * float(det["r0"]) / dstar, "{:.0f}")

    # -- angular continuation against the Lipschitz step ---------------
    mp_lipschitz = read(res / "multiparameter-margin-python/multiparam_0.999.csv")
    nevs_angular = np.array(
        [float(row[key]) for row in mp for key in row if key.startswith("nev_")]
    )
    nevs_lipschitz = np.array(
        [
            float(row[key])
            for row in mp_lipschitz
            for key in row
            if key.startswith("nev_")
        ]
    )
    m.add(
        "xqAngularNoMoreEvalsPct",
        100.0 * np.mean(nevs_angular <= nevs_lipschitz),
        "{:.0f}",
    )
    m.add(
        "xqAngularMedianEvalSavingPct",
        100.0 * np.median(1.0 - nevs_angular / nevs_lipschitz),
        "{:.0f}",
    )

    # -- adversarial counterexample to constant-scaling ----------------
    witnesses = read(res / "time-bandwidth-bound-python/validity_witness_0.999.csv")
    violations = [row for row in witnesses if row["violated"] == "1"]
    if not violations:
        raise SystemExit(
            "ERROR: no constant-margin counterexample recorded; the paper's "
            "claim that one exists has no evidence behind it. Rerun "
            "run_kosut_validity.py."
        )
    # The count is emitted, not asserted. A stronger adversary finding a
    # second counterexample is a result, not a build failure: the paper
    # quotes this macro, so its sentence follows the run instead of the
    # run being required to agree with the sentence. Aborting here would
    # also contradict --allow-violations, which exists precisely so more
    # counterexamples do not stop the driver.
    # The witness the paper discusses in detail, identified by name rather
    # than by being the only one present.
    named = [
        r
        for r in violations
        if r["controller"] == WITNESS_CONTROLLER and r["structure"] == WITNESS_STRUCTURE
    ]
    if len(named) != 1:
        raise SystemExit(
            f"ERROR: expected the documented witness (controller "
            f"{WITNESS_CONTROLLER}, structure {WITNESS_STRUCTURE}) among the "
            f"{len(violations)} recorded; found {len(named)}. Either the run "
            "does not reproduce it, or the paper should discuss a different "
            "one -- do not simply relabel."
        )
    excess = named[0].get("omega_avg_excess")
    if not excess:
        raise SystemExit(
            "ERROR: violating trajectory lacks omega_avg_excess; "
            "rerun run_kosut_validity.py"
        )
    m.add("xqValidityOmegaAvgExcess", float(excess), "{:.1f}")

    # -- open system: one-step radius against the resolved crossing ----
    # For dephasing the diamond constant equals the first-order slope, so
    # the ratio is 1 up to nominal error and curvature; for amplitude
    # damping the slope is half the diamond rate, hence the factor 2.
    m.add(
        "xqAmpConservatism",
        np.median(col(amp, "amp_star") / col(amp, "r0_amp")),
        "{:.4f}",
    )
    # The one-step radius sits this far below the resolved crossing; the
    # iterated margin closes essentially all of it.
    # The prose states this as a bound across every controller, so it is
    # the maximum and not the median, rounded up: a median would be false
    # for the worst controller, and ordinary rounding would print a value
    # the worst controller exceeds.
    m.add(
        "xqDephasingOneStepExcessPct",
        _ceil_to(
            100.0 * (col(deph, "gamma_star") / col(deph, "r0_gamma") - 1.0).max(), 3
        ),
        "{:.3f}",
    )
    # Evaluation cost of the dephasing iteration, so the paper quotes a
    # generated count rather than a remembered one.
    m.add(
        "xqDephasingEvals", int(round(float(np.median(col(deph, "n_evals"))))), "{:d}"
    )

    # -- coherent perturbation through the open-system functional -------
    # The open-system constant covers dissipation; used on a coherent
    # structure it is much looser than the closed-system one. The paper
    # quotes the size of that gap as a reason to prefer the closed-system
    # certificate when the perturbation is known to be coherent.
    coh = read(res / "lindblad-margin-python/open_coherent_0.999.csv")
    m.add("xqOpenCoherentNum", len(coh), "{:d}")
    m.add("xqOpenCoherentDnorm", np.median(col(coh, "dnorm_unit")), "{:.0f}")
    # Constant across controllers: both constants scale with the same
    # per-interval control amplitudes, so only the norms differ.
    m.add("xqOpenCoherentLRatio", np.median(col(coh, "ratio_L")), "{:.0f}")
    # Outward, so the printed range contains every controller: rounding
    # the ends to nearest reported 110--142 for data spanning 109.6--142.2.
    m.add(
        "xqOpenCoherentRZeroRatioMin", _floor_to(col(coh, "ratio_r0").min()), "{:.0f}"
    )
    m.add("xqOpenCoherentRZeroRatioMax", _ceil_to(col(coh, "ratio_r0").max()), "{:.0f}")
    m.add("xqOpenCoherentIteratedRatio", np.median(col(coh, "ratio_M")), "{:.2f}")

    # -- nominal-error absorption: additive vs angular ------------------
    # The additive allowance uses a threshold that is too low; the angular
    # one is the sufficient reading (triangle inequality on Choi states).
    # The absorption must cover the worst controller in the ensemble, so
    # the quoted nominal error is the maximum, not the median.
    eps0 = float(col(mp, "err").max())
    m.add("xqEpsZero", f"{eps0 * 1e5:.1f}\\times10^{{-5}}", raw=eps0)
    m.add(
        "xqThresholdAdditive",
        effective_threshold(THRESHOLD, eps0, "additive"),
        "{:.4f}",
    )
    m.add(
        "xqThresholdAngular", effective_threshold(THRESHOLD, eps0, "angular"), "{:.5f}"
    )

    # -- threshold sweep: conservatism against the threshold ------------
    # -- what certifying a diamond norm costs in accuracy --------------
    # The appendix bounds the inflation the upward evaluation adds and the
    # deviation of the certified value from the closed forms of the 2n
    # lemma. Both are recorded per generator, so both are read back here
    # rather than typed.
    dnc = read(res / "lindblad-margin-python/dnorm_certificates.csv")
    m.add("xqDnormInflationMax", _sci(max(float(r["rel_inflation"]) for r in dnc)))
    dev = [float(r["dev_closed_form"]) for r in dnc if r["dev_closed_form"] != ""]
    m.add("xqDnormClosedFormDev", _sci(max(dev)))
    m.add(
        "xqDnormFamilySizeMax",
        max(int(r["n_qubits"]) for r in dnc if r["closed_form"]),
        "{:d}",
    )

    ts = read(res / "lindblad-margin-python/open_threshold_sweep.csv")
    for tag, ch in (("Dephasing", "dephasing"), ("Amp", "amp_damping")):
        sub = [r for r in ts if r["channel"] == ch and float(r["FT"]) == 0.99]
        m.add(f"xq{tag}ConservatismLooseFT", np.median(col(sub, "ratio")), "{:.3f}")
    m.add("xqLooseThreshold", 0.99, "{:.2f}")
    m.add("xqSweepNum", len({r["controller"] for r in ts}), "{:d}")

    dsw = read(res / "cnot-python/duration_sweep_0.999.csv")
    m.add(
        "xqDurationNum",
        min(sum(1 for r in dsw if r["tf"] == t) for t in {r["tf"] for r in dsw}),
        "{:d}",
    )

    # -- transfer: a synthesised CNOT ensemble --------------------------
    cn = read(res / "cnot-python/cnot_margins_0.999.csv")
    m.add("xqCnotNum", len(cn), "{:d}")

    # -- margins as a selection tool ------------------------------------
    rn = read(res / "cnot-python/robust_vs_nominal_0.999.csv")
    rob = [r for r in rn if r["kind"] == "robust"]
    nom = [r for r in rn if r["kind"] == "nominal"]
    for tag, key in (("Drift", "H0"), ("CtrlOne", "X1")):
        m.add(
            f"xqRobustGain{tag}",
            np.median(col(rob, f"M_{key}")) / np.median(col(nom, f"M_{key}")),
        )
    m.add("xqRobustNum", len(rob), "{:d}")
    # Adversarial upper witnesses on M_tv: the medians move the opposite way
    # to the static margins, which is what makes the reversal a statement
    # about the true margins rather than about certificate conservatism.
    for tag, key in (("CtrlOne", "X1"), ("CtrlTwo", "X2")):
        wn = col(nom, f"madv_{key}")
        wr = col(rob, f"madv_{key}")
        m.add(
            f"xqRobustWitnessNom{tag}",
            f"{np.median(wn) * 1e2:.1f}\\times10^{{-2}}",
            raw=float(np.median(wn)),
        )
        m.add(
            f"xqRobustWitnessRob{tag}",
            f"{np.median(wr) * 1e2:.1f}\\times10^{{-2}}",
            raw=float(np.median(wr)),
        )
        m.add(f"xqRobustWitnessBelow{tag}", int((wr < np.median(wn)).sum()), "{:d}")
    # The families share synthesis seeds, so the comparison is paired.
    # A paired sign count is a real statistic; the "overlap" measure this
    # replaced changed definition with the order of the medians.
    pairs = {}
    for r in rn:
        pairs.setdefault(r["seed"], {})[r["kind"]] = r
    both = [v for v in pairs.values() if len(v) == 2]
    for name, key in (
        ("Rfs", "rfs_{}"),
        ("Adv", "madv_{}"),
        ("M", "M_{}"),
    ):
        for tag, st in (("CtrlOne", "X1"), ("CtrlTwo", "X2")):
            k = key.format(st)
            down = sum(
                1 for v in both if float(v["robust"][k]) < float(v["nominal"][k])
            )
            m.add(f"xqPaired{name}Down{tag}", down, "{:d}")
    m.add("xqPairedNum", len(both), "{:d}")

    # The sharpest instances, and the comparison is paired because the
    # families share their seeds: a
    # robustified controller whose upper witness falls below its own
    # paired nominal controller's certified lower bound. Comparing against
    # the nominal family's median leaves open which nominal controller was
    # beaten; this does not.
    for tag, k in (("CtrlOne", "X1"), ("CtrlTwo", "X2")):
        paired = sum(
            1
            for v in both
            if float(v["robust"][f"madv_{k}"]) < float(v["nominal"][f"rfs_{k}"])
        )
        m.add(f"xqRobustWitnessBelowPairedRfs{tag}", paired, "{:d}")

    # The mechanism, measured rather than argued: the coherent sum over
    # the gate in the toggling frame is what a static objective can drive
    # down, and the free certificates cannot see it. If robustification
    # lowers this while raising the pulse area, "cancellation bought with
    # trajectory exposure" is demonstrated and not merely plausible.
    # Paired, because a median of ratios hides how many pairs moved: the
    # drift falls on every pair, the controls on most but not all.
    for tag, k in (("Drift", "H0"), ("CtrlOne", "X1"), ("CtrlTwo", "X2")):
        ratio = np.array(
            [
                float(v["robust"][f"cancel_{k}"]) / float(v["nominal"][f"cancel_{k}"])
                for v in both
            ]
        )
        m.add(f"xqRobustCancelRatio{tag}", np.median(ratio))
        m.add(f"xqRobustCancelDown{tag}", int((ratio < 1.0).sum()), "{:d}")

    # Every recorded witness re-evaluated through the independent expm
    # route. A witness that survives only in the search's own arithmetic
    # would show up here.
    recheck = [
        float(r[f"madvF_{k}"])
        for r in rn
        for k in ("X1", "X2")
        if r.get(f"madvF_{k}") not in (None, "")
        and not np.isnan(float(r[f"madvF_{k}"]))
    ]
    m.add("xqWitnessRecheckNum", len(recheck), "{:d}")
    m.add(
        "xqWitnessRecheckViolating",
        int(sum(1 for f in recheck if f < THRESHOLD)),
        "{:d}",
    )
    # The margin below the threshold, not the fidelity: 0.9989999 printed
    # to six places reads as 0.999000 and so as no violation at all. Floored,
    # so the quoted slack is one every witness really has.
    m.add(
        "xqWitnessRecheckSlack",
        f"{_floor_to((THRESHOLD - max(recheck)) * 1e8, 1):.1f}\\times10^{{-8}}",
        raw=THRESHOLD - max(recheck),
    )

    # Why r_FS fell. It is budget/speed, the speed of a multiplicative
    # control structure is proportional to the pulse area ||u_j||_1
    # (Proposition prop:amplitude-only), and the budget depends on the
    # control only through the nominal error. Recovering the two factors
    # from the recorded radii separates "robustification spent amplitude"
    # from "robustification lost fidelity" without re-synthesising
    # anything: the controls themselves are not archived, but the
    # proposition makes the decomposition exact.
    def _angle_budget(rows):
        return np.arccos(THRESHOLD) - np.arccos(np.minimum(1.0, 1.0 - col(rows, "err")))

    bud_nom, bud_rob = _angle_budget(nom), _angle_budget(rob)
    m.add("xqRobustBudgetRatio", np.median(bud_rob / bud_nom))
    for tag, k in (("CtrlOne", "X1"), ("CtrlTwo", "X2")):
        area = (bud_rob / col(rob, f"rfs_{k}")) / (bud_nom / col(nom, f"rfs_{k}"))
        m.add(f"xqRobustAreaRatio{tag}", np.median(area))
        m.add(f"xqRobustAreaUp{tag}", int((area > 1.0).sum()), "{:d}")

    # -- a larger system -------------------------------------------------
    sc = read(res / "scaling-python/scaling4q_margins_0.999.csv")
    m.add("xqScalingNum", len(sc), "{:d}")
    # Deterministic cost of the iterated margin, so the prose can compare
    # certificate classes without quoting a wall-clock that does not
    # reproduce between runs, let alone between machines.
    m.add("xqScalingEvals", int(np.median(col(sc, "n_evals_iter"))), "{:d}")

    # Figure fig:ratios explains its flat row of drift squares by the
    # four-qubit gates being numerically exact. Both numbers it quotes for
    # that come from this file.
    nom = 1.0 - col(sc, "fid")
    exps = {int(np.floor(np.log10(v))) for v in nom}
    if len(exps) != 1:
        raise SystemExit(
            f"ERROR: nominal errors span decades {sorted(exps)}; the caption's "
            "single order of magnitude would be wrong"
        )
    m.add("xqScalingNomErrExp", exps.pop(), "{:d}")
    # The trajectory-class drift ratio, whose spread is the flat row. The
    # constant-class ratio is not flat, so the two must not be conflated.
    dr = col(sc, "rfs_H0") / col(sc, "KMtv_H0")
    m.add(
        "xqScalingDriftFlatRel",
        f"{(dr.max() - dr.min()) / np.median(dr) * 1e6:.1f}\\times10^{{-6}}",
        raw=float((dr.max() - dr.min()) / np.median(dr)),
    )

    # -- comparison with the implied margins of Berberich et al. --------
    # Joined on (controller, structure): both files cover the same
    # ensemble x structure instances.
    ber = read(res / "time-bandwidth-bound-python/berberich_comparison_0.999.csv")
    fsv = read(res / "time-bandwidth-bound-python/fs_validity_0.999.csv")
    fs_by = {(r["controller"], r["structure"]): float(r["r_fs"]) for r in fsv}
    tv_by = {r["controller"]: r for r in kos_tv}
    ratio, ktv_ratio = [], []
    for r in ber:
        key = (r["controller"], r["structure"])
        if key not in fs_by:
            continue
        mb = float(r["MB_ind"])
        ratio.append(fs_by[key] / mb)
        ktv = tv_by.get(r["controller"])
        if ktv is not None:
            ktv_ratio.append(float(ktv[f"KM_{r['structure']}"]) / mb)
    ratio = np.array(ratio)
    m.add("xqBerberichNum", len(ber), "{:d}")
    m.add("xqBerberichFsGain", np.median(ratio))
    m.add("xqBerberichFsGainMin", _floor_to(ratio.min(), 2))
    m.add("xqBerberichFsGainMax", _ceil_to(ratio.max(), 2))
    if ktv_ratio:
        kr = np.array(ktv_ratio)
        m.add("xqBerberichKosutTvMin", _floor_to(kr.min(), 2))
        m.add("xqBerberichKosutTvMax", _ceil_to(kr.max(), 2))

    # -- the correction's effect on the universal-bound comparison ------
    # The published (additive) ratios and the corrected (angular) ones.
    kos_add = read(res / "time-bandwidth-bound-python/kosut_comparison_0.999.csv")

    # -- the trajectory certificate against its competitors -------------
    fsv = read(res / "time-bandwidth-bound-python/fs_validity_0.999.csv")
    fs_by = {(r["controller"], r["structure"]): float(r["r_fs"]) for r in fsv}
    for tag, key in (("Drift", "H0"), ("CtrlOne", "H1"), ("CtrlTwo", "H2")):
        v = [
            fs_by[(r["controller"], key)] / float(r[f"KM_{key}"])
            for r in kos_tv
            if (r["controller"], key) in fs_by
        ]
        m.add(f"xqFsOverKosutTv{tag}", np.median(v))
    # r_FS/r_0 is structure-independent: both scale with the same speed.
    v = [
        fs_by[(r["controller"], "H0")] / float(r["r0_H0"])
        for r in mp
        if (r["controller"], "H0") in fs_by
    ]
    m.add("xqFsOverRZero", np.median(v))
    # Corollary: the ratio is a function of the two fidelities alone, and
    # this is its value for a numerically exact gate, which is what the
    # CNOT and four-qubit ensembles are.
    m.add(
        "xqFsOverRZeroLimit",
        np.sqrt(1.0 - THRESHOLD**2) * np.arccos(THRESHOLD) / (1.0 - THRESHOLD),
    )

    m.add("xqInradiusLTwo", np.median(col(mp, "inradius_l2")), "{:.1f}\\times10^{{-4}}")
    m.items[-1] = (
        "xqInradiusLTwo",
        f"{np.median(col(mp, 'inradius_l2')) * 1e4:.1f}\\times10^{{-4}}",
        float(np.median(col(mp, "inradius_l2"))),
        "",
    )

    # -- ensemble-wide numerical verification of every certificate ------
    # The paper claims a stress-testing harness covers each implemented
    # certificate; these macros make that claim carry its own evidence
    # (breadth, depth, and the failure count) instead of asserting it.
    # tv_slope is one of the checks counted here.
    ver = read(res / "verification-python/verification_0.999.csv")
    m.add("xqVerifyCheckKinds", len({r["check"] for r in ver}), "{:d}")
    m.add("xqVerifyInstances", len(ver), "{:d}")
    m.add("xqVerifyProbes", sum(int(r["n"]) for r in ver), "{:d}")
    m.add("xqVerifyFailures", sum(1 - int(r["passed"]) for r in ver), "{:d}")
    # Worst observed slack as a fraction of the certified bound, over the
    # checks that report one: how close the ensemble came to a violation.
    frac = [
        float(r["max_fraction_of_bound"])
        for r in ver
        if r.get("max_fraction_of_bound") not in (None, "")
    ]
    if frac:
        m.add("xqVerifyMaxFraction", max(frac))

    # The appendix bounds the cross-route fidelity discrepancy and the
    # unitarity defect. fidelity_cross_check returns their sum as the
    # trajectory certificate's tol, so this bounds each of them.
    tol_tc = [float(r["tol"]) for r in ver if r["check"] == "trajectory_certificate"]
    m.add(
        "xqVerifyCrossRouteTol",
        f"{max(tol_tc) * 1e14:.1f}\\times10^{{-14}}",
        raw=max(tol_tc),
    )

    # -- resolved static margin against the trajectory lower certificates
    # These are M/r_0 and M/r_FS: the ratio of a RESOLVED STATIC margin to
    # an available TRAJECTORY lower certificate. They mix uncertainty-class
    # separation with certificate slack and are not a measure of
    # trajectory-certificate conservatism, which would be M_tv/r_FS.
    fsv = read(res / "time-bandwidth-bound-python/fs_validity_0.999.csv")
    fs_by2 = {(r["controller"], r["structure"]): float(r["r_fs"]) for r in fsv}
    price_r0, price_fs, r0_over_ktv = [], [], []
    tv_by2 = {r["controller"]: r for r in kos_tv}
    for i, key in enumerate(("H0", "H1", "H2")):
        M = col(mp, f"M_+e{i}")
        r0 = col(mp, f"r0_{key}")
        fs = np.array([fs_by2[(r["controller"], key)] for r in mp])
        price_r0.append(np.median(M / r0))
        price_fs.append(np.median(M / fs))
        ktv = np.array([float(tv_by2[r["controller"]][f"KM_{key}"]) for r in mp])
        r0_over_ktv.append(np.median(r0 / ktv))
    m.add("xqPriceRZeroMin", _floor_to(min(price_r0)), "{:.0f}")
    m.add("xqPriceRZeroMax", _ceil_to(max(price_r0)), "{:.0f}")
    m.add("xqPriceFsMin", _floor_to(min(price_fs), 2))
    m.add("xqPriceFsMax", _ceil_to(max(price_fs), 2))
    m.add("xqRZeroOverKosutTvDrift", r0_over_ktv[0])
    m.add("xqRZeroOverKosutTvCtrlOne", r0_over_ktv[1])
    m.add("xqRZeroOverKosutTvCtrlTwo", r0_over_ktv[2])

    # -- directional anisotropy of the certified region -----------------
    dirs = [k[2:] for k in mp[0] if k.startswith("M_")]
    Md = np.array([[float(r[f"M_{d}"]) for d in dirs] for r in mp])
    aniso = Md.max(axis=1) / Md.min(axis=1)
    m.add("xqAnisotropyMedian", np.median(aniso))
    m.add("xqAnisotropyMax", _ceil_to(aniso.max(), 2))

    # -- adversarial brackets on the trajectory margin ------------------
    tvb = read(res / "multiparameter-margin-python/tv_bracket_0.999.csv")
    frac = col(tvb, "m_adv") / col(tvb, "M_const")
    m.add("xqTvBracketProbed", len(tvb), "{:d}")
    m.add("xqTvBracketFracMin", _floor_to(frac.min(), 2), "{:.2f}")
    m.add("xqTvBracketFracMax", _ceil_to(frac.max(), 2), "{:.2f}")

    # -- the universal-bound violation, found adversarially -------------
    val = read(res / "time-bandwidth-bound-python/validity_0.999.csv")
    # Three grid levels, so "reachable only by sub-interval refinement"
    # is shown rather than asserted: the control grid stays above FT and
    # only the refinements dip below it.
    # Attacks behind the r_FS sweep: rows x (grids x budget factors), so the
    # count follows the protocol instead of being restated in the prose.
    fsval = read(res / "time-bandwidth-bound-python/fs_validity_0.999.csv")
    n_fmin = len([k for k in fsval[0] if k.startswith("Fmin_")])
    m.add("xqFsAttacks", len(fsval) * n_fmin, "{:d}")

    m.add("xqValidityFminGrid", min(float(r["Fmin_m1_grid"]) for r in val), "{:.6f}")
    m.add("xqValidityFminCoarse", min(float(r["Fmin_m1_x4"]) for r in val), "{:.6f}")
    m.add("xqValidityFminFine", min(float(r["Fmin_m1_x16"]) for r in val), "{:.6f}")
    # Section sec:cases-tv is written around a single violating
    # trajectory: "structure H_1", "that trajectory", "the violating
    # controller", and a singular noun after this count. A different count
    # needs that paragraph rewritten, so stop rather than print a number
    # the sentences around it contradict.
    n_viol = sum(int(r["violated"]) for r in val)
    if n_viol != 1:
        raise SystemExit(
            f"ERROR: the constant-class attack found {n_viol} violations, not 1. "
            "The sec:cases-tv paragraph is written for exactly one (singular "
            "noun, named structure and controller) and must be rewritten before "
            "this macro can be regenerated."
        )
    m.add("xqValidityViolations", n_viol, "{:d}")

    # -- against the implied margins of Berberich et al. ----------------
    # -- the synthesised CNOT ensemble against the universal bound ------
    kk = [np.median(col(cn, f"M_{k}") / col(cn, f"KM_{k}")) for k in ("H0", "X1", "X2")]
    m.add("xqCnotOverKosutMin", _floor_to(min(kk), 2))
    m.add("xqCnotOverKosutMax", _ceil_to(max(kk), 2))
    m.add("xqCnotFsOverRZero", np.median(col(cn, "rfs_X1") / col(cn, "r0_X1")))

    tvb = read(res / "multiparameter-margin-python/tv_bracket_0.999.csv")

    # -- the two-sided bracket on the trajectory margin -----------------
    # Certified lower ends (r_FS) and adversarial violation witnesses
    # (m_adv) on the probed controllers; the interval is strictly
    # positive. Both come from the SAME record as Table tab:tvbracket,
    # joined by controller, structure and threshold, rather than from a
    # separate run whose margins are computed to a different tolerance.
    lo = col(tvb, "r_fs")
    hi = col(tvb, "m_adv")
    m.add("xqTvBracketLowMin", _floor_to(lo.min() * 1e3, 2), "{:.2f}")
    m.add("xqTvBracketLowMax", _ceil_to(lo.max() * 1e3, 2), "{:.2f}")
    m.add("xqTvBracketHighMin", _floor_to(hi.min() * 1e3, 2), "{:.2f}")
    m.add("xqTvBracketHighMax", _ceil_to(hi.max() * 1e3, 2), "{:.2f}")

    # -- the constancy gap M_const - M_tv, per controller ---------------
    # A different quantity from the bracket above, and it was previously
    # reported using the bracket's numbers. Per controller:
    #
    #   gap_lower = max(0, M_const - m_adv)      needs a violating
    #                                            trajectory at m_adv
    #   gap_upper = M_const_upper - r_FS         needs an unsafe constant
    #                                            perturbation at M_upper
    #
    # Endpoints are formed per controller in full precision first and
    # only then reduced to ensemble extremes; a controller with no upper
    # witness contributes no upper endpoint rather than a clipped one.
    gap_lo, gap_hi = [], []
    for r in tvb:
        if not int(r["adv_violated"]):
            raise SystemExit(
                f"ERROR: controller {r['controller']} has no violating "
                "trajectory, so m_adv is not a witness and the constancy "
                "gap has no lower endpoint. Rerun the adversary or drop "
                "the controller from the probed set."
            )
        lo_i = max(0.0, float(r["M_const"]) - float(r["m_adv"]))
        gap_lo.append(lo_i)
        up = float(r["M_const_upper"])
        if not np.isfinite(up):
            continue
        hi_i = up - float(r["r_fs"])
        if lo_i > hi_i:
            raise SystemExit(
                f"ERROR: controller {r['controller']} gives constancy-gap "
                f"lower endpoint {lo_i:.6e} above upper endpoint "
                f"{hi_i:.6e}. The endpoints are computed from inconsistent "
                "records; fix the inputs rather than clipping the interval."
            )
        gap_hi.append(hi_i)
    gap_lo = np.array(gap_lo)
    m.add("xqConstGapLowMin", _floor_to(gap_lo.min() * 1e3, 2), "{:.2f}")
    m.add("xqConstGapLowMax", _ceil_to(gap_lo.max() * 1e3, 2), "{:.2f}")
    if not gap_hi:
        raise SystemExit(
            "ERROR: no controller carries an unsafe constant witness, so "
            "the constancy gap has no upper endpoints. The prose range "
            "must be rewritten before these macros can be regenerated."
        )
    gap_hi = np.array(gap_hi)
    m.add("xqConstGapUpMin", _floor_to(gap_hi.min() * 1e3, 2), "{:.2f}")
    m.add("xqConstGapUpMax", _ceil_to(gap_hi.max() * 1e3, 2), "{:.2f}")
    m.add("xqConstGapBracketed", len(gap_hi), "{:d}")

    # -- how much the absorption choice actually moves ------------------
    # The share of the angle budget the nominal error consumes, and the
    # resulting reduction in the implied universal-bound margin.
    th_T = math.acos(THRESHOLD)
    th_0 = math.acos(1.0 - eps0)
    m.add("xqAbsorptionAngleShare", 100.0 * th_0 / th_T, "{:.0f}")
    dec = [
        100.0
        * (1.0 - np.median(col(kos, f"KM_{k}")) / np.median(col(kos_add, f"KM_{k}")))
        for k in ("H0", "H1", "H2")
    ]
    m.add("xqAbsorptionMarginDropMin", _floor_to(min(dec)), "{:.0f}")
    m.add("xqAbsorptionMarginDropMax", _ceil_to(max(dec)), "{:.0f}")

    out = res / "paper-xqrm" / "macros.tex"
    n = m.write(out)
    if args.show:
        for name, text, raw, _ in m.items:
            print(f"  \\{name:28s} = {text:>10s}   ({raw})")
    print(f"wrote {n} macros to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

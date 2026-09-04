# Fidelity-based robustness margins -- reproduction and tests
#
# Python is the reference implementation. MATLAB and Octave are peers, held
# to it by cross-engine comparison and the golden fixtures.

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
MATLAB ?= matlab
OCTAVE ?= octave
# Which implementation a paper or test target runs. Deliberately NOT
# called LANG: that is the process locale, so it is always set in the
# environment, and make exports a command-line assignment to every recipe
# and sub-make -- `make LANG=matlab` would hand every child process a
# locale of "matlab", in a repository whose whole point is byte-
# reproducible output.
ENGINE ?= python
ENGINES := python matlab octave

# Worker processes for the adversarial sweeps, which are the only targets
# long enough to care. Each (controller, structure) job is independent and
# its seeds depend on the job, not on execution order, so this changes the
# wall clock and nothing else. Capped at 32: the gain past that is small
# against 183 jobs, and it leaves the machine usable.
NPROC := $(shell nproc 2>/dev/null || echo 4)
JOBS ?= $(shell n=$(NPROC); test $$n -gt 32 && echo 32 || echo $$n)

# SERIAL_BLAS and the per-driver flags come from the generated
# fragment below, so the Makefile and check_reproducible cannot
# diverge again. scripts/_invocations.py documents the measurements.

ifeq ($(filter $(ENGINE),$(ENGINES)),)
$(error ENGINE must be one of $(ENGINES), got '$(ENGINE)')
endif

PYTHON := $(ROOT)/.venv/bin/python
# Console scripts in a venv hard-code the absolute path they were created at,
# so they break when the checkout moves; module invocation does not.
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
VENV := $(ROOT)/.venv
BUILD := $(ROOT)/build

# Driver flags come from scripts/_invocations.py so the Makefile and
# check_reproducible cannot drift apart again; see that file for the
# three times they did. The fragment regenerates when the table changes.
DRIVER_FLAGS := $(BUILD)/driver-flags.mk
include $(DRIVER_FLAGS)

# Analyses are named after the method they implement, not after a paper.
# Only the Python tree is published (see the sync-paper-* targets); the MATLAB
# and Octave trees are peers, compared but not published.
LIPSCHITZ_PYTHON := $(ROOT)/results/lipschitz-margin-python
TBBOUND_MATLAB := $(ROOT)/results/time-bandwidth-bound-matlab
TBBOUND_PYTHON := $(ROOT)/results/time-bandwidth-bound-python
TBBOUND_OCTAVE := $(ROOT)/results/time-bandwidth-bound-octave
SYNTH_MATLAB := $(ROOT)/results/synth-matlab
SYNTH_PYTHON := $(ROOT)/results/synth-python
SYNTH_OCTAVE := $(ROOT)/results/synth-octave

# --- Paper-specific layer -----------------------------------------------
# The only paper-aware part of the build: which analysis a given paper
# publishes, where its figures go, and which LaTeX source the verifier reads.
# A second paper is a new block here, not a code change.
#
# Papers live in SIBLING repositories, not inside this one, so results are
# copied across a repository boundary. Override PAPER_ROOT / XPAPER_ROOT if
# your checkout uses different directory names.
PAPER_ID := lcss2026
PAPER_ROOT ?= $(ROOT)/../paper-QRM
PAPER_FIGURES := $(PAPER_ROOT)/figures
PAPER_SOURCE := $(PAPER_ROOT)/main.tex
PAPER_ANALYSIS := lipschitz-margin

# Successor paper (xQRM). Its tables, figures and prose macros are generated
# HERE, into results/paper-xqrm/, and copied into the paper repository, which
# holds only LaTeX. The generators used to live in the paper repository and
# reach back into this one; that put the numbers outside the toolbox that is
# supposed to produce them.
XPAPER_ID := xqrm
XPAPER_ROOT ?= $(ROOT)/../paper-xQRM
XPAPER_OUT := $(ROOT)/results/paper-xqrm
# ------------------------------------------------------------------------

# Fail with a usable message instead of a stray `cp` error when a paper repo
# is not checked out next to this one.
define require_paper
	@test -d "$(1)" || { \
	  echo "ERROR: paper repository not found at $(1)"; \
	  echo "       check it out next to this repository, or pass $(2)=/path/to/paper"; \
	  exit 1; }
endef


# Refuse rather than substitute. Python is the reference implementation, so
# quietly running it when a peer was asked for would report a pass for work
# that never happened.
# One shell statement, so it can be embedded inside an `if` in a recipe
# line; a multi-line define would break the surrounding conditional.
define no_peer
{ echo "ERROR: $(1) has no $(ENGINE) implementation."; \
  echo "       Python is the reference; it is not substituted for ENGINE=$(ENGINE)."; \
  exit 2; }
endef

# --- What a result depends on -------------------------------------------
# A stored result is stale when the driver that wrote it, the library it
# called, or the ensemble it read has changed. Depending on the library is
# the honest choice: results really are a function of it, so editing a
# module does invalidate them. It also means a comment-only edit triggers a
# recompute; touch the outputs, or use make -o, when you know better.
LIB := $(wildcard $(ROOT)/python/src/qrobustness/*.py)
DATA_3Q := $(wildcard $(ROOT)/data/controllers/problem9_tf15_K32_quasi-newton/*)
DATA_CNOT := $(wildcard $(ROOT)/data/controllers/cnot_tf4_K20_lbfgs/*)
DATA_4Q := $(wildcard $(ROOT)/data/controllers/chain4q_tf24_K96_lbfgs/*)
DEP_3Q := $(LIB) $(DATA_3Q)

R := $(ROOT)/results
MP := $(R)/multiparameter-margin-python
TB := $(R)/time-bandwidth-bound-python
SQ := $(R)/single-qubit-python
CN := $(R)/cnot-python
SC := $(R)/scaling-python
LB := $(R)/lindblad-margin-python
VF := $(R)/verification-python

# Per-experiment file sets, named so the phony targets can ask for them.
FILES_multiparam := $(MP)/multiparam_0.999.csv $(MP)/multiparam_0.999_angular.csv \
	$(MP)/tv_bracket_0.999.csv $(MP)/joint_gauge_0.999.csv $(MP)/slice_ctrl1_0.999.npz
FILES_kosut := $(TB)/kosut_comparison_0.999_angular.csv $(TB)/validity_0.999.csv \
	$(TB)/validity_witness_0.999.csv $(TB)/fs_validity_0.999.csv \
	$(TB)/budget_sweep_ctrl16_H1.csv $(TB)/berberich_comparison_0.999.csv \
	$(TB)/kosut_comparison_0.999.csv $(TB)/kosut_comparison_0.999_angular_tv.csv \
	$(TB)/validity_0.999_tv.csv
FILES_single_qubit := $(SQ)/single_qubit_0.999.csv
FILES_cnot := $(CN)/cnot_margins_0.999.csv $(CN)/robust_vs_nominal_0.999.csv \
	$(CN)/duration_sweep_0.999.csv
FILES_scaling := $(SC)/scaling4q_margins_0.999.csv
FILES_open := $(LB)/open_margins_0.999.csv $(LB)/open_coherent_0.999.csv \
	$(LB)/open_amp_0.999.csv \
	$(LB)/open_threshold_sweep.csv $(LB)/mixed_ctrl1.npz
FILES_verification := $(VF)/verification_0.999.csv
GOLDENS := $(ROOT)/data/reference/case_study_subset.json \
	$(ROOT)/data/reference/case_study_subset_matlab.json


XRUN = PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts
MRUN = $(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/examples');
ORUN = $(OCTAVE) --no-gui --eval "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/examples');

.PHONY: help venv clean distclean \
	test test-parity test-parity-all test-synth \
	paper paper-QRM paper-QRM-margins paper-QRM-time-bandwidth \
	paper-xQRM paper-xQRM-multiparam paper-xQRM-kosut paper-xQRM-single-qubit \
	paper-xQRM-cnot paper-xQRM-scaling paper-xQRM-open paper-xQRM-verification \
	reproduce reproduce-QRM reproduce-QRM-margins reproduce-QRM-time-bandwidth \
	reproduce-xQRM \
	check check-QRM check-QRM-consistency \
	check-xQRM check-xQRM-theorems check-xQRM-synth check-xQRM-paper \
	sync-QRM sync-xQRM

help:
	@echo "Fidelity-based robustness margins."
	@echo ""
	@echo "ENGINE selects the implementation for every target below."
	@echo "  ENGINE=python (default, the reference) | matlab | octave"
	@echo "A target with no implementation for the chosen ENGINE fails; it"
	@echo "never falls back to Python."
	@echo ""
	@echo "JOBS=$(JOBS) worker processes for the adversarial sweeps, which are"
	@echo "the only targets long enough to care. Their seeds depend on the job"
	@echo "and not on execution order, so this changes the wall clock and"
	@echo "nothing else: serial and parallel runs agree byte for byte."
	@echo ""
	@echo "Tests"
	@echo "  make test                 Every test for ENGINE, synthesis smoke, then parity"
	@echo "  make test-TESTNAME        One test; TESTNAME is a file in"
	@echo "                            python/tests/test_*.py or matlab/tests/test_*.m"
	@echo "  make test-parity          ENGINE against Python, the reference"
	@echo "                            (no-op for ENGINE=python)"
	@echo "  make test-parity-all      Both non-reference engines against Python"
	@echo "  make test-synth           Synthesis on unseen controllers; never a paper input"
	@echo ""
	@echo "Results   (PAPER = QRM | xQRM)"
	@echo "  make paper                Both papers"
	@echo "  make paper-PAPER          Every experiment of one paper"
	@echo "  make paper-PAPER-EXPNAME  One experiment"
	@echo "    QRM:   margins time-bandwidth"
	@echo "    xQRM:  multiparam kosut single-qubit cnot scaling open verification"
	@echo ""
	@echo "Reproduction   Do the numbers come back the same?"
	@echo "  make reproduce            Both papers: recompute separately and compare"
	@echo "  make reproduce-PAPER      One paper"
	@echo "    QRM:   reproduce-QRM-margins reproduce-QRM-time-bandwidth"
	@echo ""
	@echo "Verification   Are the numbers right?"
	@echo "  make check                Both papers: falsifiable property checks"
	@echo "  make check-PAPER          One paper"
	@echo "  make check-PAPER-ID       One group of checks"
	@echo "    QRM:   consistency"
	@echo "    xQRM:  theorems synth paper"
	@echo ""
	@echo "Publication"
	@echo "  make sync-PAPER           Copy generated artefacts into the paper repo"
	@echo "                            (PAPER_ROOT=$(PAPER_ROOT)"
	@echo "                             XPAPER_ROOT=$(XPAPER_ROOT))"
	@echo ""
	@echo "Utilities"
	@echo "  make venv | clean | distclean"
	@echo ""
	@echo "Results are stored files with prerequisites, so paper- targets"
	@echo "recompute only what is missing or older than the driver, the"
	@echo "library or the input ensemble. The cross-engine fixtures are made"
	@echo "the same way and verified by test-parity."

venv: $(VENV)/bin/python $(VENV)/.extras

$(VENV)/bin/python:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e "$(ROOT)/python/[dev]"

# The plotting extra, installed once rather than from every recipe that
# draws a figure. Three recipes used to run this, and `make paper-QRM`
# runs two of them, so under -j they raced: pip removes and recreates the
# dist-info of an editable install, and the other process then finds it
# half-written ("No such file or directory: ... dist-info/INSTALLER").
# A stamp makes it a prerequisite instead of a side effect.
$(VENV)/.extras: $(VENV)/bin/python $(ROOT)/python/pyproject.toml
	$(PIP) install -q -e "$(ROOT)/python/[plot]"
	@touch $@

clean:
	rm -rf $(BUILD)

distclean: clean
	rm -rf $(VENV)

$(DRIVER_FLAGS): $(ROOT)/scripts/_invocations.py
	@mkdir -p $(dir $@)
	@python3 $< --make > $@

# --- Tests ---------------------------------------------------------------

test: venv
	@if test "$(ENGINE)" = python; then $(PYTEST) -q $(ROOT)/python/tests; \
	elif test "$(ENGINE)" = matlab; then \
	  $(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/tests'); run_all_tests"; \
	else \
	  $(OCTAVE) --no-gui --eval "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/tests'); run_all_tests"; \
	fi
	$(MAKE) test-synth
	$(MAKE) test-parity

# Parity compares the selected ENGINE against Python, the reference
# implementation. Comparing two non-reference engines to each other would
# establish that they agree, not that either is right, so every engine is
# checked against Python directly.
#
# ENGINE=python has nothing to compare: the reference cannot disagree with
# itself, and its golden-fixture test already ran as part of `make test`.
# For the other engines that fixture test runs here too, because it carries
# the cross-engine assertion (Python against the MATLAB-generated golden);
# its Python-against-Python-golden case is only a regression check between
# regenerations, since make rebuilds that golden whenever the library
# changes.
test-parity: venv $(GOLDENS)
	@if test "$(ENGINE)" = python; then \
	  echo "test-parity: ENGINE=python is the reference; nothing to compare."; \
	  echo "  Use ENGINE=matlab or ENGINE=octave to check an engine against it."; \
	else \
	  set -e; \
	  $(PYTEST) -q $(ROOT)/python/tests/test_consistency.py; \
	  if test "$(ENGINE)" = matlab; then \
	    $(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/tests'); test_consistency_matlab"; \
	  else \
	    $(OCTAVE) --no-gui --eval "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/tests'); test_consistency_matlab"; \
	  fi; \
	  $(PYTHON) $(ROOT)/scripts/compare_margins_full.py --a $(ENGINE) --b python; \
	  PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/compare_time_bandwidth_bound.py \
	    --a $(ROOT)/results/time-bandwidth-bound-$(ENGINE)/kosut_comparison_0.999.csv \
	    --b $(TBBOUND_PYTHON)/kosut_comparison_0.999.csv \
	    --label-a $(ENGINE) --label-b python; \
	fi

test-parity-all:
	$(MAKE) test-parity ENGINE=matlab
	$(MAKE) test-parity ENGINE=octave

# Synthesis exercises the one path the frozen ensemble cannot: producing
# controllers the toolbox has never seen. Part of `make test` for that
# reason. The frozen ensemble stays the sole paper input, so this writes
# only to build/.
test-synth: venv
	@if test "$(ENGINE)" = python; then \
	  PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/run_synthesize_controllers.py \
	    --n-opt 2 --maxiter 30 --out $(BUILD)/synth-smoke-python; \
	elif test "$(ENGINE)" = matlab; then \
	  $(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/examples'); run_synthesize_controllers('n_opt',2,'maxiter',30,'out','$(BUILD)/synth-smoke-matlab')"; \
	else \
	  $(OCTAVE) --no-gui --eval "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/examples'); run_synthesize_controllers('n_opt',2,'maxiter',30,'out','$(BUILD)/synth-smoke-octave');"; \
	fi

# One named test. TESTNAME is the file stem, so the available names are
# whatever exists for the engine; a name with no file for this ENGINE is an
# error rather than a silent pass.
test-%: venv
	@if test "$(ENGINE)" = python; then \
	  f=$(ROOT)/python/tests/test_$*.py; \
	  test -f $$f || { echo "ERROR: no python test named '$*' ($$f)"; exit 2; }; \
	  $(PYTEST) -q $$f; \
	else \
	  f=$(ROOT)/matlab/tests/test_$*.m; \
	  test -f $$f || { echo "ERROR: no $(ENGINE) test named '$*' ($$f)"; exit 2; }; \
	  if test "$(ENGINE)" = matlab; then \
	    $(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/tests'); test_$*"; \
	  else \
	    $(OCTAVE) --no-gui --eval "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/tests'); test_$*"; \
	  fi; \
	fi

# --- Stored results: what makes each file -------------------------------
# These are real file rules, so `make paper` recomputes only what is
# missing or out of date. `check` recomputes independently into a scratch
# tree and compares, which is a different question and stays phony.
#
# Grouped targets (&:) where one driver writes several files, so deleting
# any one of them rebuilds the set rather than leaving it half-present.

$(MP)/multiparam_0.999.csv $(MP)/tv_bracket_0.999.csv &: \
		$(ROOT)/scripts/run_multiparameter_case_study.py $(DEP_3Q) | venv
	$(SERIAL_BLAS) $(XRUN)/run_multiparameter_case_study.py \
		$(FLAGS_run_multiparameter_case_study)

$(MP)/multiparam_0.999_angular.csv: \
		$(ROOT)/scripts/run_multiparameter_case_study.py $(DEP_3Q) | venv
	$(SERIAL_BLAS) $(XRUN)/run_multiparameter_case_study.py \
		$(FLAGS_run_multiparameter_case_study_1)

$(MP)/joint_gauge_0.999.csv: $(ROOT)/scripts/run_joint_gauge.py $(DEP_3Q) | venv
	$(SERIAL_BLAS) $(XRUN)/run_joint_gauge.py

# Reads the Lipschitz-stepped table, so it genuinely depends on it.
$(MP)/slice_ctrl1_0.999.npz: $(ROOT)/scripts/run_slice_scan.py \
		$(MP)/multiparam_0.999.csv $(DEP_3Q) | venv
	$(SERIAL_BLAS) $(XRUN)/run_slice_scan.py

$(TB)/kosut_comparison_0.999_angular.csv $(TB)/kosut_comparison_0.999.csv \
$(TB)/kosut_comparison_0.999_angular_tv.csv &: \
		$(ROOT)/scripts/run_time_bandwidth_bound_comparison.py $(DEP_3Q) | venv
	$(SERIAL_BLAS) $(XRUN)/run_time_bandwidth_bound_comparison.py \
		$(FLAGS_run_time_bandwidth_bound_comparison)
	$(SERIAL_BLAS) $(XRUN)/run_time_bandwidth_bound_comparison.py \
		$(FLAGS_run_time_bandwidth_bound_comparison_1)
	$(SERIAL_BLAS) $(XRUN)/run_time_bandwidth_bound_comparison.py \
		$(FLAGS_run_time_bandwidth_bound_comparison_2)

$(TB)/validity_0.999.csv $(TB)/validity_witness_0.999.csv \
$(TB)/validity_0.999_tv.csv &: \
		$(ROOT)/scripts/run_kosut_validity.py $(DEP_3Q) | venv
	$(SERIAL_BLAS) $(XRUN)/run_kosut_validity.py $(FLAGS_run_kosut_validity) \
		--jobs $(JOBS)
	$(SERIAL_BLAS) $(XRUN)/run_kosut_validity.py $(FLAGS_run_kosut_validity_1) \
		--jobs $(JOBS)

$(TB)/fs_validity_0.999.csv: $(ROOT)/scripts/run_fs_validity.py $(DEP_3Q) | venv
	$(SERIAL_BLAS) $(XRUN)/run_fs_validity.py --jobs $(JOBS)

$(TB)/budget_sweep_ctrl16_H1.csv: $(ROOT)/scripts/run_budget_sweep.py $(DEP_3Q) | venv
	$(SERIAL_BLAS) $(XRUN)/run_budget_sweep.py

$(TB)/berberich_comparison_0.999.csv: \
		$(ROOT)/scripts/run_berberich_comparison.py $(DEP_3Q) | venv
	$(SERIAL_BLAS) $(XRUN)/run_berberich_comparison.py

$(SQ)/single_qubit_0.999.csv: $(ROOT)/scripts/run_single_qubit_example.py $(LIB) | venv
	$(SERIAL_BLAS) $(XRUN)/run_single_qubit_example.py

$(CN)/cnot_margins_0.999.csv: $(ROOT)/scripts/run_cnot_case_study.py \
		$(LIB) $(DATA_CNOT) | venv
	$(SERIAL_BLAS) $(XRUN)/run_cnot_case_study.py

$(CN)/robust_vs_nominal_0.999.csv: $(ROOT)/scripts/run_robust_vs_nominal.py \
		$(LIB) $(DATA_CNOT) | venv
	$(SERIAL_BLAS) $(XRUN)/run_robust_vs_nominal.py $(FLAGS_run_robust_vs_nominal)

$(CN)/duration_sweep_0.999.csv: $(ROOT)/scripts/run_duration_sweep.py \
		$(LIB) $(DATA_CNOT) | venv
	$(SERIAL_BLAS) $(XRUN)/run_duration_sweep.py

$(SC)/scaling4q_margins_0.999.csv: $(ROOT)/scripts/run_scaling_example.py \
		$(LIB) $(DATA_4Q) | venv
	$(SERIAL_BLAS) $(XRUN)/run_scaling_example.py

$(LB)/open_margins_0.999.csv $(LB)/open_coherent_0.999.csv &: \
		$(ROOT)/scripts/run_open_system_case_study.py $(DEP_3Q) | venv
	$(SERIAL_BLAS) $(XRUN)/run_open_system_case_study.py

$(LB)/open_amp_0.999.csv: $(ROOT)/scripts/run_open_amplitude_damping.py $(DEP_3Q) | venv
	$(SERIAL_BLAS) $(XRUN)/run_open_amplitude_damping.py

$(LB)/open_threshold_sweep.csv: $(ROOT)/scripts/run_open_threshold_sweep.py $(DEP_3Q) | venv
	$(SERIAL_BLAS) $(XRUN)/run_open_threshold_sweep.py

$(LB)/mixed_ctrl1.npz: $(ROOT)/scripts/run_mixed_example.py $(DEP_3Q) | venv
	$(SERIAL_BLAS) $(XRUN)/run_mixed_example.py

$(VF)/verification_0.999.csv: $(ROOT)/scripts/run_theorem_verification.py $(DEP_3Q) | venv
	$(SERIAL_BLAS) $(XRUN)/run_theorem_verification.py

# The cross-engine fixtures are generated files like any other: paper makes
# them, the parity test verifies MATLAB against them.
$(ROOT)/data/reference/case_study_subset.json: \
		$(ROOT)/scripts/export_golden.py $(DEP_3Q) | venv
	$(XRUN)/export_golden.py

$(ROOT)/data/reference/case_study_subset_matlab.json: \
		$(ROOT)/matlab/tests/export_golden.m $(DATA_3Q)
	$(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/tests'); export_golden"

# --- Results: paper-QRM --------------------------------------------------

paper-QRM-margins: venv
	@if test "$(ENGINE)" = python; then \
	  PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/run_lipschitz_margin_case_study.py --sweep; \
	elif test "$(ENGINE)" = matlab; then \
	  $(MRUN) run_lipschitz_margin_case_study('results_id','lipschitz-margin-matlab')"; \
	else \
	  $(ORUN) run_lipschitz_margin_case_study('results_id','lipschitz-margin-octave');"; \
	fi

paper-QRM-time-bandwidth: venv
	@if test "$(ENGINE)" = python; then \
	  PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/run_time_bandwidth_bound_comparison.py --out $(TBBOUND_PYTHON); \
	elif test "$(ENGINE)" = matlab; then \
	  $(MRUN) run_time_bandwidth_bound_comparison('publish_dir','$(TBBOUND_MATLAB)')"; \
	else \
	  $(ORUN) run_time_bandwidth_bound_comparison('publish_dir','$(TBBOUND_OCTAVE)')"; \
	fi

paper-QRM: paper-QRM-margins paper-QRM-time-bandwidth

# --- Results: paper-xQRM -------------------------------------------------
# The multiparameter study writes multiparam_<FT>.csv under --step lipschitz
# and multiparam_<FT>_angular.csv under --step angular. The paper needs both:
# the evaluation-count macros compare the two stepping rules against each
# other, so a single run cannot reproduce the tree.
paper-xQRM-multiparam: venv
	@if test "$(ENGINE)" = python; then $(MAKE) $(FILES_multiparam); \
	elif test "$(ENGINE)" = matlab; then \
	  $(MRUN) run_multiparameter_case_study('out','$(ROOT)/results/multiparameter-margin-matlab')"; \
	else \
	  $(ORUN) run_multiparameter_case_study('out','$(ROOT)/results/multiparameter-margin-octave');"; \
	fi

paper-xQRM-open: venv
	@if test "$(ENGINE)" = python; then $(MAKE) $(FILES_open); \
	elif test "$(ENGINE)" = matlab; then \
	  $(MRUN) run_open_system_case_study('out','$(ROOT)/results/lindblad-margin-matlab')"; \
	else \
	  $(ORUN) run_open_system_case_study('out','$(ROOT)/results/lindblad-margin-octave');"; \
	fi

paper-xQRM-kosut: venv
	@if test "$(ENGINE)" != python; then $(call no_peer,paper-xQRM-kosut); fi
	$(MAKE) $(FILES_kosut)

paper-xQRM-single-qubit: venv
	@if test "$(ENGINE)" != python; then $(call no_peer,paper-xQRM-single-qubit); fi
	$(MAKE) $(FILES_single_qubit)

paper-xQRM-cnot: venv
	@if test "$(ENGINE)" != python; then $(call no_peer,paper-xQRM-cnot); fi
	$(MAKE) $(FILES_cnot)

paper-xQRM-scaling: venv
	@if test "$(ENGINE)" != python; then $(call no_peer,paper-xQRM-scaling); fi
	$(MAKE) $(FILES_scaling)

paper-xQRM-verification: venv
	@if test "$(ENGINE)" != python; then $(call no_peer,paper-xQRM-verification); fi
	$(MAKE) $(FILES_verification)

# Sequential: the artefact step must see a complete results tree.
paper-xQRM:
	$(MAKE) paper-xQRM-multiparam
	$(MAKE) paper-xQRM-kosut
	$(MAKE) paper-xQRM-single-qubit
	$(MAKE) paper-xQRM-cnot
	$(MAKE) paper-xQRM-scaling
	$(MAKE) paper-xQRM-open
	$(MAKE) paper-xQRM-verification
	@if test "$(ENGINE)" = python; then $(MAKE) sync-xQRM; fi

paper: paper-QRM paper-xQRM

# --- Publication ---------------------------------------------------------
# Per paper, not per engine: Python is the reference and the only
# publication path; the peers are compared, not published.

sync-QRM:
	$(call require_paper,$(PAPER_ROOT),PAPER_ROOT)
	mkdir -p $(PAPER_FIGURES)
	cp -f $(LIPSCHITZ_PYTHON)/H0_all.png $(PAPER_FIGURES)/
	cp -f $(LIPSCHITZ_PYTHON)/H1_all.png $(PAPER_FIGURES)/
	cp -f $(LIPSCHITZ_PYTHON)/H2_all.png $(PAPER_FIGURES)/
	cp -f $(LIPSCHITZ_PYTHON)/robustness_margins_fid_err.png $(PAPER_FIGURES)/
	cp -f $(LIPSCHITZ_PYTHON)/robustness_margins_sensitivity.png $(PAPER_FIGURES)/
	@echo "Published paper-QRM figures from $(LIPSCHITZ_PYTHON) into $(PAPER_FIGURES)"

# Tables, figures and prose macros from whatever is in results/. The
# generators fail on a missing input rather than leaving a stale artefact.
sync-xQRM: venv
	$(call require_paper,$(XPAPER_ROOT),XPAPER_ROOT)
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/gen_paper_xqrm_tables.py
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/gen_paper_xqrm_figures.py
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/gen_paper_xqrm_macros.py
	@mkdir -p $(XPAPER_ROOT)/tables $(XPAPER_ROOT)/figures
	cp -f $(XPAPER_OUT)/tables/*.tex $(XPAPER_ROOT)/tables/
	cp -f $(XPAPER_OUT)/figures/*.pdf $(XPAPER_ROOT)/figures/
	cp -f $(XPAPER_OUT)/macros.tex $(XPAPER_ROOT)/macros.tex
	@echo "Published paper-xQRM tables, figures and macros into $(XPAPER_ROOT)"

# --- Reproduction --------------------------------------------------------
# Recompute into a SEPARATE tree and compare. Recomputing in place proves
# nothing: a driver that silently did not run leaves its old answer behind
# and the tree then agrees with itself.

reproduce-QRM-margins: paper-QRM-margins
	@if test "$(ENGINE)" != python; then \
	  echo "ERROR: independent $(ENGINE) QRM reproduction is not implemented."; exit 2; fi
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/check_reproducible.py --paper qrm

reproduce-QRM-time-bandwidth: paper-QRM-time-bandwidth
	@echo "No independent time-bandwidth reproducer is registered; covered by"
	@echo "make reproduce-QRM-margins and make test-parity."

reproduce-QRM: reproduce-QRM-margins reproduce-QRM-time-bandwidth

reproduce-xQRM: paper-xQRM
	@if test "$(ENGINE)" != python; then \
	  echo "ERROR: independent $(ENGINE) xQRM reproduction is not implemented."; exit 2; fi
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/check_reproducible.py --paper xqrm

reproduce: reproduce-QRM reproduce-xQRM

# --- Verification --------------------------------------------------------
# Falsifiable property checks on the results a paper actually quotes, plus
# the paper's own LaTeX check. Distinct from reproduction: reproduction asks
# whether the numbers come back the same, verification asks whether they are
# right. A tree can reproduce perfectly and still violate a certificate.

check-QRM-consistency: paper-QRM-margins
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/verify_paper_consistency.py \
		--results-id lipschitz-margin-python --paper-source $(PAPER_SOURCE)

check-QRM: check-QRM-consistency

check-xQRM-theorems: venv
	$(MAKE) $(FILES_verification)
	@PYTHONPATH=$(ROOT)/python/src $(PYTHON) -c "import csv,sys; \
rows=list(csv.DictReader(open('$(VF)/verification_0.999.csv'))); \
bad=[r for r in rows if not int(r['passed'])]; \
print('verification: %d checks, %d probes, %d failed' % (len(rows), sum(int(r['n']) for r in rows), len(bad))); \
[print('  FAIL', r['scope'], r['check'], r['min_slack']) for r in bad]; \
sys.exit(1 if bad else 0)"

check-xQRM-paper:
	$(MAKE) -C $(XPAPER_ROOT) check

# The frozen ensemble cannot answer "do the certificates hold on
# controllers you have never seen?", because every certificate here was
# developed against it. This synthesises a small ensemble and runs the
# same harness on it, so a certificate that happened to hold only on the
# shipped controllers would fail here. Writes to build/; never a paper
# input. Its controllers are optimised briefly, so they need a looser
# nominal-error filter than the shipped ensemble.
SYNTH_CHECK := $(BUILD)/synth-check
check-xQRM-synth: venv
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/run_synthesize_controllers.py \
	  --n-opt 4 --maxiter 200 --out $(SYNTH_CHECK)
	$(SERIAL_BLAS) PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/run_theorem_verification.py \
	  --controller-dir $(SYNTH_CHECK) --max-error 1e-2 --out $(SYNTH_CHECK)/verification

check-xQRM: check-xQRM-theorems check-xQRM-synth check-xQRM-paper

check: check-QRM check-xQRM

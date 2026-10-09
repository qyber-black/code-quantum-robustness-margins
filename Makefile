# Fidelity-based robustness margins -- reproduction and tests
#
# Python is the reference implementation. MATLAB and Octave are peers. They
# are held to Python by comparing their committed result tables.

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
MATLAB ?= matlab
OCTAVE ?= octave
# Which implementation a paper or test target runs. The variable is not
# named LANG: that name is the locale, and make would export it to every child.
ENGINE ?= python
ENGINES := python matlab octave

# Worker processes for the adversarial sweeps, capped at 32. The seed depends
# on the job, not on the order of execution, so JOBS does not change the results.
NPROC := $(shell nproc 2>/dev/null || echo 4)
JOBS ?= $(shell n=$(NPROC); test $$n -gt 32 && echo 32 || echo $$n)

ifeq ($(filter $(ENGINE),$(ENGINES)),)
$(error ENGINE must be one of $(ENGINES), got '$(ENGINE)')
endif

PYTHON := $(ROOT)/.venv/bin/python
# A venv console script stores the absolute path it was created with, so it
# breaks when the checkout moves. Invoking the module does not.
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
VENV := $(ROOT)/.venv
BUILD := $(ROOT)/build

# SERIAL_BLAS and the per-driver flags (FLAGS_*, MFLAGS_*), generated from
# scripts/_invocations.py so the Makefile and check_reproducible share them.
DRIVER_FLAGS := $(BUILD)/driver-flags.mk
include $(DRIVER_FLAGS)

# Analyses are named after the method they implement. They are not named
# after a paper. Only the Python tree is published (see the sync-PAPER
# targets). The MATLAB and Octave trees are peers: they are compared, and
# they are not published.
LIPSCHITZ_PYTHON := $(ROOT)/results/lipschitz-margin-python
TBBOUND_MATLAB := $(ROOT)/results/time-bandwidth-bound-matlab
TBBOUND_PYTHON := $(ROOT)/results/time-bandwidth-bound-python
TBBOUND_OCTAVE := $(ROOT)/results/time-bandwidth-bound-octave
SYNTH_MATLAB := $(ROOT)/results/synth-matlab
SYNTH_PYTHON := $(ROOT)/results/synth-python
SYNTH_OCTAVE := $(ROOT)/results/synth-octave

# --- Paper-specific layer -----------------------------------------------
# Which analysis each paper publishes and where its repository is. Papers
# are sibling repositories; the paths are relative to this repository so no
# machine is named. Override PAPER_ROOT / XPAPER_ROOT if yours differ.
PAPER_ID := lcss2026
PAPER_ROOT ?= ../paper-QRM
PAPER_FIGURES := $(PAPER_ROOT)/figures
PAPER_ANALYSIS := lipschitz-margin

# xQRM paper: tables, figures and macros are generated into
# results/paper-xqrm/ and copied into the paper repository.
XPAPER_ID := xqrm
XPAPER_ROOT ?= ../paper-xQRM
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


# Fail when a target has no implementation for ENGINE; Python is never
# substituted. One shell statement, so it can sit inside an `if` in a recipe.
define no_peer
{ echo "ERROR: $(1) has no $(ENGINE) implementation."; \
  echo "       Python is the reference; it is not substituted for ENGINE=$(ENGINE)."; \
  exit 2; }
endef

# --- What a result depends on -------------------------------------------
# A stored result depends on its driver, the library and the controller
# ensemble it reads. A comment-only library edit also triggers a recompute;
# touch the outputs or use make -o to avoid it.
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
LM := $(R)/lipschitz-margin-python
VF := $(R)/verification-python
SE := $(R)/state-examples-python
AT := $(R)/algorithm-tests-python
BA := $(R)/bracket-audit-python

# Per-experiment file sets, used by the run-* targets.
FILES_qrm_margins := $(LM)/margins_table_0.999.csv $(LM)/focal_tests_0.999.csv \
	$(LM)/correlations_0.999.tex \
	$(LM)/H0_all.png $(LM)/H1_all.png $(LM)/H2_all.png \
	$(LM)/robustness_margins_fid_err.png \
	$(LM)/robustness_margins_sensitivity.png
FILES_qrm_time_bandwidth := $(TB)/kosut_comparison_0.999_angular.csv \
	$(TB)/kosut_comparison_0.999.csv $(TB)/kosut_comparison_0.999_angular_tv.csv
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
	$(LB)/open_threshold_sweep.csv $(LB)/open_threshold_cohort.csv \
	$(LB)/mixed_ctrl1.npz \
	$(LB)/dnorm_certificates.csv
FILES_verification := $(VF)/verification_0.999.csv
FILES_states := $(SE)/ghz_detuning_0.999.csv $(SE)/tfim_preparation_0.999.csv \
	$(SE)/ghz_dephasing_0.999.csv $(SE)/closed_limit.csv \
	$(SE)/state_variance_0.999.csv
FILES_algorithm := $(AT)/crosstalk_0.999.csv $(AT)/rays_0.999.csv \
	$(BA)/brackets_0.999.csv $(BA)/timing_0.999.csv $(BA)/environment.json

XRUN = PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts
MRUN = $(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/examples');
ORUN = $(OCTAVE) --no-gui --eval "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/examples');

.PHONY: help install test lint lint-python lint-matlab \
	test-lint test-unit test-parity test-synth \
	run run-QRM run-QRM-margins run-QRM-time-bandwidth \
	run-xQRM run-xQRM-multiparam run-xQRM-kosut run-xQRM-single-qubit \
	run-xQRM-cnot run-xQRM-scaling run-xQRM-open run-xQRM-states \
	run-xQRM-algorithm run-xQRM-verification \
	verify verify-QRM verify-QRM-reproduce verify-QRM-consistency \
	verify-xQRM verify-xQRM-reproduce verify-xQRM-theorems verify-xQRM-synth \
	sync sync-QRM sync-xQRM \
	clean distclean maintainer-clean

help:
	@echo "Fidelity-based robustness margins."
	@echo ""
	@echo "  make install            Set up the environment (.venv; see PINS)"
	@echo "  make test               Every test: lint, then the unit suite, synthesis"
	@echo "                          smoke and parity against Python for every engine;"
	@echo "                          each stage runs whatever the ones before it did"
	@echo "  make run                Run every experiment of both papers"
	@echo "  make verify             Reproduce the experiments in a separate tree and"
	@echo "                          check their falsifiable properties"
	@echo "  make sync-PAPER         Copy one paper's generated tables, figures and"
	@echo "                          macros into its repository; make sync does both"
	@echo "  make clean | distclean | maintainer-clean"
	@echo ""
	@echo "The parts each of these runs, PAPER = QRM | xQRM:"
	@echo "  test-lint test-unit test-synth test-parity   the stages of test"
	@echo "  lint                    lint-python (ruff) and lint-matlab (miss_hit)"
	@echo "  test-TESTNAME           one file of python/tests or matlab/tests"
	@echo "  run-PAPER, run-PAPER-EXPNAME"
	@echo "    QRM:   margins time-bandwidth"
	@echo "    xQRM:  multiparam kosut single-qubit cnot scaling open states"
	@echo "           algorithm verification"
	@echo "  verify-PAPER, verify-PAPER-ID"
	@echo "    QRM:   reproduce consistency"
	@echo "    xQRM:  reproduce theorems synth"
	@echo "  sync-PAPER uses PAPER_ROOT=$(PAPER_ROOT)"
	@echo "             and XPAPER_ROOT=$(XPAPER_ROOT)"
	@echo ""
	@echo "ENGINE=python (the reference) | matlab | octave selects the"
	@echo "implementation of run and its parts, and restricts test to that engine."
	@echo "A target with no implementation for the chosen ENGINE fails; it never"
	@echo "falls back to Python. verify and sync are Python only: the peers are"
	@echo "compared by test-parity, not published."
	@echo ""
	@echo "JOBS=$(JOBS) worker processes for the adversarial sweeps. Their seeds"
	@echo "depend on the job, not on execution order, so serial and parallel"
	@echo "runs agree byte for byte."
	@echo ""
	@echo "PINS=-r python/requirements-repro.txt (the default) installs the pinned"
	@echo "set the published results were computed with; PINS= installs current"
	@echo "releases, which is how to check the code as the ecosystem moves."
	@echo ""
	@echo "Results are stored files with prerequisites, so run recomputes only"
	@echo "what is missing or older than its driver, the library or the input"
	@echo "ensemble. maintainer-clean removes every generated result, so the next"
	@echo "run starts from nothing; the frozen ensembles in data/ are kept."

install: $(VENV)/bin/python $(VENV)/.extras

# Installed from the pinned set the published results were computed with;
# other versions move the open-system results. PINS= installs unpinned.
PINS ?= -r $(ROOT)/python/requirements-repro.txt

$(VENV)/bin/python:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install $(PINS)
	$(PIP) install -e "$(ROOT)/python/[dev]"

# The plotting extra, installed once behind a stamp file so parallel
# recipes do not run pip concurrently.
$(VENV)/.extras: $(VENV)/bin/python $(ROOT)/python/pyproject.toml
	$(PIP) install -q -e "$(ROOT)/python/[plot]"
	@touch $@

clean:
	rm -rf $(BUILD)

distclean: clean
	rm -rf $(VENV)

# Removes results/ (everything make run rebuilds); after a rebuild,
# `git status` shows which results changed. data/ (the inputs) is kept.
maintainer-clean: distclean
	rm -rf $(R)

$(DRIVER_FLAGS): $(ROOT)/scripts/_invocations.py
	@mkdir -p $(dir $@)
	@python3 $< --make > $@

# --- Tests ---------------------------------------------------------------

# ruff on the Python and miss_hit on the MATLAB code, as in CI. Neither
# needs an engine or a MATLAB licence.
lint: lint-python lint-matlab

lint-python: install
	$(VENV)/bin/ruff check $(ROOT)/python/ $(ROOT)/scripts/
	$(VENV)/bin/ruff format --check $(ROOT)/python/ $(ROOT)/scripts/

# miss_hit on matlab/, configured by matlab/miss_hit.cfg.
lint-matlab: install
	$(VENV)/bin/mh_lint $(ROOT)/matlab/
	$(VENV)/bin/mh_metric $(ROOT)/matlab/
	$(VENV)/bin/mh_style $(ROOT)/matlab/

# Every test: lint, unit suite, synthesis smoke and parity against Python,
# for every engine (or only ENGINE when it is given). scripts/_test_suite.py
# runs every stage regardless of failures and prints a tally per engine.
TEST_ENGINES := $(if $(filter command line environment,$(origin ENGINE)),$(ENGINE),$(ENGINES))

test: install
	@rc=0; for e in $(TEST_ENGINES); do \
	  $(PYTHON) $(ROOT)/scripts/_test_suite.py --engine $$e || rc=1; \
	done; exit $$rc

# The stages, individually runnable. test-lint is all of `make lint`
# whatever the ENGINE.
test-lint: lint

test-unit: install
	@if test "$(ENGINE)" = python; then $(PYTEST) -q $(ROOT)/python/tests; \
	elif test "$(ENGINE)" = matlab; then \
	  $(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/tests'); run_all_tests"; \
	else \
	  $(OCTAVE) --no-gui --eval "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/tests'); run_all_tests"; \
	fi

# Compares ENGINE's committed result tables with Python's (the reference);
# nothing to do for ENGINE=python. Also runs the Python consistency test,
# which checks live Python against the peer's committed margins table.
test-parity: install
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
	  PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/compare_time_bandwidth_bound.py \
	    --a $(ROOT)/results/time-bandwidth-bound-$(ENGINE)/kosut_comparison_0.999_angular.csv \
	    --b $(TBBOUND_PYTHON)/kosut_comparison_0.999_angular.csv \
	    --label-a $(ENGINE) --label-b python; \
	  PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/compare_time_bandwidth_bound.py \
	    --a $(ROOT)/results/time-bandwidth-bound-$(ENGINE)/kosut_comparison_0.999_angular_tv.csv \
	    --b $(TBBOUND_PYTHON)/kosut_comparison_0.999_angular_tv.csv \
	    --label-a $(ENGINE) --label-b python; \
	fi

# Smoke test of controller synthesis; writes only to build/.
test-synth: install
	@if test "$(ENGINE)" = python; then \
	  PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/run_synthesize_controllers.py \
	    --n-opt 2 --maxiter 30 --out $(BUILD)/synth-smoke-python; \
	elif test "$(ENGINE)" = matlab; then \
	  $(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/examples'); run_synthesize_controllers('n_opt',2,'maxiter',30,'out','$(BUILD)/synth-smoke-matlab')"; \
	else \
	  $(OCTAVE) --no-gui --eval "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/examples'); run_synthesize_controllers('n_opt',2,'maxiter',30,'out','$(BUILD)/synth-smoke-octave');"; \
	fi

# One test file, test_<TESTNAME> for ENGINE; a missing file is an error.
test-%: install
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
# File rules, so `make run` recomputes only what is missing or out of date.
# Grouped targets (&:) where one driver writes several files.

$(MP)/multiparam_0.999.csv $(MP)/tv_bracket_0.999.csv &: \
		$(ROOT)/scripts/run_multiparameter_case_study.py $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_multiparameter_case_study.py \
		$(FLAGS_run_multiparameter_case_study)

$(MP)/multiparam_0.999_angular.csv: \
		$(ROOT)/scripts/run_multiparameter_case_study.py $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_multiparameter_case_study.py \
		$(FLAGS_run_multiparameter_case_study_1)

$(MP)/joint_gauge_0.999.csv: $(ROOT)/scripts/run_joint_gauge.py $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_joint_gauge.py

# Reads the Lipschitz-stepped table.
$(MP)/slice_ctrl1_0.999.npz: $(ROOT)/scripts/run_slice_scan.py \
		$(MP)/multiparam_0.999.csv $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_slice_scan.py

# One run writes all eight files (tables, correlations, five figures).
# Not pinned to one BLAS thread: the committed results were produced
# unpinned and pinning changes their last bits.
$(FILES_qrm_margins) &: \
		$(ROOT)/scripts/run_lipschitz_margin_case_study.py $(DEP_3Q) | install
	$(XRUN)/run_lipschitz_margin_case_study.py \
		$(FLAGS_run_lipschitz_margin_case_study)

$(TB)/kosut_comparison_0.999_angular.csv $(TB)/kosut_comparison_0.999.csv \
$(TB)/kosut_comparison_0.999_angular_tv.csv &: \
		$(ROOT)/scripts/run_time_bandwidth_bound_comparison.py $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_time_bandwidth_bound_comparison.py \
		$(FLAGS_run_time_bandwidth_bound_comparison)
	$(SERIAL_BLAS) $(XRUN)/run_time_bandwidth_bound_comparison.py \
		$(FLAGS_run_time_bandwidth_bound_comparison_1)
	$(SERIAL_BLAS) $(XRUN)/run_time_bandwidth_bound_comparison.py \
		$(FLAGS_run_time_bandwidth_bound_comparison_2)

$(TB)/validity_0.999.csv $(TB)/validity_witness_0.999.csv \
$(TB)/validity_0.999_tv.csv &: \
		$(ROOT)/scripts/run_kosut_validity.py $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_kosut_validity.py $(FLAGS_run_kosut_validity) \
		--jobs $(JOBS)
	$(SERIAL_BLAS) $(XRUN)/run_kosut_validity.py $(FLAGS_run_kosut_validity_1) \
		--jobs $(JOBS)

$(TB)/fs_validity_0.999.csv: $(ROOT)/scripts/run_fs_validity.py $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_fs_validity.py --jobs $(JOBS)

$(TB)/budget_sweep_ctrl16_H1.csv: $(ROOT)/scripts/run_budget_sweep.py $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_budget_sweep.py

$(TB)/berberich_comparison_0.999.csv: \
		$(ROOT)/scripts/run_berberich_comparison.py $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_berberich_comparison.py

$(SQ)/single_qubit_0.999.csv: $(ROOT)/scripts/run_single_qubit_example.py $(LIB) | install
	$(SERIAL_BLAS) $(XRUN)/run_single_qubit_example.py

# Algorithm tests and the bracket accuracy/cost audit. The audit's timing
# columns vary between runs; check_reproducible exempts only them.
$(AT)/crosstalk_0.999.csv $(AT)/rays_0.999.csv &: \
		$(ROOT)/scripts/run_algorithm_tests.py $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_algorithm_tests.py

$(BA)/brackets_0.999.csv $(BA)/timing_0.999.csv $(BA)/environment.json &: \
		$(ROOT)/scripts/run_bracket_audit.py $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_bracket_audit.py --jobs $(JOBS)

# One driver writes all five state-example files.
$(FILES_states) &: $(ROOT)/scripts/run_state_examples.py $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_state_examples.py

$(CN)/cnot_margins_0.999.csv: $(ROOT)/scripts/run_cnot_case_study.py \
		$(LIB) $(DATA_CNOT) | install
	$(SERIAL_BLAS) $(XRUN)/run_cnot_case_study.py

$(CN)/robust_vs_nominal_0.999.csv: $(ROOT)/scripts/run_robust_vs_nominal.py \
		$(LIB) $(DATA_CNOT) | install
	$(SERIAL_BLAS) $(XRUN)/run_robust_vs_nominal.py $(FLAGS_run_robust_vs_nominal)

$(CN)/duration_sweep_0.999.csv: $(ROOT)/scripts/run_duration_sweep.py \
		$(LIB) $(DATA_CNOT) | install
	$(SERIAL_BLAS) $(XRUN)/run_duration_sweep.py

$(SC)/scaling4q_margins_0.999.csv: $(ROOT)/scripts/run_scaling_example.py \
		$(LIB) $(DATA_4Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_scaling_example.py

$(LB)/open_margins_0.999.csv $(LB)/open_coherent_0.999.csv &: \
		$(ROOT)/scripts/run_open_system_case_study.py $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_open_system_case_study.py

$(LB)/open_amp_0.999.csv: $(ROOT)/scripts/run_open_amplitude_damping.py $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_open_amplitude_damping.py

# The sweep and the cohort it was computed over.
$(LB)/open_threshold_sweep.csv $(LB)/open_threshold_cohort.csv &: \
		$(ROOT)/scripts/run_open_threshold_sweep.py $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_open_threshold_sweep.py

$(LB)/mixed_ctrl1.npz: $(ROOT)/scripts/run_mixed_example.py $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_mixed_example.py

$(LB)/dnorm_certificates.csv: $(ROOT)/scripts/run_dnorm_certificates.py $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_dnorm_certificates.py

$(VF)/verification_0.999.csv: $(ROOT)/scripts/run_theorem_verification.py $(DEP_3Q) | install
	$(SERIAL_BLAS) $(XRUN)/run_theorem_verification.py

# Non-Python engines write their own result trees, compared by test-parity.

run-QRM-margins: install
	@if test "$(ENGINE)" = python; then $(MAKE) $(FILES_qrm_margins); \
	elif test "$(ENGINE)" = matlab; then \
	  $(MRUN) run_lipschitz_margin_case_study('results_id','lipschitz-margin-matlab')"; \
	else \
	  $(ORUN) run_lipschitz_margin_case_study('results_id','lipschitz-margin-octave');"; \
	fi

# Angular absorption writes *_angular.csv, additive the plain name, and
# *_angular_tv.csv the time-varying run; every engine runs all three.
run-QRM-time-bandwidth: install
	@if test "$(ENGINE)" = python; then $(MAKE) $(FILES_qrm_time_bandwidth); \
	elif test "$(ENGINE)" = matlab; then \
	  $(MRUN) run_time_bandwidth_bound_comparison('publish_dir','$(TBBOUND_MATLAB)'$(MFLAGS_run_time_bandwidth_bound_comparison))"; \
	  $(MRUN) run_time_bandwidth_bound_comparison('publish_dir','$(TBBOUND_MATLAB)'$(MFLAGS_run_time_bandwidth_bound_comparison_1))"; \
	  $(MRUN) run_time_bandwidth_bound_comparison('publish_dir','$(TBBOUND_MATLAB)'$(MFLAGS_run_time_bandwidth_bound_comparison_2))"; \
	else \
	  $(ORUN) run_time_bandwidth_bound_comparison('publish_dir','$(TBBOUND_OCTAVE)'$(MFLAGS_run_time_bandwidth_bound_comparison))"; \
	  $(ORUN) run_time_bandwidth_bound_comparison('publish_dir','$(TBBOUND_OCTAVE)'$(MFLAGS_run_time_bandwidth_bound_comparison_1))"; \
	  $(ORUN) run_time_bandwidth_bound_comparison('publish_dir','$(TBBOUND_OCTAVE)'$(MFLAGS_run_time_bandwidth_bound_comparison_2))"; \
	fi

run-QRM: run-QRM-margins run-QRM-time-bandwidth

# --- Results: run-xQRM -------------------------------------------------
# multiparam_<FT>.csv (--step lipschitz) and multiparam_<FT>_angular.csv
# (--step angular); the paper uses both.
run-xQRM-multiparam: install
	@if test "$(ENGINE)" = python; then $(MAKE) $(FILES_multiparam); \
	elif test "$(ENGINE)" = matlab; then \
	  $(MRUN) run_multiparameter_case_study('out','$(ROOT)/results/multiparameter-margin-matlab')"; \
	else \
	  $(ORUN) run_multiparameter_case_study('out','$(ROOT)/results/multiparameter-margin-octave');"; \
	fi

run-xQRM-open: install
	@if test "$(ENGINE)" = python; then $(MAKE) $(FILES_open); \
	elif test "$(ENGINE)" = matlab; then \
	  $(MRUN) run_open_system_case_study('out','$(ROOT)/results/lindblad-margin-matlab')"; \
	else \
	  $(ORUN) run_open_system_case_study('out','$(ROOT)/results/lindblad-margin-octave');"; \
	fi

run-xQRM-kosut: install
	@if test "$(ENGINE)" != python; then $(call no_peer,run-xQRM-kosut); fi
	$(MAKE) $(FILES_kosut)

run-xQRM-single-qubit: install
	@if test "$(ENGINE)" != python; then $(call no_peer,run-xQRM-single-qubit); fi
	$(MAKE) $(FILES_single_qubit)

run-xQRM-cnot: install
	@if test "$(ENGINE)" != python; then $(call no_peer,run-xQRM-cnot); fi
	$(MAKE) $(FILES_cnot)

run-xQRM-scaling: install
	@if test "$(ENGINE)" != python; then $(call no_peer,run-xQRM-scaling); fi
	$(MAKE) $(FILES_scaling)

run-xQRM-algorithm: install
	@if test "$(ENGINE)" != python; then $(call no_peer,run-xQRM-algorithm); fi
	$(MAKE) $(FILES_algorithm)

run-xQRM-states: install
	@if test "$(ENGINE)" != python; then $(call no_peer,run-xQRM-states); fi
	$(MAKE) $(FILES_states)

run-xQRM-verification: install
	@if test "$(ENGINE)" != python; then $(call no_peer,run-xQRM-verification); fi
	$(MAKE) $(FILES_verification)

# Sequential, so the results tree is complete at the end.
run-xQRM:
	$(MAKE) run-xQRM-multiparam
	$(MAKE) run-xQRM-kosut
	$(MAKE) run-xQRM-single-qubit
	$(MAKE) run-xQRM-cnot
	$(MAKE) run-xQRM-scaling
	$(MAKE) run-xQRM-open
	$(MAKE) run-xQRM-states
	$(MAKE) run-xQRM-algorithm
	$(MAKE) run-xQRM-verification

run: run-QRM run-xQRM

# --- Publication ---------------------------------------------------------
# Copies one paper's generated artefacts into its repository. Python only.

sync-QRM:
	$(call require_paper,$(PAPER_ROOT),PAPER_ROOT)
	mkdir -p $(PAPER_FIGURES)
	cp -f $(LIPSCHITZ_PYTHON)/H0_all.png $(PAPER_FIGURES)/
	cp -f $(LIPSCHITZ_PYTHON)/H1_all.png $(PAPER_FIGURES)/
	cp -f $(LIPSCHITZ_PYTHON)/H2_all.png $(PAPER_FIGURES)/
	cp -f $(LIPSCHITZ_PYTHON)/robustness_margins_fid_err.png $(PAPER_FIGURES)/
	cp -f $(LIPSCHITZ_PYTHON)/robustness_margins_sensitivity.png $(PAPER_FIGURES)/
	@echo "Published paper-QRM figures from $(LIPSCHITZ_PYTHON) into $(PAPER_FIGURES)"

sync: sync-QRM sync-xQRM

# Generates tables, figures and macros from results/ into
# results/paper-xqrm/ and copies them; a missing input is an error.
sync-xQRM: install
	$(call require_paper,$(XPAPER_ROOT),XPAPER_ROOT)
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/gen_paper_xqrm_tables.py
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/gen_paper_xqrm_figures.py
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/gen_paper_xqrm_macros.py
	@mkdir -p $(XPAPER_ROOT)/tables $(XPAPER_ROOT)/figures
	cp -f $(XPAPER_OUT)/tables/*.tex $(XPAPER_ROOT)/tables/
	cp -f $(XPAPER_OUT)/figures/*.pdf $(XPAPER_ROOT)/figures/
	cp -f $(XPAPER_OUT)/macros.tex $(XPAPER_ROOT)/macros.tex
	@echo "Published paper-xQRM tables, figures and macros into $(XPAPER_ROOT)"

# --- Verification ----------------------------------------------------------
# Reproduction: recompute into a separate tree and compare with results/.
# Checks: properties the quoted results must satisfy. Every part runs; the
# target fails if any did. verify-xQRM-reproduce also compares the generated
# artefacts with the paper repository's copies (reported, never rewritten).
# Python only.

define verify_all
	@rc=0; for t in $(1); do $(MAKE) $$t || rc=1; done; exit $$rc
endef

verify-QRM-reproduce: run-QRM-margins
	@if test "$(ENGINE)" != python; then $(call no_peer,verify-QRM-reproduce); fi
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/check_reproducible.py --paper qrm \
		--jobs $(JOBS)

verify-QRM-consistency: run-QRM-margins
	@if test "$(ENGINE)" != python; then $(call no_peer,verify-QRM-consistency); fi
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/verify_paper_consistency.py \
		--results-id lipschitz-margin-python

verify-QRM:
	$(call verify_all,verify-QRM-reproduce verify-QRM-consistency)

verify-xQRM-reproduce: run-xQRM
	@if test "$(ENGINE)" != python; then $(call no_peer,verify-xQRM-reproduce); fi
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/check_reproducible.py --paper xqrm \
		--jobs $(JOBS)

verify-xQRM-theorems: install
	@if test "$(ENGINE)" != python; then $(call no_peer,verify-xQRM-theorems); fi
	$(MAKE) $(FILES_verification)
	@PYTHONPATH=$(ROOT)/python/src $(PYTHON) -c "import csv,sys; \
rows=list(csv.DictReader(open('$(VF)/verification_0.999.csv'))); \
bad=[r for r in rows if not int(r['passed'])]; \
print('verification: %d checks, %d probes, %d failed' % (len(rows), sum(int(r['n']) for r in rows), len(bad))); \
[print('  FAIL', r['scope'], r['check'], r['min_slack']) for r in bad]; \
sys.exit(1 if bad else 0)"

# Runs the theorem-verification harness on a freshly synthesised ensemble
# in build/ (looser nominal-error filter, since it is briefly optimised).
SYNTH_CHECK := $(BUILD)/synth-check
verify-xQRM-synth: install
	@if test "$(ENGINE)" != python; then $(call no_peer,verify-xQRM-synth); fi
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/run_synthesize_controllers.py \
	  --n-opt 4 --maxiter 200 --out $(SYNTH_CHECK)
	$(SERIAL_BLAS) PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/run_theorem_verification.py \
	  --controller-dir $(SYNTH_CHECK) --max-error 1e-2 --out $(SYNTH_CHECK)/verification

verify-xQRM:
	$(call verify_all,verify-xQRM-reproduce verify-xQRM-theorems verify-xQRM-synth)

verify:
	$(call verify_all,verify-QRM verify-xQRM)

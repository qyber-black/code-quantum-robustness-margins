# Fidelity-based robustness margins -- reproduction and tests
#
# Python is the reference implementation. MATLAB and Octave are peers, held
# to it by cross-engine comparison and the golden fixtures.

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
MATLAB ?= matlab
OCTAVE ?= octave
PYTHON := $(ROOT)/.venv/bin/python
# Console scripts in a venv hard-code the absolute path they were created at,
# so they break when the checkout moves; module invocation does not.
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
VENV := $(ROOT)/.venv
BUILD := $(ROOT)/build

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
# The paper lives in a SIBLING repository, not inside this one, so results are
# copied across a repository boundary. Override PAPER_ROOT if your checkout
# uses a different directory name.
PAPER_ID := lcss2026
PAPER_ROOT ?= $(ROOT)/../paper-QRM
PAPER_FIGURES := $(PAPER_ROOT)/figures
PAPER_SOURCE := $(PAPER_ROOT)/main.tex
PAPER_ANALYSIS := lipschitz-margin
# ------------------------------------------------------------------------

# Fail with a usable message instead of a stray `cp` error when a paper repo
# is not checked out next to this one.
define require_paper
	@test -d "$(1)" || { \
	  echo "ERROR: paper repository not found at $(1)"; \
	  echo "       check it out next to this repository, or pass $(2)=/path/to/paper"; \
	  exit 1; }
endef

.PHONY: help venv test test-fast test-matlab test-python test-consistency \
	export-golden lipschitz-margin-matlab lipschitz-margin-python lipschitz-margin-octave compare-full compare-octave \
	sync-paper sync-paper-qrm \
	verify-paper-matlab verify-paper-python verify-paper-octave verify-paper check-margins \
	synth-matlab synth-python synth-octave synth-smoke \
	time-bandwidth-bound time-bandwidth-bound-matlab time-bandwidth-bound-python \
	time-bandwidth-bound-octave compare-time-bandwidth-bound \
	analyse-synth-matlab analyse-synth-python \
	clean distclean

help:
	@echo "Targets:"
	@echo "  make venv                 Create Python venv and install package"
	@echo "  make test                 Python + MATLAB + consistency"
	@echo "  make test-python          Python unit tests only"
	@echo "  make test-matlab          MATLAB unit tests only"
	@echo "  make test-consistency     Python vs MATLAB against the golden fixtures"
	@echo "  make test-fast            Smoke tests only"
	@echo "  make lipschitz-margin-matlab         Full case study (MATLAB) -> results/lipschitz-margin-matlab/"
	@echo "  make lipschitz-margin-python         Full case study (Python) -> results/lipschitz-margin-python/"
	@echo "  make lipschitz-margin-octave         Full case study (Octave) -> results/lipschitz-margin-octave/"
	@echo "  make compare-full         Compare Python vs MATLAB margin tables"
	@echo "  make compare-octave       Compare Python vs Octave margin tables"
	@echo "  make sync-paper-qrm       Publish paper-QRM figures (Python results) -> \$$(PAPER_ROOT)/figures/"
	@echo "  make sync-paper           Alias for sync-paper-qrm"
	@echo "                            (the paper is a sibling repo: PAPER_ROOT=$(PAPER_ROOT))"
	@echo "  make verify-paper-matlab  Checks vs results/lipschitz-margin-matlab/ -> verify_paper.md there"
	@echo "  make verify-paper-python  Checks vs results/lipschitz-margin-python/ -> verify_paper.md there"
	@echo "  make verify-paper-octave  Checks vs results/lipschitz-margin-octave/ -> verify_paper.md there"
	@echo "  make verify-paper         Both verify-paper-python and verify-paper-matlab"
	@echo "  make check-margins        lipschitz-margin-python/matlab + compare-full + verify-paper (release gate)"
	@echo "  make synth-matlab         Optimize 100 controllers -> results/synth-matlab/"
	@echo "  make synth-python         Optimize 100 controllers -> results/synth-python/"
	@echo "  make synth-octave         Optimize 100 controllers -> results/synth-octave/ (core Octave only)"
	@echo "  make synth-smoke          2-start synthesis smoke (Python + MATLAB)"
	@echo "  make analyse-synth-matlab Margins for synth-matlab -> results/synth-matlab-margins/"
	@echo "  make analyse-synth-python Margins for synth-python -> results/synth-python-margins/"
	@echo "  make time-bandwidth-bound     Kosut et al. bound comparison (Python + MATLAB + compare)"
	@echo "  make time-bandwidth-bound-matlab   ... MATLAB only -> results/time-bandwidth-bound-matlab/"
	@echo "  make time-bandwidth-bound-python   ... Python only -> results/time-bandwidth-bound-python/"
	@echo "  make time-bandwidth-bound-octave   ... Octave peer  -> results/time-bandwidth-bound-octave/"
	@echo "  make compare-time-bandwidth-bound        Compare Python vs MATLAB Kosut tables"
	@echo "  make export-golden        Refresh Python + MATLAB goldens in data/reference/"
	@echo "  make clean                Remove build/"
	@echo "  make distclean            Remove build/ and venv"

venv: $(VENV)/bin/python

$(VENV)/bin/python:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e "$(ROOT)/python/[dev]"

test: test-python test-matlab test-consistency

test-fast: venv
	$(PYTEST) -q $(ROOT)/python/tests/test_core.py
	$(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/tests'); run_all_tests"
	$(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/examples'); run_smoke_case_study"

test-matlab:
	$(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/tests'); run_all_tests"

test-python: venv
	$(PYTEST) -q $(ROOT)/python/tests

test-consistency: venv
	@test -f $(ROOT)/data/reference/case_study_subset.json || $(MAKE) export-golden
	@test -f $(ROOT)/data/reference/case_study_subset_matlab.json || $(MAKE) export-golden
	$(PYTEST) -q $(ROOT)/python/tests/test_consistency.py
	$(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/tests'); test_consistency_matlab"

export-golden: venv
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/export_golden.py
	$(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/tests'); export_golden"

lipschitz-margin-matlab:
	$(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/examples'); run_lipschitz_margin_case_study('results_id','lipschitz-margin-matlab')"

lipschitz-margin-python: venv
	$(PIP) install -q -e "$(ROOT)/python/[plot]"
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/run_lipschitz_margin_case_study.py --sweep

lipschitz-margin-octave:
	$(OCTAVE) --no-gui --eval "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/examples'); run_lipschitz_margin_case_study('results_id','lipschitz-margin-octave');"

compare-full: venv
	$(PYTHON) $(ROOT)/scripts/compare_matlab_python_full.py

compare-octave: venv
	$(PYTHON) $(ROOT)/scripts/compare_matlab_octave_full.py

# Publishing is per PAPER, not per language: Python is the reference
# implementation and produces the manuscript figures (MATLAB and Octave are
# peers held to it by compare-full / compare-octave, not publication paths).
sync-paper-qrm:
	$(call require_paper,$(PAPER_ROOT),PAPER_ROOT)
	mkdir -p $(PAPER_FIGURES)
	cp -f $(LIPSCHITZ_PYTHON)/H0_all.png $(PAPER_FIGURES)/
	cp -f $(LIPSCHITZ_PYTHON)/H1_all.png $(PAPER_FIGURES)/
	cp -f $(LIPSCHITZ_PYTHON)/H2_all.png $(PAPER_FIGURES)/
	cp -f $(LIPSCHITZ_PYTHON)/robustness_margins_fid_err.png $(PAPER_FIGURES)/
	cp -f $(LIPSCHITZ_PYTHON)/robustness_margins_sensitivity.png $(PAPER_FIGURES)/
	@echo "Published paper-QRM figures from $(LIPSCHITZ_PYTHON) into $(PAPER_FIGURES)"

sync-paper: sync-paper-qrm

verify-paper-matlab:
	$(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/scripts'); verify_paper_consistency('results_id','lipschitz-margin-matlab','paper_source','$(PAPER_SOURCE)')"

verify-paper-python: venv
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/verify_paper_consistency.py \
		--results-id lipschitz-margin-python --paper-source $(PAPER_SOURCE)

verify-paper-octave:
	$(OCTAVE) --no-gui --eval "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/scripts'); verify_paper_consistency('results_id','lipschitz-margin-octave','paper_source','$(PAPER_SOURCE)');"

verify-paper: verify-paper-python verify-paper-matlab

# Sequential: compare/verify must see freshly written results (safe under make -j).
check-margins:
	$(MAKE) lipschitz-margin-python
	$(MAKE) lipschitz-margin-matlab
	$(MAKE) compare-full
	$(MAKE) verify-paper

synth-matlab:
	$(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/examples'); run_synthesize_controllers('out','$(SYNTH_MATLAB)')"

synth-python: venv
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/run_synthesize_controllers.py --out $(SYNTH_PYTHON)

# Uses core Octave only: fminunc and optimset ship with Octave, so no Forge
# package is needed (optimoptions is MATLAB-only and is not used there).
synth-octave:
	$(OCTAVE) --no-gui --eval "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/examples'); run_synthesize_controllers('out','$(SYNTH_OCTAVE)');"

synth-smoke: venv
	$(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/examples'); run_synthesize_controllers('n_opt',2,'maxiter',30,'out','$(BUILD)/synth-smoke-matlab')"
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/run_synthesize_controllers.py \
		--n-opt 2 --maxiter 30 --out $(BUILD)/synth-smoke-python

analyse-synth-matlab:
	@test -f $(SYNTH_MATLAB)/controllers.csv || $(MAKE) synth-matlab
	$(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/examples'); run_lipschitz_margin_case_study('controller_dir','$(SYNTH_MATLAB)','results_id','synth-matlab-margins')"

analyse-synth-python: venv
	@test -f $(SYNTH_PYTHON)/controllers.csv || $(MAKE) synth-python
	$(PIP) install -q -e "$(ROOT)/python/[plot]"
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/run_lipschitz_margin_case_study.py --sweep \
		--controller-dir $(SYNTH_PYTHON) --out $(ROOT)/results/synth-python-margins

time-bandwidth-bound-matlab:
	$(MATLAB) -batch "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/examples'); run_time_bandwidth_bound_comparison('publish_dir','$(TBBOUND_MATLAB)')"

time-bandwidth-bound-python: venv
	$(PIP) install -q -e "$(ROOT)/python/[plot]"
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/run_time_bandwidth_bound_comparison.py \
		--out $(TBBOUND_PYTHON)

time-bandwidth-bound-octave:
	$(OCTAVE) --no-gui --eval "addpath('$(ROOT)/matlab'); addpath('$(ROOT)/matlab/examples'); run_time_bandwidth_bound_comparison('publish_dir','$(TBBOUND_OCTAVE)')"

compare-time-bandwidth-bound: venv
	PYTHONPATH=$(ROOT)/python/src $(PYTHON) $(ROOT)/scripts/compare_time_bandwidth_bound.py

# Sequential: compare must see freshly written results (safe under make -j).
time-bandwidth-bound:
	$(MAKE) time-bandwidth-bound-python
	$(MAKE) time-bandwidth-bound-matlab
	$(MAKE) compare-time-bandwidth-bound

clean:
	rm -rf $(BUILD)

distclean: clean
	rm -rf $(VENV)

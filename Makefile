# Convenience targets. Every step can also be run as `python scripts/<name>.py`.
# Steps marked (db) need the OMOP database (.env); the others run from the cache.
PY ?= python

.PHONY: help install test lint cohort extract model tests-paired lgbm calibration symptom-groups descriptives figures all clean

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

install:          ## editable install with the HANA driver, notebook and dev extras
	pip install -e ".[hana,figures,dev]"

test:             ## unit tests (no database needed)
	$(PY) -m pytest

lint:             ## ruff
	ruff check src scripts tests

cohort:           ## (db) cohort table -> cohort_parquets/combined_strict_all_patients.parquet
	$(PY) scripts/build_cohort.py

extract:          ## (db) per-window feature matrices, no modelling
	PASC_REEXTRACT=1 PASC_EXTRACT_ONLY=1 $(PY) scripts/run_main_analysis.py

model:            ## 7 configurations x 6 windows x 100 repeats from the cache -> results/main/
	PASC_REEXTRACT=0 $(PY) scripts/run_main_analysis.py

tests-paired:     ## paired Wilcoxon + BH for every comparison -> results/main/perc97/paired_wilcoxon_bh.csv
	$(PY) scripts/paired_tests.py

lgbm:             ## LightGBM robustness check (Table B.5)
	$(PY) scripts/run_lgbm_check.py

calibration:      ## calibration of both primary configurations at w0-90
	PASC_CALIB_CONFIG=baseline_ext $(PY) scripts/run_calibration.py
	PASC_CALIB_CONFIG=mechsig_all  $(PY) scripts/run_calibration.py

symptom-groups:   ## (db for the first pass) secondary analysis S1
	PASC_REEXTRACT=1 PASC_EXTRACT_ONLY=1 PASC_FEATURE_WINDOWS=0_90 $(PY) scripts/run_symptom_groups.py
	PASC_REEXTRACT=0 PASC_FEATURE_WINDOWS=0_90 $(PY) scripts/run_symptom_groups.py

descriptives:     ## Table 4.1, Table B.1 (db once, for race/ethnicity) and Figure 4.1
	$(PY) scripts/make_table1.py
	$(PY) scripts/make_consort.py

figures:          ## Figures 4.2-4.7, B.1 and appendix tables from the result files
	jupyter nbconvert --execute --to notebook --inplace notebooks/make_thesis_figures.ipynb

all: cohort extract model tests-paired lgbm calibration symptom-groups descriptives figures  ## everything, in order

clean:            ## remove caches (never touches cohort_parquets/ or results/)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache src/*.egg-info

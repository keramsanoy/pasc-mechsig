# pasc-mechsig

Analysis code for the Master's thesis **Predicting Post-Acute Sequelae of COVID-19 from
Electronic Health Records: The Added Value of Mechanism-Based Feature Sets** (Hasso Plattner
Institute, University of Potsdam, 2026).

The study asks whether the three-cluster mechanism framework of Greenhalgh et al. (viral
persistence, immunoinflammatory dysregulation, endothelial dysfunction) improves the prediction
of a post-acute COVID-19 diagnosis when the mechanisms are operationalised as features of the
OMOP Common Data Model. The reference is an independent re-implementation of the Antony et al.
(2023) acute-phase model and its extension with routinely collected covariates; a second
question is whether any gain depends on the feature-extraction window.

The code ran against AIR·MS, the OMOP CDM v5.3 implementation of the Mount Sinai Health System.
**This repository contains no data.** The analysis used protected health information under an
IRB-approved protocol inside the MSHS enclave, and nothing patient-level or data-derived is
released. What is here is enough for someone with that access, or with another OMOP CDM site,
to repeat the analysis. See [Data access](#data-access).

## Repository layout

```
pasc-mechsig/
├── src/pasc/                     the package
│   ├── config/
│   │   ├── paths.py              repository-relative directories (PASC_COHORT_DIR, PASC_RESULTS_DIR)
│   │   └── omop.py               CDM schema name (OMOP_CDM_SCHEMA)
│   ├── db.py                     connect(): database connection from .env, SSH tunnel, schema rewrite
│   ├── cohort/
│   │   ├── antony.py             COVID-positive base cohort, PASC label, acute window (§3.2, §3.3)
│   │   └── combined.py           follow-up, death handling, fixed data cutoff (§3.2, Table 3.1)
│   ├── features/
│   │   ├── antony.py             families A–E of the Antony et al. replication (§3.4.1)
│   │   ├── extended.py           extended covariates, engagement controls, feature-matrix orchestrator (§3.4.1)
│   │   ├── indicators.py         the binary mechanism indicators of the three clusters (Tables A.1–A.3)
│   │   ├── signals.py            concept resolution and the SQL that extracts an indicator
│   │   ├── enhanced.py           laboratory summaries, CBC indices, composites, leakage gate,
│   │   │                         configuration builder (§3.4.2, §3.5, Table 3.2)
│   │   └── omop2obo.py           OMOP2OBO mapping of acute symptoms to HPO terms
│   ├── modeling/
│   │   ├── pipeline.py           one repeat: split, balance, prevalence filter, Boruta, imputation,
│   │   │                         grid-searched Random Forest, SHAP (Table 3.5)
│   │   └── calibration.py        calibration measures, recalibration, decision curves (Appendix A.3)
│   └── analysis/
│       └── paired.py             paired Wilcoxon + Benjamini–Hochberg for every comparison (§3.8)
├── scripts/                      entry points, in the order they are run
│   ├── build_cohort.py           (db)  cohort table
│   ├── run_main_analysis.py      (db)  7 configurations × 6 windows × 100 repeats  ← Appendix A.5
│   ├── paired_tests.py                 Tables 4.3, B.2–B.4
│   ├── run_lgbm_check.py               LightGBM robustness check, Table B.5
│   ├── make_lgbm_table.py              Table B.5 from the repeats
│   ├── run_calibration.py              Table B.11, predictions for Figure B.1
│   ├── run_symptom_groups.py     (db)  secondary analysis S1, Table B.12
│   ├── make_table1.py            (db)  Table 4.1, Table B.1
│   └── make_consort.py                 Figure 4.1
├── notebooks/make_thesis_figures.ipynb   Figures 4.2–4.7, B.1 and the appendix tables, from result files only
├── hpc/                          LSF submit scripts used for the per-window runs
├── tests/                        unit tests that need no database
├── docs/FEATURE_CODINGS_REFERENCE.md     every concept ID, LOINC code, RxNorm ingredient, SNOMED ancestor
│                                         and source-value pattern the pipeline uses
├── data/omop2obo/                place the OMOP2OBO mapping file here (see below)
├── cohort_parquets/              created at run time: cohort table and cached feature matrices (git-ignored)
└── results/                      created at run time: every output (git-ignored)
```

`(db)` marks the steps that query the database; everything else runs from the cached feature
matrices in `cohort_parquets/`.

## Installation

```bash
git clone <this repository> pasc-mechsig && cd pasc-mechsig
conda create -n pasc python=3.12 && conda activate pasc      # or any Python ≥ 3.11 environment
pip install -e ".[hana,figures,dev]"                            # package + HANA driver + notebook + tests
cp .env.example .env                                            # database host, port, user, schema
python -m pytest                                                # 21 tests, no database needed
```

The `scripts/` also run from a plain clone without `pip install` (they add `src/` to the path
themselves), but installing is what the notebook and the tests expect.

Library versions matter here more than usual. The forest, imputer, splitting and grid search
come from scikit-learn 1.8.0, feature selection from Boruta 0.4.3, attributions from shap 0.49.1
and the robustness check from LightGBM 4.6.0 (Python 3.12). `pyproject.toml` pins them. Given
the same code, data and seeds a run reproduces exactly; a different scikit-learn release can fit
different forests.

**OMOP2OBO.** The acute-symptom family maps diagnosis codes to HPO terms through OMOP2OBO
v2.0.0 (Callahan et al., Zenodo [10.5281/zenodo.7255922](https://doi.org/10.5281/zenodo.7255922)).
Download `OMOP2OBO_v2.0.0_N3C_Enclave_CSV_concept_set_expression_items.csv` from that record
into `data/omop2obo/`.

**Database.** `pasc.db.connect()` opens an `hdbcli` connection to SAP HANA, optionally through an
SSH tunnel from a compute node, with every host, port, credential and schema read from `.env`
(`.env.example` lists the variables). The SQL names the CDM schema literally as `CDMPHI`, the
AIR·MS convention; setting `OMOP_CDM_SCHEMA` substitutes another name into every statement at
execution time. The SQL is HANA-flavoured (`#temp` tables, `ADD_DAYS`, `TOP n`); another engine
needs the templates in `pasc/features/signals.py`, `pasc/cohort/antony.py`,
`pasc/features/antony.py` and `pasc/features/extended.py` adapted.

## Reproducing the analysis

```bash
make cohort            # 1. (db) cohort table -> cohort_parquets/combined_strict_all_patients.parquet
make extract           # 2. (db) six per-window feature matrices -> cohort_parquets/enhanced_<window>_strict_*
make model             # 3. every configuration, 100 repeats each -> results/main/perc97/
make tests-paired      # 4. paired Wilcoxon + BH -> results/main/perc97/paired_wilcoxon_bh.csv
make lgbm              # 5. LightGBM check on the same divisions -> results/lgbm_robustness/
make calibration       # 6. calibration of baseline_ext and mechsig_all at w0-90 -> results/calibration/
make symptom-groups    # 7. (db once) symptom-group models at w0-90 -> results/symptom_groups/
make descriptives      # 8. (db once) Table 4.1, Table B.1, Figure 4.1 -> results/descriptives/
make figures           # 9. Figures 4.2–4.7, B.1 -> results/figures/
```

`make help` lists the targets; each one is a single `python scripts/<name>.py` call that can
be run directly. On an LSF cluster, step 3 is launched as one job per window with
`hpc/submit_window.sh` after copying `hpc/env.sh` to `hpc/env.local.sh` and filling in the
allocation, queue and conda environment.

### What one run of `run_main_analysis.py` does

For every window in `FEATURE_WINDOWS` (w0-21, w0-30, w0-60, w0-90, w30-60, w60-90) the script
re-extracts every feature inside that window: the Antony et al. families A–E, the six blocks of
extended covariates, the six engagement controls, the 70 binary indicators of the three
clusters, the six-column laboratory summaries, the CBC-derived indices, the composites, and the
temporal-divergence composites gated by `max_window_end_day`. The matrix is cached as
`cohort_parquets/enhanced_<window>_strict_all_patients.parquet` with a `families.json` beside it.

Each configuration in `ACTIVE_CONFIGS` (the seven primary ones of Table 3.2 and the `_no_td` /
`_no_eng` refits of §3.9) is then fitted 100 times with `pasc.modeling.pipeline.run_full_pipeline`.
Repeat *i* uses seed 42 + 1000·*i* for everything random in it: the stratified 80/20 split, 1:1
undersampling of the training part, the < 1 % prevalence filter, Boruta (50 rounds, 500 trees,
depth ≤ 7, 97th shadow percentile), median imputation, and the five-fold grid search over 100/300/500
trees, depth 5/10/20/unlimited and minimum leaf 1/5/10 scored on accuracy. The engagement
controls are forced past the prevalence filter and Boruta. The index-date calendar feature
`f_ext_index_year_month` is withheld from modelling (§3.6). Each repeat records AUROC and AUPRC
on the untouched test part, the features Boruta kept, the chosen settings, and exact tree SHAP
values on up to 2,000 test patients.

### Switches

All are environment variables; the defaults are the thesis settings.

| Variable | Default | Effect |
|---|---|---|
| `PASC_REEXTRACT` | `1` | `0` skips the database and models from the cached matrices |
| `PASC_EXTRACT_ONLY` | `0` | `1` writes the matrices and stops before modelling |
| `PASC_FEATURE_WINDOWS` | all six | e.g. `0_90` or `0_21,60_90` |
| `PASC_ACTIVE_CONFIGS` | thesis set | comma-separated subset of the configuration names |
| `PASC_MIN_FOLLOWUP_DAYS` | `365` | follow-up a control needs to count as a confirmed negative |
| `PASC_DROP_INDEX_CALENDAR` | `1` | `0` lets the model see the index-date feature |
| `PASC_SKIP_EXISTING` | `0` | `1` resumes: configurations with an `iterations_*.csv` are skipped |
| `PASC_FAST_MODE` | `0` | `1` = 5 repeats, small Boruta, for a smoke test only |
| `ANTONY_N_JOBS` | `4` | cores for the forests |
| `PASC_COHORT_DIR`, `PASC_RESULTS_DIR` | in repo | move the cache and the outputs elsewhere |
| `OMOP_CDM_SCHEMA` | `CDMPHI` | schema substituted into the SQL at execution time |

`run_lgbm_check.py`, `run_calibration.py` and `run_symptom_groups.py` document their own
switches in their headers. `PASC_LGBM_EXPECT` and `PASC_TABLE1_EXPECT` (`"168345,2102"`) refuse
to run on a cache whose size differs from the thesis population.

### Results layout

```
results/
  main/perc97/<window>/strict/iterations_<config>.csv     one row per repeat: AUROC, AUPRC, n features, settings
  main/perc97/<window>/strict/features_<config>_RF.json   features Boruta kept, per repeat
  main/perc97/<window>/strict/shap_<config>_RF.csv        mean |SHAP| per feature, per repeat
  main/perc97/all_iterations.csv                          all of the above concatenated
  main/perc97/paired_wilcoxon_bh.csv                      paired tests (scripts/paired_tests.py)
  lgbm_robustness/                                        run_lgbm_check.py, Table B.5
  calibration/<config>/                                   predictions (patient-level), metrics, summary
  symptom_groups/                                         run_symptom_groups.py
  descriptives/                                           Table 4.1, Table B.1
  figures/                                                make_thesis_figures.ipynb
```

`results/` and `cohort_parquets/` must stay out of version control: the prediction files and the
cohort tables are patient-level, and even the aggregate tables are derived from protected data
and fall under the data-use agreement of the site that produced them.

## Data access

The patient-level data cannot be shared. Access requires an appointment at the Icahn School of
Medicine at Mount Sinai and inclusion on the governing IRB protocol. The extract
analysed in the thesis was pulled from AIR·MS on 28 May 2026 (`pasc.cohort.combined.DATA_CUTOFF`),
against OMOP CDM v5.3 with vocabulary release v5.0 (27 February 2026).

Two site-specific definitions matter before porting:

* **Outcome label.** AIR·MS has no standard mapping for ICD-10 U09.9. The label uses two
  MSHS-local non-standard concepts, `condition_concept_id` 600588 and 600589, as a U09.9 proxy
  (`pasc/cohort/antony.py`, §3.3). Another site will use its own U09.9 concept.
* **Cohort entry.** A U07.1 diagnosis (concept 37311061) or a positive SARS-CoV-2 PCR / antigen
  result; the positive / negative `value_as_concept_id` sets were audited against the MSHS
  extract (`pasc/features/signals.py`).

## Porting to another OMOP CDM site

1. Point `.env` at your database and set `OMOP_CDM_SCHEMA`; replace the `hdbcli` connection in
   `pasc/db.py` if the CDM is not on HANA, and adapt the HANA SQL idioms.
2. Replace the two MSHS-local label concepts (600588, 600589) in `pasc/cohort/antony.py` with
   your site's U09.9 concept.
3. Check the concept catalogue in `docs/FEATURE_CODINGS_REFERENCE.md` against your vocabulary
   release; source-value `LIKE` patterns (smoking, alcohol, ICU care-site names, some assays)
   are Epic/MSHS conventions and will need local equivalents.
4. Run the concept-availability audit that `run_main_analysis.py` prints in step 4 before
   modelling: it lists which analytes and indicators exist in your extract at all.
5. Leave `PASC_LGBM_EXPECT` and `PASC_TABLE1_EXPECT` unset and set `CHECK_THESIS_VALUES = False`
   in the figure notebook; those checks pin the MSHS numbers.

## Development

`make test` runs the unit tests (configuration builder against Table 3.2, indicator catalogue
against Table 3.3, one seeded pipeline repeat on synthetic data, calibration measures, paired
tests), `make lint` runs ruff, and the GitHub Actions workflow does both on every push. None of
the tests touch a database.

## Provenance and scope

This is a cleaned release of the thesis repository. Exploratory tracks that the thesis does not
report (a confounder-pruning and leave-one-family-out analysis, treatment-guideline-concordance
features, a survival model, temporal cross-validation, an inpatient-only sensitivity run) and the
notebooks in which the feature definitions were first developed were removed; the feature
definitions themselves live in `pasc/features/`. Optional feature families that are empty in
the thesis configuration (`lab_trajectories`, `temporal_trends`, `cross_window_summaries`) and
the `relaxed` cohort mode remain in `pasc/features/enhanced.py` and `pasc/cohort/combined.py`
but are not used by any entry point.

Reporting follows TRIPOD+AI and STROBE/RECORD; the completed checklists are Appendix A.6 of the
thesis.

## Citation

See `CITATION.cff`.

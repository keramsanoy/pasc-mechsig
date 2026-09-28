# Predicting Post-Acute Sequelae of COVID-19 from Electronic Health Records: The Added Value of Mechanism-Based Feature Sets

Analysis code for the Master's thesis of the same title (Hasso Plattner Institute,
University of Potsdam, 2026). The study asks whether the three-cluster mechanism
framework of Greenhalgh et al. (viral persistence, immunoinflammatory dysregulation,
endothelial dysfunction) improves the prediction of a post-acute COVID-19 diagnosis when the
mechanisms are operationalised as features of the OMOP Common Data Model, measured against an
independent re-implementation of the Antony et al. (2023) acute-phase model and its extension
with routinely collected covariates, and whether any gain depends on the feature-extraction
window.

The code was run against AIR·MS, the OMOP CDM v5.3 implementation of the Mount Sinai Health
System. **No data are in this repository and none can be shared**: the analysis used protected
health information under an IRB-approved protocol inside the MSHS enclave. What is released
is enough for someone with that access, or with another OMOP CDM site, to repeat the analysis;
it is not enough to do so without a database. See [Data access](#data-access).

## Contents

| Path | Role in the thesis |
|---|---|
| `rebuild_combined_cohort.py` | Builds the cohort table: COVID-19-positive patients, PASC label, follow-up, fixed data cutoff (§3.2, §3.3) |
| `run_enhanced_mechsig.py` | **Main entry point** (Appendix A.5): seven configurations × six windows × 100 repeats, Random Forest, Boruta, SHAP, plus the `_no_td` / `_no_eng` refits (§3.4–§3.10) |
| `run_lgbm_final.py` | LightGBM robustness check on the same divisions (§3.6.1, §4.3.3, Table B.5) |
| `run_calibration_configs.py` | Calibration, logistic recalibration, held-out predictions for the decision curve (§3.7, §4.6, Table B.11, Figure B.1) |
| `run_subtype_multilabel_aligned.py` | Secondary analysis S1: symptom-group models at w0-90 (§3.11, §4.7, Table B.12, Table A.4) |
| `make_consort_enhanced_mechsig.py` | Patient-flow diagram of the modelled population (Figure 4.1) |
| `scripts/make_table1.py` | Cohort characteristics with standardised mean differences (Table 4.1) and laboratory missingness by window (Table B.1) |
| `scripts/paired_wilcoxon_bh.py` | Paired Wilcoxon signed-rank tests with Benjamini–Hochberg correction for every comparison (§3.8; Tables 4.3, B.2–B.4) |
| `scripts/summarize_lgbm_table.py` | Table B.5 from the LightGBM and Random Forest repeats |
| `make_thesis_figures.ipynb` | Figures 4.2–4.7, B.1 and the appendix tables on feature selection and net benefit, from the result files only |
| `antony_cohort.py`, `antony_features.py`, `antony_pipeline.py` | Cohort, feature families A–E and modelling protocol of the Antony et al. replication (Table 3.1, Table 3.5) |
| `combined_cohort.py`, `combined_features.py` | Cohort with follow-up and death handling; extended covariates and engagement controls (§3.4.1) |
| `indicator_definitions.py`, `mech_signals_common.py` | The 70 binary mechanism indicators and the SQL that extracts them (Tables A.1–A.3) |
| `enhanced_mech_signals.py` | Laboratory summaries, CBC indices, composites, temporal-divergence composites with the leakage gate, and the configuration builder (§3.4.2, §3.5, Table 3.2) |
| `omop2obo_mapping.py` | OMOP2OBO mapping of acute symptoms to HPO terms (§3.4.1) |
| `calibration.py` | Calibration measures and recalibration methods (Appendix A.3) |
| `init.py`, `omop_config.py`, `pasc_paths.py` | Database connection, schema name, repository-relative paths |
| `hpc/` | LSF submit scripts used for the per-window runs |
| `docs/FEATURE_CODINGS_REFERENCE.md` | Every OMOP concept ID, LOINC code, RxNorm ingredient, SNOMED ancestor and source-value pattern the pipeline uses |

The directories `cohort_parquets/` (cohort table and cached feature matrices) and `results/`
(every output) are created at run time. Both hold patient-level data or data-derived outputs and
are git-ignored.

## Data access

The patient-level data cannot be shared. Access requires an appointment at the Icahn School of
Medicine at Mount Sinai and inclusion on the governing IRB protocol. The extract
analysed in the thesis was pulled from AIR·MS on 28 May 2026 (`combined_cohort.DATA_CUTOFF`),
against OMOP CDM v5.3 with vocabulary release v5.0 (27 February 2026).

Two site-specific definitions are worth knowing before porting the code:

* **Outcome label.** AIR·MS has no standard mapping for ICD-10 U09.9. The label uses two
  MSHS-local non-standard concepts, `condition_concept_id` 600588 and 600589, as a U09.9 proxy
  (`antony_cohort.py`, §3.3). Another site will use its own U09.9 concept.
* **Cohort entry.** A U07.1 diagnosis (concept 37311061) or a positive SARS-CoV-2 PCR / antigen
  result; the positive / negative `value_as_concept_id` sets were audited against the MSHS
  extract (`mech_signals_common.py`).

## Setting up

```bash
conda env create -f environment.yml && conda activate pasc      # or: pip install -r requirements.txt
cp .env.example .env                                            # database host, port, user, schema
```

Library versions matter here more than usual: the forest, imputer, splitting and grid search come
from scikit-learn 1.8.0, feature selection from Boruta 0.4.3, attributions from shap 0.49.1 and
the robustness check from LightGBM 4.6.0 (Python 3.12). Given the same code, data and seeds a run
reproduces exactly; a different scikit-learn release can fit different forests.

The acute-symptom family maps diagnosis codes to HPO terms through OMOP2OBO v2.0.0 (Callahan et
al., Zenodo [10.5281/zenodo.7255922](https://doi.org/10.5281/zenodo.7255922)). Download
`OMOP2OBO_v2.0.0_N3C_Enclave_CSV_concept_set_expression_items.csv` from that record into
`data/omop2obo/`.

`init.py` connects to SAP HANA through `hdbcli`, optionally over an SSH tunnel, with every
host, port, credential and schema read from the environment (`.env.example` lists them). The SQL
names the CDM schema literally as `CDMPHI`, the AIR·MS convention; setting `OMOP_CDM_SCHEMA`
substitutes another name at execution time. The SQL is HANA-flavoured (`#temp` tables,
`ADD_DAYS`, `TOP n`); another database engine needs the templates in `mech_signals_common.py`,
`antony_cohort.py`, `antony_features.py` and `combined_features.py` adapted.

## Reproducing the analysis

Every step below runs from the repository root. Steps 1–2 need the database; everything after
runs from the cached feature matrices.

```bash
# 1. Cohort table -> cohort_parquets/combined_strict_all_patients.parquet
python rebuild_combined_cohort.py

# 2. Feature matrices for the six windows (needs the database), then the models.
#    One call does both; on a cluster, extract once on a node with database access
#    and model per window on compute nodes (hpc/submit_window.sh).
PASC_REEXTRACT=1 PASC_EXTRACT_ONLY=1 python run_enhanced_mechsig.py
PASC_REEXTRACT=0 python run_enhanced_mechsig.py
#    -> results/main/perc97/<window>/strict/{iterations,features,shap}_<config>_RF.{csv,json}
#       and results/main/perc97/all_iterations.csv

# 3. Paired tests for every comparison (H2, H3/H4 plain, no_td, no_eng, single clusters)
python scripts/paired_wilcoxon_bh.py           # -> results/main/perc97/paired_wilcoxon_bh.csv

# 4. LightGBM robustness check (Table B.5)
python run_lgbm_final.py                        # -> results/lgbm_robustness/, runs summarize_lgbm_table.py

# 5. Calibration of both primary configurations at w0-90 (Table B.11, Figure B.1)
PASC_CALIB_CONFIG=baseline_ext python run_calibration_configs.py
PASC_CALIB_CONFIG=mechsig_all  python run_calibration_configs.py

# 6. Symptom groups (S1): one extraction pass with the database, then the models
PASC_REEXTRACT=1 PASC_EXTRACT_ONLY=1 PASC_FEATURE_WINDOWS=0_90 python run_subtype_multilabel_aligned.py
PASC_REEXTRACT=0 PASC_FEATURE_WINDOWS=0_90 python run_subtype_multilabel_aligned.py

# 7. Descriptives and figures
python scripts/make_table1.py                   # Table 4.1, Table B.1 (queries race/ethnicity once)
python make_consort_enhanced_mechsig.py         # Figure 4.1
jupyter nbconvert --execute make_thesis_figures.ipynb --to notebook   # Figures 4.2-4.7, B.1
```

### What one run of `run_enhanced_mechsig.py` does

For every window in `FEATURE_WINDOWS` (w0-21, w0-30, w0-60, w0-90, w30-60, w60-90) the script
re-extracts every feature inside that window: the Antony et al. families A–E, the six blocks of
extended covariates, the six engagement controls, the binary indicators of the three clusters,
the six-column laboratory summaries, the CBC-derived indices, the composites, and the
temporal-divergence composites gated by `max_window_end_day`. The matrix is cached as
`cohort_parquets/enhanced_<window>_strict_all_patients.parquet` with a `families.json` beside it.

Each configuration in `ACTIVE_CONFIGS` (the seven primary ones of Table 3.2 and the `_no_td` /
`_no_eng` refits of §3.9) is then fitted 100 times with `antony_pipeline.run_full_pipeline`.
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

`run_lgbm_final.py`, `run_calibration_configs.py` and `run_subtype_multilabel_aligned.py`
document their own switches in their headers. `PASC_LGBM_EXPECT` and `PASC_TABLE1_EXPECT`
(`"168345,2102"`) refuse to run on a cache whose size differs from the thesis population.

### Results layout

```
results/
  main/perc97/<window>/strict/iterations_<config>.csv     one row per repeat: AUROC, AUPRC, n features, settings
  main/perc97/<window>/strict/features_<config>_RF.json   features Boruta kept, per repeat
  main/perc97/<window>/strict/shap_<config>_RF.csv        mean |SHAP| per feature, per repeat
  main/perc97/all_iterations.csv                          all of the above concatenated
  main/perc97/paired_wilcoxon_bh.csv                      paired tests (scripts/paired_wilcoxon_bh.py)
  lgbm_robustness/                                        run_lgbm_final.py, Table B.5
  calibration/<config>/                                   predictions (patient-level), metrics, summary
  symptom_groups/                                         run_subtype_multilabel_aligned.py
  descriptives/                                           Table 4.1, Table B.1
  figures/                                                make_thesis_figures.ipynb
```

`results/` and `cohort_parquets/` must stay out of version control: the prediction files and the
cohort tables are patient-level, and even the aggregate tables are derived from protected data
and fall under the data-use agreement of the site that produced them.

## Porting to another OMOP CDM site

1. Point `.env` at your database and set `OMOP_CDM_SCHEMA`; replace the `hdbcli` connection in
   `init.py` if the CDM is not on HANA, and adapt the HANA SQL idioms.
2. Replace the two MSHS-local label concepts (600588, 600589) in `antony_cohort.py` with your
   site's U09.9 concept.
3. Check the concept catalogue in `docs/FEATURE_CODINGS_REFERENCE.md` against your vocabulary
   release; source-value `LIKE` patterns (smoking, alcohol, ICU care-site names, some assays)
   are Epic/MSHS conventions and will need local equivalents.
4. Run the concept-availability audit that `run_enhanced_mechsig.py` prints in step 4 before
   modelling: it lists which analytes and indicators exist in your extract at all.
5. Leave `PASC_LGBM_EXPECT` and `PASC_TABLE1_EXPECT` unset and set `CHECK_THESIS_VALUES = False`
   in the figure notebook; those checks pin the MSHS numbers.

## Provenance and scope

This is a cleaned release of the thesis repository. Exploratory tracks that the thesis does not
report (a confounder-pruning and leave-one-family-out analysis, treatment-guideline-concordance
features, a survival model, temporal cross-validation, an inpatient-only sensitivity run) and the
notebooks in which the feature definitions were first developed were removed; the feature
definitions themselves live in the modules listed above. Optional feature families that are empty
in the thesis configuration (`lab_trajectories`, `temporal_trends`, `cross_window_summaries`) and
the `relaxed` cohort mode remain in `enhanced_mech_signals.py` and `combined_cohort.py` but are
not used by any entry point.

Reporting follows TRIPOD+AI and STROBE/RECORD; the completed checklists are Appendix A.6 of the
thesis.

## Citation

See `CITATION.cff`.

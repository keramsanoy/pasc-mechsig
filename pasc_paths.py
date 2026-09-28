"""Repository-relative paths shared by every entry point and script.

All locations can be moved off the repository tree with two environment
variables, which is what you want on an HPC system where the cohort cache
and the results are large:

    PASC_COHORT_DIR   directory for the cohort table and the cached per-window
                      feature matrices (default: <repo>/cohort_parquets)
    PASC_RESULTS_DIR  directory for every output (default: <repo>/results)

Both directories hold patient-level data or data-derived outputs. They are
git-ignored and must never be committed.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

COHORT_DIR = Path(os.environ.get("PASC_COHORT_DIR", REPO_ROOT / "cohort_parquets"))
RESULTS_DIR = Path(os.environ.get("PASC_RESULTS_DIR", REPO_ROOT / "results"))

# Output of run_enhanced_mechsig.py: perc<PERC>/<window>/strict/{iterations,features,shap}_*
MAIN_RESULTS_DIR = RESULTS_DIR / "main"
# Output of run_lgbm_final.py and scripts/summarize_lgbm_table.py (thesis Table B.5)
LGBM_RESULTS_DIR = RESULTS_DIR / "lgbm_robustness"
# Output of run_calibration_configs.py (thesis 4.6, Table B.11, Figure B.1)
CALIBRATION_RESULTS_DIR = RESULTS_DIR / "calibration"
# Output of run_subtype_multilabel_aligned.py (thesis 4.7, Table B.12)
SYMPTOM_GROUP_RESULTS_DIR = RESULTS_DIR / "symptom_groups"
# Output of scripts/make_table1.py (thesis Table 4.1, Table B.1)
DESCRIPTIVES_DIR = RESULTS_DIR / "descriptives"
# Output of make_thesis_figures.ipynb
FIGURES_DIR = RESULTS_DIR / "figures"

DATA_DIR = REPO_ROOT / "data"
OMOP2OBO_DIR = DATA_DIR / "omop2obo"


def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

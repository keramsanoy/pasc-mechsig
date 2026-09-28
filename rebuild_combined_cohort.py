#!/usr/bin/env python
"""Build the cohort table (thesis 3.2): cohort_parquets/combined_strict_all_patients.parquet.

One row per COVID-19-positive patient (U07.1 diagnosis or positive SARS-CoV-2 test;
antony_cohort.py) with the PASC label (MSHS-local U09.9 proxy concepts, thesis 3.3),
the index date, discharge date, death bucket, end of observation and observed
follow-up, all computed against the fixed ascertainment horizon
combined_cohort.DATA_CUTOFF (28 May 2026, the date of the extract). The
modelling-time gates (no PASC code < 90 d, >= 365 d follow-up for controls) are
applied later, in run_enhanced_mechsig.py.

Needs the database (init.py prompts for credentials):
    python rebuild_combined_cohort.py

Then extract the feature matrices and fit the models:
    PASC_REEXTRACT=1 python run_enhanced_mechsig.py
"""

import os
import sys

from pasc_paths import REPO_ROOT, COHORT_DIR

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# init.py opens the database connection (prompts for credentials interactively).
exec(open(os.path.join(REPO_ROOT, "init.py")).read())
cur = hana_conn.cursor()
print(f"Connection active: {hana_conn.isconnected()}")

from combined_cohort import build_combined_cohorts, DATA_CUTOFF

print(f"\nRebuilding strict base cohort with data_cutoff = {DATA_CUTOFF.date()}")
cohorts = build_combined_cohorts(
    cur,
    mode="strict",
    run_all_patients=True,
    run_inpatients=False,
    run_outpatients=False,
    acute_days=21,           # base acute window; run_enhanced_mechsig recomputes per feature window
    data_cutoff=DATA_CUTOFF,
    save_dir=str(COHORT_DIR),
)

df = cohorts["all_patients"]
print(f"\nSaved combined_strict_all_patients.parquet: {len(df):,} patients "
      f"(pos={int(df['label'].sum()):,}, neg={len(df) - int(df['label'].sum()):,})")

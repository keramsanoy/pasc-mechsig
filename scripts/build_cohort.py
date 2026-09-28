#!/usr/bin/env python
"""Build the cohort table (thesis 3.2): cohort_parquets/combined_strict_all_patients.parquet.

One row per COVID-19-positive patient (U07.1 diagnosis or positive SARS-CoV-2 test;
pasc.cohort.antony) with the PASC label (MSHS-local U09.9 proxy concepts, thesis 3.3),
the index date, discharge date, death bucket, end of observation and observed
follow-up, all computed against the fixed ascertainment horizon
pasc.cohort.combined.DATA_CUTOFF (28 May 2026, the date of the extract). The
modelling-time gates (no PASC code < 90 d, >= 365 d follow-up for controls) are
applied later, in scripts/run_main_analysis.py.

Needs the database (pasc.db.connect prompts for credentials):
    python scripts/build_cohort.py

Then extract the feature matrices and fit the models:
    PASC_REEXTRACT=1 python scripts/run_main_analysis.py
"""

import _bootstrap  # noqa: F401  (makes `pasc` importable from a plain clone)
from pasc.config.paths import COHORT_DIR
from pasc.db import connect

# Opens the database connection (prompts for credentials interactively).
hana_conn = connect()
cur = hana_conn.cursor()
print(f"Connection active: {hana_conn.isconnected()}")

from pasc.cohort.combined import build_combined_cohorts, DATA_CUTOFF

print(f"\nRebuilding strict base cohort with data_cutoff = {DATA_CUTOFF.date()}")
cohorts = build_combined_cohorts(
    cur,
    mode="strict",
    run_all_patients=True,
    run_inpatients=False,
    run_outpatients=False,
    acute_days=21,           # base acute window; run_main_analysis recomputes per feature window
    data_cutoff=DATA_CUTOFF,
    save_dir=str(COHORT_DIR),
)

df = cohorts["all_patients"]
print(f"\nSaved combined_strict_all_patients.parquet: {len(df):,} patients "
      f"(pos={int(df['label'].sum()):,}, neg={len(df) - int(df['label'].sum()):,})")

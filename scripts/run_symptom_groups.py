#!/usr/bin/env python3
"""
Secondary analysis S1: symptom-group models (thesis 3.11, 4.7, Table B.12, Appendix A.2).

Assigns PASC-coded patients to the five symptom groups of Liew et al. (2024)
(cardiorespiratory, fatigue, cognitive, anxiety/depression, gastrointestinal) by
matching condition concepts recorded between day 90 and day 540 after the index
date against per-group OMOP ancestor concepts and concept-name patterns
(SYMPTOM_GROUPS below = thesis Table A.4). One model per group is then
fitted at w0-90 with the same pipeline, estimator and 100 seeded repeats as the
main run, for the configurations in ACTIVE_CONFIGS; the features that define a
group are withheld from the model that predicts it (SUBTYPE_FEATURE_BLOCKLIST).

Shares the modelling-time gates of scripts/run_main_analysis.py: index <= cutoff,
no PASC code < 90 d, >= 365 d follow-up for controls (PASC_MIN_FOLLOWUP_DAYS),
calendar feature withheld (PASC_DROP_INDEX_CALENDAR), Boruta perc 97.

Inputs : cohort_parquets/combined_strict_all_patients.parquet (scripts/build_cohort.py)
Outputs: results/symptom_groups/perc97/w0_90/<config>/iterations_<group>.csv
         results/symptom_groups/all_iterations.csv, subtype_labels.parquet (patient-level)

Run:
    python scripts/run_symptom_groups.py
    PASC_REEXTRACT=0 python scripts/run_symptom_groups.py   # reuse cached matrices, no DB
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")

import os

# =============================================================================
# 0 — Re-extraction toggle
# =============================================================================
# When True (default), connect to the DB and extract labels + per-window feature
# matrices, saving each to COHORT_DIR. When False, skip the DB entirely and load
# the previously-saved subtype_enhanced_{wl}_{mode}.parquet / _families.json.
REEXTRACT_FEATURES = os.environ.get("PASC_REEXTRACT", "1") == "1"
print(f"[0] REEXTRACT_FEATURES = {REEXTRACT_FEATURES} "
      f"(set PASC_REEXTRACT=0 to reuse cached feature parquets)")

# =============================================================================
# 1 — DB Connection
# =============================================================================
import _bootstrap  # noqa: F401  (makes `pasc` importable from a plain clone)
from pasc.config.paths import COHORT_DIR as _COHORT_DIR, SYMPTOM_GROUP_RESULTS_DIR

if REEXTRACT_FEATURES:
    from pasc.db import connect
    hana_conn = connect()
    cur = hana_conn.cursor()
    print(f"Connection active: {hana_conn.isconnected()}")
else:
    print("[1] Skipped DB init (REEXTRACT_FEATURES=False).")
    cur = None

# =============================================================================
# 2 — Imports
# =============================================================================
import json
import pandas as pd
import numpy as np

from pasc.cohort.antony import upload_antony_cohort_temp, compute_acute_window
from pasc.features.extended import build_combined_feature_matrix
from pasc.modeling.pipeline import run_full_pipeline, results_to_dataframe
from pasc.features.signals import run_query, _ints_to_sql_in
from pasc.cohort.combined import DATA_CUTOFF

from pasc.features.indicators import (
    VIRAL_INDICATOR_LIST, IMMUNO_INDICATOR_LIST, ENDO_INDICATOR_LIST,
)

from pasc.features.enhanced import (
    window_label,
    make_window_cohort,
    extract_binary_signals_for_window,
    extract_numeric_labs,
    extract_cbc_indices,
    build_composite_features,
    build_temporal_divergence_composites,
    build_enhanced_model_configs,
    assemble_enhanced_feature_matrix,
    NUMERIC_LAB_SPECS, ENH_VIRAL_INDICATOR_LIST,
    ENH_IMMUNO_INDICATOR_LIST,
    ENH_ENDO_INDICATOR_LIST,
    ENH_EXPLORATORY_INDICATOR_LIST,
)

print("All imports successful.")

# =============================================================================
# 3 — Configuration
# =============================================================================

BASELINE_ACUTE_DAYS = 21

PASC_MIN_DAYS = 90

FEATURE_WINDOWS = [
    # (0, 21),   # Antony-matching
    # (0, 30),   # One month
    # (0, 60),   # Two months
     (0, 90),   # Three months
    # (30, 60),  # Late only: month 2
    # (60, 90),  # Late only: month 3
]

# Optional override to run a subset of windows without editing the list above.
# Format: comma-separated "start_end" pairs, e.g. PASC_FEATURE_WINDOWS="0_90"
# or "0_60,30_60". Enables one LSF job per window (see submit_subtype_window.sh).
_fw_override = os.environ.get("PASC_FEATURE_WINDOWS", "").strip()
if _fw_override:
    FEATURE_WINDOWS = [
        tuple(int(x) for x in pair.split("_"))
        for pair in _fw_override.split(",") if pair.strip()
    ]
    print(f"[feature_windows] override via PASC_FEATURE_WINDOWS -> {FEATURE_WINDOWS}")

# Heavy modeling budget (matches run_main_analysis.py).
N_ITERATIONS        = 100
BORUTA_MAX_ITER     = 50
BORUTA_N_ESTIMATORS = 500
COMPUTE_SHAP        = True
RANDOM_STATE_BASE   = 42
MODEL_TYPES         = ("RF",)

# Boruta perc sweep -- one results/.../perc{perc}/ subdir per value (non-destructive).
BORUTA_PERC_SWEEP = [97]

# --- Confirmed-negative ascertainment gate --------------------------------
# A control only counts as a negative if it was observed (followup_days) for at
# least this long without a PASC diagnosis. Controls below the gate are dropped
# (not labelled negative); cases are never affected.
#   PASC_MIN_FOLLOWUP_DAYS=365 python scripts/run_symptom_groups.py
MIN_FOLLOWUP_DAYS = int(os.environ.get("PASC_MIN_FOLLOWUP_DAYS", "365"))

# --- Withhold the index-date calendar feature from MODELING only -----------
# A follow-up/ascertainment proxy (see run_main_analysis.py, thesis 3.6). The
# saved feature parquet keeps it, so the toggle is reversible with no re-extract.
#   PASC_DROP_INDEX_CALENDAR=0 python scripts/run_symptom_groups.py  # keep
DROP_INDEX_CALENDAR = os.environ.get("PASC_DROP_INDEX_CALENDAR", "1") == "1"
_CALENDAR_DROP_COLS = [
    "f_ext_index_year_month",
]

# Toggle individual stages. Label + feature extraction require the DB, so they
# only run when REEXTRACT_FEATURES is True.
RUN_LABEL_EXTRACTION   = REEXTRACT_FEATURES
RUN_FEATURE_EXTRACTION = REEXTRACT_FEATURES
RUN_MODELING           = True

# Resume behaviour (PASC_SKIP_EXISTING mirrors run_main_analysis.py).
#   1 -> Skip any (perc, window, config, subtype) quad whose iterations CSV
#        already exists under RESULTS_DIR (resume after a timeout / partial run).
#   0 -> Fresh run. Existing CSVs are overwritten as quads complete; no skip.
RESUME_FROM_LAST = os.environ.get("PASC_SKIP_EXISTING", "0") == "1"

COHORT_MODE = "strict"

LABEL_WINDOW_START_DAYS = PASC_MIN_DAYS
LABEL_WINDOW_END_DAYS   = 540

ACTIVE_CONFIGS = {
    "baseline_ext",
    "mechsig_viral",
    "mechsig_immuno",
    "mechsig_endo",
    "mechsig_all",
}

COHORT_DIR  = str(_COHORT_DIR)
RESULTS_DIR = str(SYMPTOM_GROUP_RESULTS_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)

from pasc.config.omop import CDM_SCHEMA as CDM  # OMOP schema name; set OMOP_CDM_SCHEMA to override

print("Configuration:")
print(f"  PASC_MIN_DAYS:         {PASC_MIN_DAYS}")
print(f"  BASELINE_ACUTE_DAYS:   {BASELINE_ACUTE_DAYS}")
print(f"  FEATURE_WINDOWS:       {FEATURE_WINDOWS}")
print(f"  LABEL_WINDOW:          day {LABEL_WINDOW_START_DAYS}-{LABEL_WINDOW_END_DAYS}")
print(f"  N_ITERATIONS:          {N_ITERATIONS}")
print(f"  BORUTA_N_ESTIMATORS:   {BORUTA_N_ESTIMATORS}")
print(f"  BORUTA_PERC_SWEEP:     {BORUTA_PERC_SWEEP}")
print(f"  MODEL_TYPES:           {MODEL_TYPES}")
print(f"  COHORT_MODE:           {COHORT_MODE}")
print(f"  MIN_FOLLOWUP_DAYS:     {MIN_FOLLOWUP_DAYS}")
print(f"  DROP_INDEX_CALENDAR:   {DROP_INDEX_CALENDAR}")
print(f"  RESULTS_DIR:           {RESULTS_DIR}")
print(f"  RESUME_FROM_LAST:      {RESUME_FROM_LAST}")
print("")
print("  NOTE: resume logic skips (perc, window, config, subtype) quads whose")
print("        per-run CSVs already exist. If you changed SUBTYPE_FEATURE_BLOCKLIST,")
print(f"        delete {RESULTS_DIR}/perc*/w*/ to force regeneration.")

# =============================================================================
# 4 — Symptom-Group Label Definitions (Liew 2024 Partition)
# =============================================================================

SYMPTOM_GROUPS = {
    "cardiorespiratory": {
        "ancestor_ids": (
            312437,   # Dyspnea
            254761,   # Cough
            77670,    # Chest pain
            255573,   # Chronic obstructive pulmonary disease
            444070,   # Tachycardia (SNOMED 3424008) — covers sinus tach, SVT descendants
            315078,   # Palpitations (SNOMED 80313002) — proper ancestor for palpitations
            # removed 314379 (First degree AV block) — cardiac conduction, wrong axis
            # removed 4183748 (Kidney/ureteral surgical margin involved by tumor) — cancer pathology, unrelated
        ),
        "name_patterns": (
            "%DYSPNEA%", "%BREATHLESSNESS%", "%CHRONIC COUGH%", "%CHEST PAIN%",
            "%PALPITATION%", "%TACHYCARDIA%",
            "%INTERSTITIAL LUNG%", "%EXERTIONAL BREATHLESSNESS%",
            "%SHORTNESS OF BREATH%", "%PLEURITIC%",
        ),
    },
    "fatigue": {
        "ancestor_ids": (
            4223659,  # Fatigue
            439926,   # Malaise and fatigue
            4272240,  # Malaise
            # removed 254058 (Acute bronchiolitis caused by RSV) — respiratory infection, wrong axis
            # removed 4272472 (Anesthesia for omphalocele) — INVALID, surgical procedure
        ),
        "name_patterns": (
            "%FATIGUE%", "%MALAISE%", "%WEAKNESS%", "%ASTHENIA%",
            "%POST%EXERTIONAL%", "%POST-VIRAL FATIGUE%", "%CHRONIC FATIGUE%",
            "%LASSITUDE%", "%LETHARGY%", "%TIREDNESS%",
        ),
    },
    "anxiety_depression": {
        "ancestor_ids": (
            441542,   # Anxiety
            440383,   # Depressive disorder
            4152280,  # Major depressive disorder
            436676,   # Posttraumatic stress disorder
            4077577,  # Moderate recurrent major depression
            # removed 4212540 (Chronic liver disease) — same id as f_comor_liver_mild,
            #   created a data leak: every CLD patient was getting the anxiety/depression label
        ),
        "name_patterns": (
            "%ANXIETY%", "%DEPRESSION%", "%DEPRESSIVE%",
            "%POST%TRAUMATIC STRESS%", "%PTSD%", "%ADJUSTMENT DISORDER%",
            "%MOOD DISORDER%", "%MOOD DISTURBANCE%", "%DYSTHYMIA%",
        ),
    },
    "gastrointestinal": {
        "ancestor_ids": (
            196523,   # Diarrhea
            27674,    # Nausea and vomiting
            200219,   # Abdominal pain
            # removed 4091513 (Passing flatus) — normal physiology, not a symptom
            # removed 4201758 (Target arrow) — non-standard SNOMED Physical Object, garbage match
            # removed 75580 (Chronic ulcerative proctitis) — narrow IBD subtype, not a GI symptom umbrella
        ),
        "name_patterns": (
            "%DIARRHEA%", "%DIARRHOEA%", "%NAUSEA%", "%ABDOMINAL PAIN%",
            "%IRRITABLE BOWEL%", "%ALTERED BOWEL%", "%LOSS OF APPETITE%",
            "%ANOREXIA%", "%VOMITING%", "%GASTROPARESIS%",
        ),
    },
    "cognitive": {
        "ancestor_ids": (
            373995,    # Delirium
            374009,    # Organic mental disorder
            40480615,  # Cognitive disorder (SNOMED 443265004) — proper umbrella
            4304008,   # Memory impairment (SNOMED 386807006) — explicit memory coverage
            # removed 4178628 (Sepsis due to infected CVC) — INVALID/deprecated, also wrong axis
            # removed 436235 (Taste sense altered) — chemosensory, not cognitive
        ),
        "name_patterns": (
            "%COGNITIVE%IMPAIRMENT%", "%COGNITIVE%DYSFUNCTION%",
            "%MEMORY%IMPAIRMENT%", "%MEMORY%DEFICIT%", "%MEMORY%LOSS%",
            "%BRAIN FOG%", "%ATTENTION%DEFICIT%", "%CONCENTRATION%IMPAIRMENT%",
            "%CONCENTRATION%DIFFICULTY%", "%COGNITIVE%DECLINE%",
            "%POST%COVID%COGNITIVE%", "%AMNESIA%",
        ),
    },
}

SUBTYPE_NAMES = list(SYMPTOM_GROUPS.keys())
print(f"Defined {len(SUBTYPE_NAMES)} symptom groups: {SUBTYPE_NAMES}")

# Per-subtype blocklist: every feature column whose concept restates the
# subtype's label definition (SYMPTOM_GROUPS above). Entries are matched as
# exact column names OR as regex patterns (prefix "re:"), so window-prefixed
# indicator columns (f_ind_enh_w0_90_...) are caught in every window.
SUBTYPE_FEATURE_BLOCKLIST = {
    "cardiorespiratory": [
        "f_sym_hpo_dyspnea",
        "f_sym_hpo_cough",
        "f_sym_hpo_chest_pain",
        "f_ext_sym_palpitations",          # ancestors 315078 + 77670, both in the crosswalk
        "f_ext_sym_tachycardia",
        "f_comor_chronic_pulm",            # COPD ancestor 255573 is in the crosswalk
        r"re:^f_ind_enh_w\d+_\d+_viral_resp_dx_any$",
        r"re:^f_comp_td_post_acute_respiratory(_n_windows)?$",   # built only from resp_dx_any
    ],
    "fatigue": [
        "f_sym_hpo_fatigue",
        "f_sym_hpo_malaise",
    ],
    "anxiety_depression": [
        "f_comor_anxiety",
        "f_comor_depression",
    ],
    "gastrointestinal": [
        "f_sym_hpo_nausea",
        "f_sym_hpo_vomiting",
        "f_sym_hpo_diarrhea",
        "f_sym_hpo_abdominal_pain",
        r"re:^f_ind_enh_w\d+_\d+_viral_gi_dx_any$",
        "f_comp_gi_any",
        "f_comp_gi_burden_score",
        "f_comp_gi_abdominal_pain",        # ancestor 200219 is in the crosswalk
        "f_comp_gi_chronic_diarrhea",      # ancestor 196523 is in the crosswalk
        # f_comp_gi_ppi / _endoscopy / _biopsy / _antidiarrheal are kept: treatment and
        # procedure proxies, not symptom concepts of the label.
    ],
    "cognitive": [
        "f_ext_sym_neurocog",
        "f_comor_dementia",
    ],
}
for _g in SUBTYPE_NAMES:
    assert _g in SUBTYPE_FEATURE_BLOCKLIST, f"Missing blocklist entry for {_g}"
print("SUBTYPE_FEATURE_BLOCKLIST configured.")

import re as _re

def resolve_blocklist(subtype: str, columns: list[str]) -> list[str]:
    """Return the columns to drop for this subtype. Every blocklist entry must
    hit at least one column, otherwise raise: a silent miss is exactly the bug
    this guard had before."""
    entries = SUBTYPE_FEATURE_BLOCKLIST[subtype]
    dropped, misses = [], []
    for e in entries:
        if e.startswith("re:"):
            hits = [c for c in columns if _re.search(e[3:], c)]
        else:
            hits = [c for c in columns if c == e]
        if hits:
            dropped.extend(hits)
        else:
            misses.append(e)
    if misses:
        raise RuntimeError(
            f"[blocklist] {subtype}: no column matched {misses}. "
            f"Fix the entry or the feature table before running."
        )
    return sorted(set(dropped))

# Optional subset of subtypes to MODEL (labels for all subtypes are still
# extracted). Enables one LSF job per subtype (see submit_subtype_window.sh).
#   PASC_SUBTYPES=cognitive python scripts/run_symptom_groups.py
_subtype_override = os.environ.get("PASC_SUBTYPES", "").strip()
if _subtype_override:
    MODEL_SUBTYPES = [s.strip() for s in _subtype_override.split(",") if s.strip()]
    _unknown = [s for s in MODEL_SUBTYPES if s not in SUBTYPE_NAMES]
    if _unknown:
        raise ValueError(f"Unknown PASC_SUBTYPES {_unknown}; valid: {SUBTYPE_NAMES}")
    print(f"[subtypes] modeling restricted to {MODEL_SUBTYPES}")
else:
    MODEL_SUBTYPES = list(SUBTYPE_NAMES)

# =============================================================================
# 5 — Cohort Loading & PASC Label Enforcement
# =============================================================================

raw_cohort_path = os.path.join(COHORT_DIR, f"combined_{COHORT_MODE}_all_patients.parquet")
if not os.path.exists(raw_cohort_path):
    raise FileNotFoundError(f"Raw cohort parquet not found: {raw_cohort_path}")

raw_cohort = pd.read_parquet(raw_cohort_path)
raw_cohort["covid_index_date"] = pd.to_datetime(raw_cohort["covid_index_date"])
raw_cohort["discharge_date"] = pd.to_datetime(raw_cohort["discharge_date"], errors="coerce")
raw_cohort["first_pacs_date"] = pd.to_datetime(raw_cohort["first_pacs_date"], errors="coerce")
print(f"Loaded raw cohort: {len(raw_cohort):,} patients from {raw_cohort_path}")

# --- Fixed data-cutoff enforcement (ascertainment horizon) -------------------
# Drop future-dated index (data artifacts past the pull date) and re-cap
# follow-up at the cutoff (mirrors combined_cohort.build_combined_cohorts).
n_pre_cut = len(raw_cohort)
raw_cohort = raw_cohort[raw_cohort["covid_index_date"] <= DATA_CUTOFF].copy()
print(f"\nData cutoff enforcement ({DATA_CUTOFF.date()}):")
print(f"  Dropped patients with index date > cutoff: {n_pre_cut - len(raw_cohort):,}")
if {"obs_end", "death_date"}.issubset(raw_cohort.columns):
    for _c in ("obs_end", "death_date"):
        raw_cohort[_c] = pd.to_datetime(raw_cohort[_c], errors="coerce")
    _cut = pd.Series(DATA_CUTOFF, index=raw_cohort.index)
    raw_cohort["last_obs_date"] = pd.concat(
        [raw_cohort["obs_end"], raw_cohort["death_date"], _cut], axis=1
    ).min(axis=1)
    raw_cohort["followup_days"] = (
        raw_cohort["last_obs_date"] - raw_cohort["covid_index_date"]
    ).dt.days
    print("  Re-capped last_obs_date / followup_days at cutoff")
elif "followup_days" in raw_cohort.columns:
    _max_fu = (DATA_CUTOFF - raw_cohort["covid_index_date"]).dt.days
    raw_cohort["followup_days"] = np.minimum(raw_cohort["followup_days"], _max_fu)
    print("  Re-capped followup_days at cutoff (fallback: no obs_end/death_date)")

days_to_pasc = (raw_cohort["first_pacs_date"] - raw_cohort["covid_index_date"]).dt.days
too_early_mask = raw_cohort["first_pacs_date"].notna() & (days_to_pasc < PASC_MIN_DAYS)
n_dropped = int(too_early_mask.sum())
print(f"\nPASC label enforcement (>= {PASC_MIN_DAYS}d post-COVID):")
print(f"  Patients with first PASC dx < {PASC_MIN_DAYS}d post-COVID "
      f"(acute/prevalent) DROPPED (not counted as controls): {n_dropped:,}")
raw_cohort = raw_cohort[~too_early_mask].copy()
print(f"  Label distribution after enforcement: {raw_cohort['label'].value_counts().to_dict()}")

# --- Confirmed-negative ascertainment gate -----------------------------------
# Drop controls not observed long enough to confidently be called negative
# (PASC coding is right-skewed; short follow-up != true negative). Cases are
# never dropped here. Recomputed from stored followup_days.
if "followup_days" in raw_cohort.columns:
    is_control = raw_cohort["label"] == 0
    short_fu = is_control & (raw_cohort["followup_days"] < MIN_FOLLOWUP_DAYS)
    n_short = int(short_fu.sum())
    print(f"\nConfirmed-negative gate (>= {MIN_FOLLOWUP_DAYS}d follow-up):")
    print(f"  Controls dropped (insufficient follow-up): {n_short:,}")
    raw_cohort = raw_cohort[~short_fu].copy()

    print(f"  Label distribution after gate: {raw_cohort['label'].value_counts().to_dict()}")
else:
    print("\n[WARN] No 'followup_days' column in cohort parquet; "
          "confirmed-negative gate skipped. Rebuild cohort to enable censoring.")

# =============================================================================
# 6 — Label Extraction
# =============================================================================

def _upload_window_cohort(cur, temp_table: str, df: pd.DataFrame) -> None:
    try:
        cur.execute(f'DROP TABLE "{temp_table}"')
        cur.connection.commit()
    except Exception:
        cur.connection.rollback()
    cur.execute(f"""
        CREATE LOCAL TEMPORARY TABLE "{temp_table}" (
            person_id BIGINT,
            window_start DATE,
            window_end   DATE
        )
    """)
    cur.connection.commit()
    rows = [
        (int(pid), str(ws), str(we))
        for pid, ws, we in df[["person_id", "window_start", "window_end"]]
            .itertuples(index=False, name=None)
    ]
    insert_sql = f'INSERT INTO "{temp_table}" (person_id, window_start, window_end) VALUES (?, ?, ?)'
    batch_size = 5000
    for i in range(0, len(rows), batch_size):
        cur.executemany(insert_sql, rows[i:i + batch_size])
    cur.connection.commit()
    print(f"  Uploaded {len(rows):,} rows to {temp_table}")


def _query_symptom_hits(cur, temp_table: str, symptom_groups: dict) -> dict:
    group_results = {}
    for group_name, group_def in symptom_groups.items():
        ancestor_clause = _ints_to_sql_in(group_def["ancestor_ids"])
        like_clauses = " OR ".join(
            f"UPPER(cn.concept_name) LIKE '{p}'" for p in group_def["name_patterns"]
        )
        sql = f"""
        SELECT DISTINCT c.person_id
        FROM "{temp_table}" c
        JOIN {CDM}.condition_occurrence co
          ON co.person_id = c.person_id
         AND co.condition_start_date BETWEEN c.window_start AND c.window_end
        LEFT JOIN {CDM}.concept_ancestor ca
          ON ca.descendant_concept_id = co.condition_concept_id
         AND ca.ancestor_concept_id IN ({ancestor_clause})
        LEFT JOIN {CDM}.concept cn
          ON cn.concept_id = co.condition_concept_id
        WHERE ca.ancestor_concept_id IS NOT NULL
           OR ({like_clauses})
        """
        print(f"  Querying {group_name}...")
        hit_df = run_query(cur, sql)
        hit_df.columns = [c.lower() for c in hit_df.columns]
        group_results[group_name] = set(hit_df["person_id"].astype(int).tolist())
        print(f"    {group_name}: {len(group_results[group_name]):,} patients")
    return group_results


def extract_subtype_labels(
    cur,
    cohort_df: pd.DataFrame,
    symptom_groups: dict,
    label_start_days: int = 21,
    label_end_days: int = 540,
) -> pd.DataFrame:
    """Return per-patient label_{g} from the post-acute label window."""
    result = cohort_df[["person_id"]].copy()
    for g in symptom_groups:
        result[f"label_{g}"] = 0

    pasc_pos = cohort_df[cohort_df["label"] == 1][["person_id", "covid_index_date"]].copy()
    pasc_pos["covid_index_date"] = pd.to_datetime(pasc_pos["covid_index_date"])
    pasc_pos["window_start"] = (
        pasc_pos["covid_index_date"] + pd.Timedelta(days=label_start_days)
    ).dt.date
    pasc_pos["window_end"] = (
        pasc_pos["covid_index_date"] + pd.Timedelta(days=label_end_days)
    ).dt.date

    if not pasc_pos.empty:
        print(f"\n  [label window] day {label_start_days}..{label_end_days}")
        _upload_window_cohort(cur, "#subtype_label_cohort", pasc_pos)
        for g, hits in _query_symptom_hits(cur, "#subtype_label_cohort", symptom_groups).items():
            result.loc[result["person_id"].isin(hits), f"label_{g}"] = 1

    return result


labels_path = os.path.join(RESULTS_DIR, "subtype_labels.parquet")

if RUN_LABEL_EXTRACTION:
    print("\n" + "=" * 80)
    print("SYMPTOM-GROUP LABEL EXTRACTION")
    print("=" * 80)

    subtype_labels_df = extract_subtype_labels(
        cur, raw_cohort, SYMPTOM_GROUPS,
        label_start_days=LABEL_WINDOW_START_DAYS,
        label_end_days=LABEL_WINDOW_END_DAYS,
    )

    subtype_labels_df.to_parquet(labels_path, index=False)
    print(f"\nSaved subtype labels: {labels_path}")
else:
    if os.path.exists(labels_path):
        subtype_labels_df = pd.read_parquet(labels_path)
        print(f"Loaded existing labels: {labels_path}")
    else:
        raise FileNotFoundError(f"Labels not found: {labels_path}. Run once with PASC_REEXTRACT=1.")

# Summary
label_cols = [f"label_{g}" for g in SUBTYPE_NAMES]
merged = raw_cohort[["person_id", "label"]].merge(subtype_labels_df, on="person_id")
pasc_mask = merged["label"] == 1
n_pasc = int(pasc_mask.sum())
print(f"\nPASC-positive patients: {n_pasc:,}")
print(f"{'Subtype':<25s} {'N':>8s} {'% of PASC':>10s}")
print("-" * 45)
for g in SUBTYPE_NAMES:
    n = int(merged.loc[pasc_mask, f"label_{g}"].sum())
    pct = 100 * n / n_pasc if n_pasc > 0 else 0
    print(f"{g:<25s} {n:>8,d} {pct:>9.1f}%")

# =============================================================================
# 7 — Feature Extraction / Assembly (with caching + calendar drop)
# =============================================================================

def filter_cohort_for_window(cohort_df, feature_window_end_days):
    df = cohort_df.copy()
    df["covid_index_date"] = pd.to_datetime(df["covid_index_date"])
    df["first_pacs_date"] = pd.to_datetime(df["first_pacs_date"], errors="coerce")
    days_to_pasc = (df["first_pacs_date"] - df["covid_index_date"]).dt.days
    keep_mask = (df["label"] == 0) | (
        (df["label"] == 1) & (days_to_pasc > feature_window_end_days)
    )
    n_before = len(df)
    n_excluded = (~keep_mask).sum()
    filtered = df[keep_mask].copy()
    print(f"  Leakage filter (window end={feature_window_end_days}d): "
          f"{n_before:,} -> {len(filtered):,} patients (excluded {n_excluded:,} PASC cases)")
    print(f"    Remaining label distribution: {filtered['label'].value_counts().to_dict()}")
    return filtered


def _apply_calendar_drop(feature_df, feature_families, model_configs):
    """Strip _CALENDAR_DROP_COLS from feature_df / families / configs (no-op when OFF)."""
    if not DROP_INDEX_CALENDAR:
        print("[drop_calendar] OFF -- calendar/index-time features retained in modeling")
        return feature_df, feature_families, model_configs
    dropped = []
    for _col in _CALENDAR_DROP_COLS:
        if _col in feature_df.columns:
            feature_df = feature_df.drop(columns=[_col])
            dropped.append(_col)
        for _fam, _cols in feature_families.items():
            if _col in _cols:
                feature_families[_fam] = [c for c in _cols if c != _col]
        for _cfg, _cols in model_configs.items():
            if _col in _cols:
                model_configs[_cfg] = [c for c in _cols if c != _col]
    print(f"[drop_calendar] ON -- removed {dropped or _CALENDAR_DROP_COLS} from modeling")
    return feature_df, feature_families, model_configs


assembled_data = {}

original_indicators = {
    "viral": VIRAL_INDICATOR_LIST,
    "immuno": IMMUNO_INDICATOR_LIST,
    "endo": ENDO_INDICATOR_LIST,
}
enhanced_indicators = {
    "enh_viral": ENH_VIRAL_INDICATOR_LIST,
    "enh_immuno": ENH_IMMUNO_INDICATOR_LIST,
    "enh_endo": ENH_ENDO_INDICATOR_LIST,
    "expl": ENH_EXPLORATORY_INDICATOR_LIST,
}


def _cache_paths(wl):
    pq = os.path.join(COHORT_DIR, f"subtype_enhanced_{wl}_{COHORT_MODE}.parquet")
    fj = os.path.join(COHORT_DIR, f"subtype_enhanced_{wl}_{COHORT_MODE}_families.json")
    return pq, fj


def _run_is_complete(perc, wl, config_name, subtype, results_dir):
    """True iff the iterations CSV for one (perc, window, config, subtype) quad exists."""
    path = os.path.join(results_dir, f"perc{perc}", wl, config_name, f"iterations_{subtype}.csv")
    return os.path.exists(path)


def _window_is_complete(wl, results_dir, perc_sweep, active_configs, subtype_names):
    """True iff every (perc, config, subtype) iteration CSV exists for this window."""
    for perc in perc_sweep:
        for cfg in active_configs:
            for sub in subtype_names:
                if not _run_is_complete(perc, wl, cfg, sub, results_dir):
                    return False
    return True


if REEXTRACT_FEATURES:
    # --- Phase 1: extract raw components per window --------------------------
    for (ws, we) in FEATURE_WINDOWS:
        wl = window_label(ws, we)

        # Resume: skip windows whose every (perc, config, subtype) CSV exists.
        if RESUME_FROM_LAST and _window_is_complete(
            wl, RESULTS_DIR, BORUTA_PERC_SWEEP, ACTIVE_CONFIGS, SUBTYPE_NAMES
        ):
            print(f"\n>>> SKIP feature extraction for {wl} — all results already on disk (RESUME_FROM_LAST=True)")
            continue

        print(f"\n{'#'*80}")
        print(f"# FEATURE EXTRACTION -- {wl}")
        print(f"{'#'*80}")

        filtered_cohort = filter_cohort_for_window(raw_cohort, we)

        window_cohort = filtered_cohort.copy()
        window_cohort = compute_acute_window(window_cohort, acute_days=we)
        upload_antony_cohort_temp(cur, window_cohort)

        window_baseline_df, window_baseline_families = build_combined_feature_matrix(
            cur, window_cohort,
            include_treatment=True,
            include_extended=True,
            include_mechanistic=False,
            use_hpo_symptoms=True,
        )
        print(f"  Baseline features ({we}d window): {sum(len(v) for v in window_baseline_families.values())} columns")

        cohort_for_signals = filtered_cohort.copy()
        wdf = make_window_cohort(cohort_for_signals, ws, we)

        windowed_binary = {}
        print(f"\n  --- Binary signals: {wl} ---")
        viral_df = extract_binary_signals_for_window(cur, wdf, VIRAL_INDICATOR_LIST, "viral", wl)
        immuno_df = extract_binary_signals_for_window(cur, wdf, IMMUNO_INDICATOR_LIST, "immuno", wl)
        endo_df = extract_binary_signals_for_window(cur, wdf, ENDO_INDICATOR_LIST, "endo", wl)
        enh_viral_df = extract_binary_signals_for_window(cur, wdf, ENH_VIRAL_INDICATOR_LIST, "enh_viral", wl)
        enh_immuno_df = extract_binary_signals_for_window(cur, wdf, ENH_IMMUNO_INDICATOR_LIST, "enh_immuno", wl)
        enh_endo_df = extract_binary_signals_for_window(cur, wdf, ENH_ENDO_INDICATOR_LIST, "enh_endo", wl)
        enh_expl_df = extract_binary_signals_for_window(cur, wdf, ENH_EXPLORATORY_INDICATOR_LIST, "expl", wl)

        combined_binary = viral_df.copy()
        for df in [immuno_df, endo_df, enh_viral_df, enh_immuno_df, enh_endo_df, enh_expl_df]:
            combined_binary = combined_binary.merge(df, on="person_id", how="outer")
        windowed_binary[wl] = combined_binary

        windowed_labs = {}
        print(f"\n  --- Numeric labs: {wl} ---")
        lab_df = extract_numeric_labs(cur, wdf, NUMERIC_LAB_SPECS, wl)
        windowed_labs[wl] = lab_df

        windowed_cbc = {}
        print(f"\n  --- CBC indices: {wl} ---")
        cbc_df = extract_cbc_indices(cur, wdf, wl)
        windowed_cbc[wl] = cbc_df

        print(f"\n  --- Composite features: {wl} ---")
        composite_df = build_composite_features(
            cur, cohort_for_signals, windowed_labs, windowed_binary,
            gi_window=(ws, we),
        )

        assembled_data[wl] = {
            "baseline_df": window_baseline_df,
            "baseline_families": window_baseline_families,
            "windowed_binary": windowed_binary,
            "windowed_labs": windowed_labs,
            "windowed_cbc": windowed_cbc,
            "composite_df": composite_df,
            "filtered_cohort": filtered_cohort,
            "window_end_days": we,
        }
        print(f"\n  Feature extraction components ready for {wl}.")

    # --- Phase 2: temporal-divergence composites + assembly + cache save -----
    if assembled_data:
        print("\n[7] Building temporal-divergence composites and assembling features...")

        all_window_binaries = {}
        for _wl, _adata in assembled_data.items():
            for sub_wl, bdf in _adata["windowed_binary"].items():
                if sub_wl not in all_window_binaries:
                    all_window_binaries[sub_wl] = bdf
        print(f"  Available windows for temporal-divergence composites: {sorted(all_window_binaries.keys())}")

        for wl, adata in assembled_data.items():
            temporal_divergence_df = build_temporal_divergence_composites(
                windowed_binary=all_window_binaries,
                cohort_df=adata["filtered_cohort"],
                post_acute_days=30,
                max_window_end_day=adata["window_end_days"],
            )

            feature_df, feature_families = assemble_enhanced_feature_matrix(
                baseline_df=adata["baseline_df"],
                baseline_families=adata["baseline_families"],
                windowed_binary=adata["windowed_binary"],
                windowed_labs=adata["windowed_labs"],
                windowed_cbc=adata["windowed_cbc"],
                composite_df=adata["composite_df"],
                temporal_divergence_df=temporal_divergence_df,
                original_indicators=original_indicators,
                enhanced_indicators=enhanced_indicators,
            )

            # Save the FULL matrix (with calendar cols) BEFORE the modeling drop,
            # so the cache stays complete and the toggle is reversible.
            pq, fj = _cache_paths(wl)
            feature_df.to_parquet(pq, index=False)
            with open(fj, "w") as _fh:
                json.dump({k: list(v) for k, v in feature_families.items()}, _fh, indent=2)
            print(f"  Saved cache: {pq}")
            print(f"  Saved cache: {fj}")

            all_configs = build_enhanced_model_configs(feature_families)
            model_configs = {k: v for k, v in all_configs.items() if k in ACTIVE_CONFIGS}

            feature_df, feature_families, model_configs = _apply_calendar_drop(
                feature_df, feature_families, model_configs
            )

            adata["feature_df"] = feature_df
            adata["feature_families"] = feature_families
            adata["model_configs"] = model_configs
            print(f"  Active configs for {wl}: {list(model_configs.keys())}")
            print(f"  Feature assembly complete for {wl}. Shape: {feature_df.shape}")
else:
    # --- Reuse path: load cached per-window matrices, skip DB ----------------
    print("\n[7] Reuse mode: loading cached feature matrices (REEXTRACT_FEATURES=False)...")
    for (ws, we) in FEATURE_WINDOWS:
        wl = window_label(ws, we)
        pq, fj = _cache_paths(wl)
        if not (os.path.exists(pq) and os.path.exists(fj)):
            raise FileNotFoundError(
                f"REEXTRACT_FEATURES=False but cached artefacts missing for {wl}:\n"
                f"  {pq}\n  {fj}\nRe-run once with PASC_REEXTRACT=1 to populate."
            )
        feature_df = pd.read_parquet(pq)
        with open(fj) as _fh:
            feature_families = json.load(_fh)
        print(f"  Loaded cached feature_df: {feature_df.shape} <- {pq}")

        all_configs = build_enhanced_model_configs(feature_families)
        model_configs = {k: v for k, v in all_configs.items() if k in ACTIVE_CONFIGS}

        feature_df, feature_families, model_configs = _apply_calendar_drop(
            feature_df, feature_families, model_configs
        )

        assembled_data[wl] = {
            "feature_df": feature_df,
            "feature_families": feature_families,
            "model_configs": model_configs,
            "window_end_days": we,
        }
        print(f"  Active configs for {wl}: {list(model_configs.keys())}")

# Extraction-only mode: caches + labels are saved by now. Skip the (login-node-
# hostile) modeling sweep so a DB extraction pass can run foreground with
# credentials, then model separately per-window with PASC_REEXTRACT=0.
if os.environ.get("PASC_EXTRACT_ONLY", "0") == "1":
    print("\n[extract-only] PASC_EXTRACT_ONLY=1 -- feature caches + labels saved; "
          "skipping modeling. Model per-window with PASC_REEXTRACT=0.")
    raise SystemExit(0)

# Blocklist audit mode: resolve the per-subtype drop list against the full
# feature table and against each active config's columns, dump to CSV, exit.
# No modeling, no DB. Run on the login node before submitting the sweep.
if os.environ.get("PASC_BLOCKLIST_AUDIT", "0") == "1":
    print("\n[blocklist-audit] PASC_BLOCKLIST_AUDIT=1 -- resolving drop lists, no modeling.")
    for wl, adata in assembled_data.items():
        full_cols = list(adata["feature_df"].columns)
        model_configs = adata["model_configs"]
        audit_rows = []
        print(f"\n[blocklist-audit] window {wl} (full table: {len(full_cols)} columns)")
        for subtype in MODEL_SUBTYPES:
            full_drop = resolve_blocklist(subtype, full_cols)
            print(f"  {subtype}: {len(full_drop)} against full table -> {full_drop}")
            for feat in full_drop:
                audit_rows.append({"subtype": subtype, "config": "__full_table__", "feature": feat})
            for config_name, config_cols in model_configs.items():
                if config_name not in ACTIVE_CONFIGS:
                    continue
                cfg_drop = [c for c in full_drop if c in config_cols]
                print(f"    {config_name}: {len(cfg_drop)} against config columns")
                for feat in cfg_drop:
                    audit_rows.append({"subtype": subtype, "config": config_name, "feature": feat})
        audit_df = pd.DataFrame(audit_rows, columns=["subtype", "config", "feature"])
        audit_path = os.path.join(RESULTS_DIR, f"blocklist_audit_{wl}.csv")
        audit_df.to_csv(audit_path, index=False)
        print(f"  Saved: {audit_path}")
    raise SystemExit(0)

# =============================================================================
# 8 — Multi-Label Modeling Pipeline (Boruta perc sweep)
# =============================================================================

all_subtype_results = {}

engagement_cols = None
for _wl, _ad in assembled_data.items():
    eng = _ad["feature_families"].get("engagement_controls", [])
    if eng:
        engagement_cols = eng
        break
if engagement_cols:
    print(f"Forced features (engagement controls): {engagement_cols}")
else:
    print("WARNING: No engagement_controls found -- forced_features will be None")

if RUN_MODELING and RESUME_FROM_LAST:
    total = 0
    skipped = 0
    for perc in BORUTA_PERC_SWEEP:
        for wl in assembled_data:
            for cfg in ACTIVE_CONFIGS:
                for sub in MODEL_SUBTYPES:
                    total += 1
                    if _run_is_complete(perc, wl, cfg, sub, RESULTS_DIR):
                        skipped += 1
    print(f"\n[resume] {skipped}/{total} (perc, window, config, subtype) quads already complete; "
          f"will run the remaining {total - skipped}.")

if RUN_MODELING:
    for perc in BORUTA_PERC_SWEEP:
        perc_results_dir = os.path.join(RESULTS_DIR, f"perc{perc}")
        os.makedirs(perc_results_dir, exist_ok=True)
        print("\n" + "#" * 80)
        print(f"# BORUTA PERC SWEEP — perc={perc}  ->  {perc_results_dir}")
        print("#" * 80)

        for wl, adata in assembled_data.items():
            feature_df = adata["feature_df"]
            ff = adata["feature_families"]
            model_configs = adata["model_configs"]

            feature_with_labels = feature_df.merge(subtype_labels_df, on="person_id", how="left")
            for g in SUBTYPE_NAMES:
                col = f"label_{g}"
                if col in feature_with_labels.columns:
                    feature_with_labels[col] = feature_with_labels[col].fillna(0).astype(int)

            for config_name, config_cols in model_configs.items():
                for subtype in MODEL_SUBTYPES:
                    label_col = f"label_{subtype}"
                    run_label = f"{wl}/{COHORT_MODE}/{config_name}/{subtype} (perc={perc})"

                    if RESUME_FROM_LAST and _run_is_complete(perc, wl, config_name, subtype, RESULTS_DIR):
                        iter_path = os.path.join(
                            perc_results_dir, wl, config_name, f"iterations_{subtype}.csv"
                        )
                        print(f"\n  SKIP (resume): {run_label} -- {iter_path} exists")
                        continue

                    # Fix A: drop overlapping feature columns for this subtype.
                    # Resolve against the FULL feature table so a baseline config
                    # that never carries f_ind_enh_* does not raise on a miss.
                    dropped = resolve_blocklist(subtype, list(feature_with_labels.columns))
                    dropped = [c for c in dropped if c in config_cols]
                    config_cols_subtype = [c for c in config_cols if c not in dropped]

                    n_pos = int(feature_with_labels[label_col].sum())
                    if n_pos < 10:
                        print(f"\n  SKIP: {run_label} -- only {n_pos} positive cases")
                        continue

                    print(f"\n{'#'*80}")
                    print(f"# RUN: {run_label}  "
                          f"({len(config_cols_subtype)} features, {n_pos} positives)")
                    if dropped:
                        print(f"#   dropped {len(dropped)} overlap feature(s): {dropped}")
                    print(f"{'#'*80}")

                    model_df = feature_with_labels.copy()
                    model_df["label"] = model_df[label_col]

                    results = run_full_pipeline(
                        feature_df=model_df,
                        feature_families=ff,
                        cohort_name=run_label,
                        n_iterations=N_ITERATIONS,
                        model_types=MODEL_TYPES,
                        random_state_base=RANDOM_STATE_BASE,
                        boruta_max_iter=BORUTA_MAX_ITER,
                        boruta_n_estimators=BORUTA_N_ESTIMATORS,
                        compute_shap=COMPUTE_SHAP,
                        feature_columns=config_cols_subtype,
                        forced_features=engagement_cols,
                        boruta_perc=perc,
                    )

                    all_subtype_results[(perc, wl, config_name, subtype)] = results

                    run_dir = os.path.join(perc_results_dir, wl, config_name)
                    os.makedirs(run_dir, exist_ok=True)
                    iter_df = results_to_dataframe(results)
                    iter_df["feature_window"] = wl
                    iter_df["mode"] = COHORT_MODE
                    iter_df["config"] = config_name
                    iter_df["subtype"] = subtype
                    iter_df["n_positive"] = n_pos
                    iter_df["n_features_dropped"] = len(dropped)
                    iter_df["features_dropped"] = ";".join(dropped)
                    iter_df["boruta_perc"] = perc
                    iter_path = os.path.join(run_dir, f"iterations_{subtype}.csv")
                    iter_df.to_csv(iter_path, index=False)
                    print(f"  Saved: {iter_path}")

                    if COMPUTE_SHAP:
                        for model_type, pr in results.items():
                            shap_rows = []
                            for it in pr.iterations:
                                if it.shap_values is not None and it.shap_feature_names is not None:
                                    mean_abs = np.mean(np.abs(it.shap_values), axis=0)
                                    # Flatten per-class SHAP to scalar (binary classification)
                                    if mean_abs.ndim > 1:
                                        mean_abs = mean_abs.mean(axis=-1)
                                    for fname, val in zip(it.shap_feature_names, mean_abs):
                                        shap_rows.append({
                                            "iteration": it.iteration,
                                            "feature": fname,
                                            "mean_abs_shap": float(val),
                                        })
                            if shap_rows:
                                shap_path = os.path.join(run_dir, f"shap_{subtype}_{model_type}.csv")
                                pd.DataFrame(shap_rows).to_csv(shap_path, index=False)
                                print(f"  Saved SHAP: {shap_path}")

    # Rebuild combined CSV from ALL per-run CSVs (across every perc)
    import glob as _glob
    all_csvs = sorted(_glob.glob(os.path.join(RESULTS_DIR, "perc*", "w*", "*", "iterations_*.csv")))
    if all_csvs:
        combined_df = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
        combined_path = os.path.join(RESULTS_DIR, "all_iterations.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"\nCombined iterations saved: {combined_path}  ({len(combined_df)} rows from {len(all_csvs)} files)")
    else:
        print("\nWARNING: No iterations produced.")
else:
    print("Modeling skipped (RUN_MODELING=False)")

# =============================================================================
# Done
# =============================================================================
print("\n" + "=" * 80)
print("COMPUTATION COMPLETE")
print("=" * 80)
print(f"Results saved under: {RESULTS_DIR}")
print("Figures and the paired tests are produced by notebooks/make_thesis_figures.ipynb.")

#!/usr/bin/env python
"""
Main analysis: seven feature configurations x six feature windows x 100 repeats.

This is the entry script named in Appendix A.5 of the thesis. It

  1. loads the cohort table built by rebuild_combined_cohort.py
     (cohort_parquets/combined_strict_all_patients.parquet),
  2. applies the modelling-time gates (index date <= data cutoff, no PASC code
     in the first 90 days, >= 365 days of follow-up for controls),
  3. for every feature window in FEATURE_WINDOWS extracts the baseline blocks
     (Antony et al. families A-E, extended covariates, engagement controls) and
     the mechanism blocks (binary indicators, laboratory summaries, CBC indices,
     composites, gated temporal-divergence composites), and caches the matrix
     as cohort_parquets/enhanced_<window>_strict_all_patients.parquet plus a
     families.json,
  4. fits every configuration in ACTIVE_CONFIGS with antony_pipeline.run_full_pipeline
     (balancing, prevalence filter, Boruta, median imputation, grid-searched
     Random Forest, SHAP) on 100 seeded repeats, and
  5. writes per-repeat metrics, Boruta selections and SHAP summaries to
     results/main/perc97/<window>/strict/.

Run:
    python run_enhanced_mechsig.py                         # extract + model
    PASC_REEXTRACT=0 python run_enhanced_mechsig.py        # model from cached matrices
    PASC_EXTRACT_ONLY=1 python run_enhanced_mechsig.py     # extract only (needs DB)
    PASC_FEATURE_WINDOWS=0_90 python run_enhanced_mechsig.py   # one window

The database connection (init.py) is only needed when PASC_REEXTRACT=1.
"""

import os
import sys
import json
import importlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch jobs

from pasc_paths import REPO_ROOT, COHORT_DIR as _COHORT_DIR, MAIN_RESULTS_DIR

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print("=" * 80)
print("PASC mechanism-feature analysis: configurations x windows x repeats")
print("=" * 80)

# =============================================================================
# 0. RE-EXTRACTION TOGGLE
# =============================================================================
# When True (default), the script connects to the DB and re-runs sections 3-5
# (cohort load, baseline features, per-window binary/lab/CBC/composite
# extraction) and section 6 writes fresh parquet + families.json artefacts.
# When False, sections 1/3/4/5 are skipped entirely; section 6 loads the
# previously-saved enhanced_{wl}_{COHORT_MODE}_all_patients.parquet and
# enhanced_{wl}_{COHORT_MODE}_families.json from COHORT_DIR. Useful for
# iterating on modeling without re-querying HANA.
# Toggle via env var:  PASC_REEXTRACT=0 python run_enhanced_mechsig.py
REEXTRACT_FEATURES = os.environ.get("PASC_REEXTRACT", "1") == "1"
print(f"\n[0] REEXTRACT_FEATURES = {REEXTRACT_FEATURES} (set PASC_REEXTRACT=0 to reuse cached parquets)")

# =============================================================================
# 1. SETUP & DB CONNECTION
# =============================================================================
if REEXTRACT_FEATURES:
    print("\n[1] Initializing database connection...")
    exec(open(os.path.join(REPO_ROOT, "init.py")).read())
    cur = hana_conn.cursor()
    print(f"Connection active: {hana_conn.isconnected()}")
else:
    print("\n[1] Skipped DB init (REEXTRACT_FEATURES=False).")
    cur = None

# Import modules
from antony_cohort import upload_antony_cohort_temp, compute_acute_window
from combined_features import build_combined_feature_matrix
from antony_pipeline import run_full_pipeline, results_to_dataframe, print_summary_table
from mech_signals_common import prepare_temp_cohort
from combined_cohort import DATA_CUTOFF

from indicator_definitions import (
    VIRAL_INDICATOR_LIST, IMMUNO_INDICATOR_LIST, ENDO_INDICATOR_LIST,
)

import enhanced_mech_signals as _ems
importlib.reload(_ems)
from enhanced_mech_signals import (
    MECH_WINDOWS, window_label,
    make_window_cohort,
    extract_binary_signals_for_window,
    extract_numeric_labs,
    extract_cbc_indices,
    build_composite_features,
    build_temporal_divergence_composites,
    audit_concept_availability,
    build_enhanced_model_configs,
    assemble_enhanced_feature_matrix,
    NUMERIC_LAB_SPECS, CBC_SPECS,
    ENH_VIRAL_INDICATOR_LIST,
    ENH_IMMUNO_INDICATOR_LIST,
    ENH_ENDO_INDICATOR_LIST,
    ENH_EXPLORATORY_INDICATOR_LIST,
)

print("All imports successful.")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
print("\n[2] Loading configuration...")

BASELINE_ACUTE_DAYS = 21
FEATURE_WINDOWS = [
    (0, 21),   # Antony-matching (mech signals also from acute phase)
    (0, 30),   # Extended acute
    (0, 60),   # Two months
    (0, 90),   # Three months
    (30, 60),  # Late only: month 2
    (60, 90),  # Late only: month 3
#    (30, 90),  # Late only: months 2-3
]

# Optional override to run a subset of windows without editing the list above.
# Format: comma-separated "start_end" pairs, e.g.
#   PASC_FEATURE_WINDOWS="0_30" python run_enhanced_mechsig.py        # single
#   PASC_FEATURE_WINDOWS="0_30,30_90" python run_enhanced_mechsig.py  # several
# Useful to (re)extract/model one new window while leaving the other windows'
# results untouched. Pair with PASC_REEXTRACT=1 when no clean cache exists yet.
_fw_override = os.environ.get("PASC_FEATURE_WINDOWS", "").strip()
if _fw_override:
    FEATURE_WINDOWS = [
        tuple(int(x) for x in pair.split("_"))
        for pair in _fw_override.split(",") if pair.strip()
    ]
    print(f"[feature_windows] override via PASC_FEATURE_WINDOWS -> {FEATURE_WINDOWS}")

N_ITERATIONS       = 100    # repeats; seed of repeat i is RANDOM_STATE_BASE + 1000*i (thesis 3.6.2)
BORUTA_MAX_ITER    = 50     # Boruta rounds (thesis Table 3.5)
BORUTA_N_ESTIMATORS = 500   # trees in the Boruta forest
BORUTA_PERC_SWEEP  = [97]   # shadow percentile a feature must beat (thesis 3.6.3); a list so a sweep is possible
COMPUTE_SHAP       = True
RANDOM_STATE_BASE  = 42     # Held FIXED across all perc values (identical splits/subsamples)
MODEL_TYPES        = ("RF",)

# --- Confirmed-negative ascertainment gate -------------------------------
# A control only counts as a negative if it was observed (followup_days)
# for at least this long without a PASC diagnosis. PASC coding is heavily
# right-skewed, so a short window (e.g. 90d) leaves the door open to
# survivorship/ascertainment bias: a "control" that exits early may simply
# not have been observed long enough to be diagnosed. 365d is the default.
# Controls below the gate are dropped (not labelled negative); cases are
# never affected. Recomputed at load time from the stored followup_days, so
# no cohort rebuild is needed to change it.
#   PASC_MIN_FOLLOWUP_DAYS=365 python run_enhanced_mechsig.py
MIN_FOLLOWUP_DAYS = int(os.environ.get("PASC_MIN_FOLLOWUP_DAYS", "365"))

# --- Withhold the index-date calendar feature from MODELING ---------------
# f_ext_index_year_month is a continuous index-date feature. Follow-up ends at
# one fixed cutoff for everyone, so in this cohort it is almost perfectly
# collinear with follow-up time: recently-indexed patients have short
# observation windows and therefore little chance of being coded PASC+. A
# forest given the feature reads that censoring pattern instead of the patient
# (thesis 3.6). It is dropped from MODELING only -- the cached all_patients
# parquet + families.json keep it, so the toggle is reversible with no
# re-extract. Set to 0 to restore it for an A/B.
#   PASC_DROP_INDEX_CALENDAR=0 python run_enhanced_mechsig.py   # keep it
DROP_INDEX_CALENDAR = os.environ.get("PASC_DROP_INDEX_CALENDAR", "1") == "1"
_CALENDAR_DROP_COLS = [
    "f_ext_index_year_month",
]

# Fast mode: lower-cost knobs for a quick end-to-end smoke test (not the thesis setting).
# Toggle via env var: PASC_FAST_MODE=1 python run_enhanced_mechsig.py
FAST_MODE = os.environ.get("PASC_FAST_MODE", "0") == "1"
if FAST_MODE:
    N_ITERATIONS        = 5
    BORUTA_MAX_ITER     = 30
    BORUTA_N_ESTIMATORS = 100
    COMPUTE_SHAP        = True
    print(f"[fast_mode] N_ITERATIONS={N_ITERATIONS} BORUTA_MAX_ITER={BORUTA_MAX_ITER} "
          f"BORUTA_N_ESTIMATORS={BORUTA_N_ESTIMATORS} COMPUTE_SHAP={COMPUTE_SHAP}")

RUN_CONCEPT_AUDIT       = True
RUN_BINARY_SIGNALS      = True
RUN_NUMERIC_LABS        = True
RUN_CBC_INDICES         = True
RUN_COMPOSITES          = True
RUN_MODELING            = True

COHORT_MODE = "strict"   # COVID-positive patients only (thesis 3.2); "relaxed" adds PASC-coded patients without a COVID record and is not used

COHORT_DIR  = str(_COHORT_DIR)
RESULTS_DIR = str(MAIN_RESULTS_DIR)
os.makedirs(COHORT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Configuration:")
print(f"  BASELINE_ACUTE_DAYS: {BASELINE_ACUTE_DAYS}")
print(f"  FEATURE_WINDOWS:     {FEATURE_WINDOWS}")
print(f"  N_ITERATIONS:        {N_ITERATIONS}")
print(f"  MODEL_TYPES:         {MODEL_TYPES}")
print(f"  COHORT_MODE:         {COHORT_MODE}")
print(f"  RESULTS_DIR:         {RESULTS_DIR}")

if REEXTRACT_FEATURES:
    # =============================================================================
    # 3. COHORT LOADING & BASELINE FEATURE EXTRACTION
    # =============================================================================
    print("\n[3] Loading cohort and extracting baseline features...")

    raw_cohort_path = os.path.join(COHORT_DIR, f"combined_{COHORT_MODE}_all_patients.parquet")
    if not os.path.exists(raw_cohort_path):
        raise FileNotFoundError(f"Raw cohort parquet not found: {raw_cohort_path}")

    raw_cohort = pd.read_parquet(raw_cohort_path)
    raw_cohort["covid_index_date"] = pd.to_datetime(raw_cohort["covid_index_date"])
    raw_cohort["discharge_date"] = pd.to_datetime(raw_cohort["discharge_date"], errors="coerce")
    raw_cohort["first_pacs_date"] = pd.to_datetime(raw_cohort["first_pacs_date"], errors="coerce")
    print(f"Loaded raw cohort: {len(raw_cohort):,} patients from {raw_cohort_path}")

    # --- Fixed data-cutoff enforcement (ascertainment horizon) ---------------
    # Drop future-dated index (data artifacts past the pull date) and re-cap
    # follow-up at the cutoff so the confirmed-negative gate below operates on
    # cutoff-consistent follow-up (mirrors combined_cohort.build_combined_cohorts).
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

    # PASC label enforcement: a PASC diagnosis only counts as a case if it
    # occurs >= MIN_DAYS_POST_ACUTE days post-COVID. The case rule uses 90 days (thesis 3.2) so no
    # feature window (<=90d) can overlap the outcome.
    #
    # Patients whose FIRST PASC dx falls < 90d post-index (acute-phase or
    # pre-index/prevalent coding -- many have days_to_pasc <= 0) are NOT clean
    # negatives: they carry a PASC code, just outside the post-acute window.
    # Folding them into the control pool contaminates the negatives with the
    # highest-risk patients. We therefore DROP them from the cohort entirely
    # (neither case nor control), rather than relabelling them to 0.
    MIN_DAYS_POST_ACUTE = 90
    days_to_pasc = (raw_cohort["first_pacs_date"] - raw_cohort["covid_index_date"]).dt.days
    acute_phase_mask = raw_cohort["first_pacs_date"].notna() & (days_to_pasc < MIN_DAYS_POST_ACUTE)
    n_drop_acute = int(acute_phase_mask.sum())
    print(f"\nPASC label enforcement (>= {MIN_DAYS_POST_ACUTE}d post-COVID):")
    print(f"  Patients with first PASC dx < {MIN_DAYS_POST_ACUTE}d post-COVID "
          f"(acute/prevalent) DROPPED (not counted as controls): {n_drop_acute:,}")
    raw_cohort = raw_cohort[~acute_phase_mask].copy()
    print(f"  Label distribution after enforcement: {raw_cohort['label'].value_counts().to_dict()}")

    # --- Confirmed-negative ascertainment gate -------------------------------
    # Drop controls that were not observed long enough to confidently be called
    # negative (PASC coding is right-skewed; short follow-up != true negative).
    # Cases (label==1) are never dropped here. Recomputed from stored
    # followup_days so the threshold can change without a cohort rebuild.
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

    print(f"\n{'#'*80}")
    print(f"# BASELINE — {BASELINE_ACUTE_DAYS}d / {COHORT_MODE} / all_patients")
    print(f"{'#'*80}")

    cohort_df = raw_cohort.copy()
    cohort_df = compute_acute_window(cohort_df, acute_days=BASELINE_ACUTE_DAYS)
    upload_antony_cohort_temp(cur, cohort_df)

    baseline_feature_df, baseline_feature_families = build_combined_feature_matrix(
        cur, cohort_df,
        include_treatment=True,
        include_extended=True,
        include_mechanistic=False,
        use_hpo_symptoms=True,
    )

    print(f"  Baseline features: {sum(len(v) for v in baseline_feature_families.values())} columns")
    print(f"  Cohort size: {len(cohort_df):,} patients")

    # =============================================================================
    # 4. CONCEPT AVAILABILITY AUDIT
    # =============================================================================
    if RUN_CONCEPT_AUDIT:
        print("\n[4] Running concept availability audit...")
        wdf = make_window_cohort(cohort_df, 0, 180)
        prepare_temp_cohort(cur, wdf)

        audit_df = audit_concept_availability(
            cur, cohort_df,
            lab_specs={**NUMERIC_LAB_SPECS, **CBC_SPECS},
            indicator_lists={
                "enh_viral": ENH_VIRAL_INDICATOR_LIST,
                "enh_immuno": ENH_IMMUNO_INDICATOR_LIST,
                "enh_endo": ENH_ENDO_INDICATOR_LIST,
                "enh_exploratory": ENH_EXPLORATORY_INDICATOR_LIST,
            },
        )

        print("\n" + "=" * 80)
        print("CONCEPT AVAILABILITY AUDIT")
        print("=" * 80)
        labs = audit_df[audit_df["family"] == "numeric_lab"].sort_values("pct", ascending=False)
        print("\nNumeric Labs (measured in cohort):")
        for _, row in labs.iterrows():
            flag = "✓" if row["pct"] >= 1 else "⊗ SPARSE"
            print(f"  {row['name']:25s}: {row['n_patients']:6,} patients ({row['pct']:5.1f}%) {flag}")

        indicators = audit_df[audit_df["family"] != "numeric_lab"]
        print(f"\nEnhanced binary indicators: {len(indicators)} defined")
    else:
        print("\n[4] Concept audit skipped.")

    # =============================================================================
    # 5. SINGLE-WINDOW FEATURE EXTRACTION
    # =============================================================================
    print("\n[5] Extracting features for each window...")

    all_enhanced_data = {}

    for (ws, we) in FEATURE_WINDOWS:
        wl = window_label(ws, we)
        print(f"\n{'#'*80}")
        print(f"# ENHANCED EXTRACTION — feature_window={wl}")
        print(f"{'#'*80}")

        # --- 5.0 Baseline features for this window ---
        # Re-extract baseline features using acute_days=we so that baseline configs
        # (baseline_antony, baseline_ext) also vary with the feature window.
        window_cohort = raw_cohort.copy()
        window_cohort = compute_acute_window(window_cohort, acute_days=we, window_start=ws)
        upload_antony_cohort_temp(cur, window_cohort)

        window_baseline_df, window_baseline_families = build_combined_feature_matrix(
            cur, window_cohort,
            include_treatment=True,
            include_extended=True,
            include_mechanistic=False,
            use_hpo_symptoms=True,
        )
        print(f"  Baseline features ({we}d window): {sum(len(v) for v in window_baseline_families.values())} columns")

        wdf = make_window_cohort(cohort_df, ws, we)

        # --- 5a. Binary signals ---
        windowed_binary = {}
        if RUN_BINARY_SIGNALS:
            print(f"\n  --- Binary signals: {wl} ---")
            viral_df = extract_binary_signals_for_window(cur, wdf, VIRAL_INDICATOR_LIST, "viral", wl)
            immuno_df = extract_binary_signals_for_window(cur, wdf, IMMUNO_INDICATOR_LIST, "immuno", wl)
            endo_df = extract_binary_signals_for_window(cur, wdf, ENDO_INDICATOR_LIST, "endo", wl)
            enh_viral_df = extract_binary_signals_for_window(cur, wdf, ENH_VIRAL_INDICATOR_LIST, "enh_viral", wl)
            enh_immuno_df = extract_binary_signals_for_window(cur, wdf, ENH_IMMUNO_INDICATOR_LIST, "enh_immuno", wl)
            enh_endo_df = extract_binary_signals_for_window(cur, wdf, ENH_ENDO_INDICATOR_LIST, "enh_endo", wl)
            enh_expl_df = extract_binary_signals_for_window(cur, wdf, ENH_EXPLORATORY_INDICATOR_LIST, "expl", wl)

            combined = viral_df.copy()
            for df in [immuno_df, endo_df, enh_viral_df, enh_immuno_df, enh_endo_df, enh_expl_df]:
                combined = combined.merge(df, on="person_id", how="outer")

            windowed_binary[wl] = combined
            print(f"    Total binary columns for {wl}: {len([c for c in combined.columns if c.startswith('f_ind_')])}")

        windowed_labs = {}
        if RUN_NUMERIC_LABS:
            print(f"\n  --- Numeric labs: {wl} ---")
            lab_df = extract_numeric_labs(cur, wdf, NUMERIC_LAB_SPECS, wl)
            windowed_labs[wl] = lab_df

        windowed_cbc = {}
        if RUN_CBC_INDICES:
            print(f"\n  --- CBC indices: {wl} ---")
            cbc_df = extract_cbc_indices(cur, wdf, wl)
            windowed_cbc[wl] = cbc_df

        composite_df = pd.DataFrame({"person_id": cohort_df["person_id"].unique()})
        if RUN_COMPOSITES:
            print(f"\n  --- Composite features: {wl} ---")
            composite_df = build_composite_features(
                cur, cohort_df, windowed_labs, windowed_binary,
                gi_window=(ws, we),
            )

        # Store everything for this feature window (including per-window baselines)
        all_enhanced_data[wl] = {
            "baseline_df": window_baseline_df,
            "baseline_families": window_baseline_families,
            "windowed_binary": windowed_binary,
            "windowed_labs": windowed_labs,
            "windowed_cbc": windowed_cbc,
            "composite_df": composite_df,
        }
        print(f"\n  Extraction complete for {wl}.")

    # =============================================================================
    # 5e. TEMPORAL-DIVERGENCE COMPOSITES (built per-window inside the loop below)
    # =============================================================================
    # v2 leakage fix: previously a single, ungated temporal_divergence_df was built
    # here and reused for every feature window, which let a w0-21 model see
    # divergence signals from sub-windows ending at day 60 / 90. The build now lives
    # inside the per-window assembly loop and passes max_window_end_day=we so the
    # gate in build_temporal_divergence_composites can apply correctly.
    print("\n[5e] Temporal-divergence composites will be built per-window (v2 leakage gate).")

    # Collect all per-window binary DFs from across experiments (used by every window's
    # td build; the gate inside build_temporal_divergence_composites is responsible
    # for restricting which sub-windows actually contribute).
    all_window_binaries = {}
    for wl, enh in all_enhanced_data.items():
        for sub_wl, bdf in enh["windowed_binary"].items():
            if sub_wl not in all_window_binaries:
                merged = bdf
                # Defect-6: temporal-divergence composites also draw on the
                # numeric-lab abnormal_any flags (f_lab_*_abnormal_any), which the
                # v2 purification introduced in place of the old binary elev_*
                # indicators.  Those flags live in the separate windowed_labs
                # frames, so merge them in here; otherwise composites like
                # crp_esr_elevated / sustained_coag silently match nothing.
                ldf = enh.get("windowed_labs", {}).get(sub_wl)
                if ldf is not None and not ldf.empty:
                    abn_cols = [c for c in ldf.columns if c.endswith("_abnormal_any")]
                    if abn_cols:
                        merged = bdf.merge(ldf[["person_id"] + abn_cols],
                                           on="person_id", how="left")
                all_window_binaries[sub_wl] = merged

    print(f"Available sub-windows for temporal-divergence composites: {sorted(all_window_binaries.keys())}")
else:
    print("\n[3-5] Skipped cohort + per-window extraction (REEXTRACT_FEATURES=False).")
    raw_cohort = None
    cohort_df = None
    all_enhanced_data = {}
    all_window_binaries = {}

# =============================================================================
# 6. FEATURE MATRIX ASSEMBLY & MODEL CONFIGS
# =============================================================================
print("\n[6] Assembling feature matrices...")

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

# =============================================================================
# ACTIVE_CONFIGS -- which feature configurations to fit
# -----------------------------------------------------------------------------
# Each name is a feature-column list built by
# enhanced_mech_signals.build_enhanced_model_configs(). The f_eng_* block
# (six healthcare-engagement controls, combined_features.extract_engagement_features)
# is forced into every configuration from baseline_ext onward.
#
# Composition (chk = included, dot = not):
#   config                     | Antony | Ext | Eng | Mechanism
#   ---------------------------+--------+-----+-----+-----------
#   baseline_antony            |  chk   | dot | dot |    dot
#   baseline_antony_eng        |  chk   | dot | chk |    dot
#   baseline_ext               |  chk   | chk | chk |    dot
#   mechsig_<cluster>          |  chk   | chk | chk |  one cluster
#   mechsig_all                |  chk   | chk | chk |  all three + 5 non-cluster features
#   <name>_no_td               |  as <name>, minus the f_comp_td_* temporal-divergence composites
#   <name>_no_eng              |  as <name>, minus the engagement controls
#
# The seven primary configurations answer H1-H4 (thesis Table 3.2); the _no_td
# and _no_eng pairs are the leakage and ascertainment refits of thesis 3.9.
# Override from the shell with a comma-separated list, e.g.
#   PASC_ACTIVE_CONFIGS="baseline_ext,mechsig_all" python run_enhanced_mechsig.py
# =============================================================================
ACTIVE_CONFIGS = {
    # --- Primary configurations (thesis Table 3.2) ---
    "baseline_antony",            # H1: Antony et al. feature set
    "baseline_antony_eng",        # Antony + engagement controls only
    "baseline_ext",               # H2; reference for the mechanism lift
    "mechsig_viral",
    "mechsig_immuno",
    "mechsig_endo",
    "mechsig_all",                # H3 / H4
    # --- Leakage refit: composites removed (thesis 3.9) ---
    "baseline_ext_no_td",
    "mechsig_all_no_td",
    # --- Ascertainment refit: engagement controls stripped (thesis 3.9) ---
    "baseline_ext_no_eng",
    "mechsig_viral_no_eng",
    "mechsig_immuno_no_eng",
    "mechsig_endo_no_eng",
    "mechsig_all_no_eng",
}
_cfg_override = os.environ.get("PASC_ACTIVE_CONFIGS", "").strip()
if _cfg_override:
    ACTIVE_CONFIGS = {c.strip() for c in _cfg_override.split(",") if c.strip()}
    print(f"[active_configs] override via PASC_ACTIVE_CONFIGS -> {sorted(ACTIVE_CONFIGS)}")

for (ws, we) in FEATURE_WINDOWS:
    wl = window_label(ws, we)
    print(f"\n{'='*80}")
    print(f"ASSEMBLING — {wl} (max_window_end_day={we})")
    print(f"{'='*80}")

    if REEXTRACT_FEATURES:
        enh = all_enhanced_data[wl]

        # v2 leakage fix: gate td composites by the current window's end day
        # so a w0-21 model cannot draw on data from sub-windows ending after day 21.
        window_td_df = build_temporal_divergence_composites(
            windowed_binary=all_window_binaries,
            cohort_df=cohort_df,
            post_acute_days=30,
            max_window_end_day=we,
        )

        feature_df, feature_families = assemble_enhanced_feature_matrix(
            baseline_df=enh["baseline_df"],
            baseline_families=enh["baseline_families"],
            windowed_binary=enh["windowed_binary"],
            windowed_labs=enh["windowed_labs"],
            windowed_cbc=enh["windowed_cbc"],
            composite_df=enh["composite_df"],
            temporal_divergence_df=window_td_df,
            original_indicators=original_indicators,
            enhanced_indicators=enhanced_indicators,
        )
    else:
        # Reuse path: load the parquet + families.json saved by a previous run.
        cached_parquet = os.path.join(COHORT_DIR, f"enhanced_{wl}_{COHORT_MODE}_all_patients.parquet")
        cached_fams    = os.path.join(COHORT_DIR, f"enhanced_{wl}_{COHORT_MODE}_families.json")
        if not (os.path.exists(cached_parquet) and os.path.exists(cached_fams)):
            raise FileNotFoundError(
                f"REEXTRACT_FEATURES=False but cached artefacts missing for {wl}:\n"
                f"  {cached_parquet}\n  {cached_fams}\n"
                f"Re-run once with PASC_REEXTRACT=1 to populate."
            )
        feature_df = pd.read_parquet(cached_parquet)
        with open(cached_fams) as _fh:
            feature_families = json.load(_fh)
        print(f"  Loaded cached feature_df: {feature_df.shape} <- {cached_parquet}")
        print(f"  Loaded cached feature_families: {len(feature_families)} families <- {cached_fams}")

    all_configs = build_enhanced_model_configs(feature_families)
    _unknown = sorted(ACTIVE_CONFIGS - set(all_configs))
    if _unknown:
        raise KeyError(f"ACTIVE_CONFIGS names not built by build_enhanced_model_configs: {_unknown}. "
                       f"Available: {sorted(all_configs)}")
    model_configs = {k: v for k, v in all_configs.items() if k in ACTIVE_CONFIGS}
    print(f"  Active configs: {list(model_configs.keys())}")

    # Save the FULL feature matrix to parquet (only when fresh extraction
    # occurred) BEFORE any inpatient restriction, so the cached all_patients
    # artefacts stay complete and reusable.
    if REEXTRACT_FEATURES:
        out_path = os.path.join(COHORT_DIR, f"enhanced_{wl}_{COHORT_MODE}_all_patients.parquet")
        feature_df.to_parquet(out_path, index=False)
        print(f"  Saved: {out_path}")

        # Save feature-family grouping alongside parquet (consumed by run_lgbm_final.py, run_calibration_configs.py and the figure notebook)
        fam_path = os.path.join(COHORT_DIR, f"enhanced_{wl}_{COHORT_MODE}_families.json")
        with open(fam_path, "w") as _fh:
            json.dump({k: list(v) for k, v in feature_families.items()}, _fh, indent=2)
        print(f"  Saved: {fam_path}")

    # Withhold the index-date calendar feature from MODELING only (toggle).
    # Done AFTER the cache save above so the saved all_patients parquet +
    # families.json stay complete and the toggle is reversible without a
    # re-extract. Removes the column from feature_df, every feature_families
    # list, and every model_config feature list.
    if DROP_INDEX_CALENDAR:
        _dropped = []
        for _col in _CALENDAR_DROP_COLS:
            if _col in feature_df.columns:
                feature_df = feature_df.drop(columns=[_col])
                _dropped.append(_col)
            for _fam, _cols in feature_families.items():
                if _col in _cols:
                    feature_families[_fam] = [c for c in _cols if c != _col]
            for _cfg, _cols in model_configs.items():
                if _col in _cols:
                    model_configs[_cfg] = [c for c in _cols if c != _col]
        print(f"[drop_calendar] ON -- removed {_dropped or _CALENDAR_DROP_COLS} "
              f"from modeling (feature_df / families / configs)")
    else:
        print("[drop_calendar] OFF -- f_ext_index_year_month retained in modeling")

    assembled_data[wl] = {
        "feature_df": feature_df,
        "feature_families": feature_families,
        "model_configs": model_configs,
    }

# =============================================================================
# 7. MODELING PIPELINE
# =============================================================================
print("\n[7] Running modeling pipeline...")

all_pipeline_results = {}

# Force engagement controls past prevalence filter + Boruta so ascertainment
# adjustment is always present during model selection.
engagement_cols = None
for _wl, _ad in assembled_data.items():
    eng = _ad["feature_families"].get("engagement_controls", [])
    if eng:
        engagement_cols = eng
        break
if engagement_cols:
    print(f"Forced features (engagement controls): {engagement_cols}")
else:
    print("WARNING: No engagement_controls found — forced_features will be None")

# Sweep Boruta perc while holding RANDOM_STATE_BASE fixed -- identical splits,
# balanced subsamples and feature matrices across all perc values, so the only
# thing that changes between runs is the Boruta acceptance threshold. Each perc
# writes to its own results/.../perc{perc}/ subdir (non-destructive).
sweep_summary = []  # list of (perc, combined_df) for the final cross-perc table

# Extraction-only mode: every per-window feature parquet is saved by now (step 6).
# Skip the (login-node-hostile) modeling sweep so the DB re-extraction can run
# foreground with credentials, then model separately on a compute node with
# PASC_REEXTRACT=0.  Exits before the all_iterations.csv rebuild, so nothing is
# clobbered.
if os.environ.get("PASC_EXTRACT_ONLY", "0") == "1":
    print("\n[extract-only] PASC_EXTRACT_ONLY=1 -- all feature parquets saved; "
          "skipping modeling. Model separately with PASC_REEXTRACT=0.")
    raise SystemExit(0)

for perc in BORUTA_PERC_SWEEP:
    perc_results_dir = os.path.join(RESULTS_DIR, f"perc{perc}")
    os.makedirs(perc_results_dir, exist_ok=True)
    print("\n" + "#" * 80)
    print(f"# BORUTA PERC SWEEP — perc={perc}  ->  {perc_results_dir}")
    print("#" * 80)

    perc_iterations = []

    if RUN_MODELING:
        for wl, adata in assembled_data.items():
            feature_df = adata["feature_df"]
            ff = adata["feature_families"]
            model_configs = adata["model_configs"]

            for config_name, config_cols in model_configs.items():
                run_label = f"{wl}/{COHORT_MODE}/{config_name} (perc={perc})"
                print(f"\n{'#'*80}")
                print(f"# RUN: {run_label}  ({len(config_cols)} features)")
                print(f"{'#'*80}")

                run_dir = os.path.join(perc_results_dir, wl, COHORT_MODE)
                iter_path = os.path.join(run_dir, f"iterations_{config_name}.csv")
                if os.environ.get("PASC_SKIP_EXISTING", "0") == "1" and os.path.exists(iter_path):
                    print(f"  [skip] PASC_SKIP_EXISTING=1 and {iter_path} exists")
                    perc_iterations.append(pd.read_csv(iter_path))
                    continue

                results = run_full_pipeline(
                    feature_df=feature_df,
                    feature_families=ff,
                    cohort_name=run_label,
                    n_iterations=N_ITERATIONS,
                    model_types=MODEL_TYPES,
                    random_state_base=RANDOM_STATE_BASE,
                    boruta_max_iter=BORUTA_MAX_ITER,
                    boruta_n_estimators=BORUTA_N_ESTIMATORS,
                    compute_shap=COMPUTE_SHAP,
                    feature_columns=config_cols,
                    forced_features=engagement_cols,
                    boruta_perc=perc,
                )

                all_pipeline_results[(perc, wl, config_name)] = results

                os.makedirs(run_dir, exist_ok=True)
                iter_df = results_to_dataframe(results)
                iter_df["feature_window"] = wl
                iter_df["mode"] = COHORT_MODE
                iter_df["config"] = config_name
                iter_df["boruta_perc"] = perc
                iter_df.to_csv(iter_path, index=False)
                perc_iterations.append(iter_df)
                print(f"  Saved: {iter_path}")

                for model_type, pr in results.items():
                    feature_selections = []
                    for it in pr.iterations:
                        feature_selections.append({
                            "iteration": it.iteration,
                            "selected_features": list(it.selected_features),
                        })
                    features_path = os.path.join(run_dir, f"features_{config_name}_{model_type}.json")
                    with open(features_path, "w") as f:
                        json.dump(feature_selections, f, indent=2)
                    print(f"  Saved feature selections: {features_path}")

                    # Save SHAP values (mean |SHAP| per feature per iteration)
                    if COMPUTE_SHAP:
                        shap_rows = []
                        for it in pr.iterations:
                            if it.shap_values is not None and it.shap_feature_names is not None:
                                mean_abs = np.mean(np.abs(it.shap_values), axis=0)
                                for fname, val in zip(it.shap_feature_names, mean_abs):
                                    shap_rows.append({
                                        "iteration": it.iteration,
                                        "feature": fname,
                                        "mean_abs_shap": val,
                                    })
                        if shap_rows:
                            shap_path = os.path.join(run_dir, f"shap_{config_name}_{model_type}.csv")
                            pd.DataFrame(shap_rows).to_csv(shap_path, index=False)
                            print(f"  Saved SHAP values: {shap_path}")

        if perc_iterations:
            combined_df = pd.concat(perc_iterations, ignore_index=True)
            combined_path = os.path.join(perc_results_dir, "all_iterations.csv")
            combined_df.to_csv(combined_path, index=False)
            print(f"\nCombined iterations saved: {combined_path}")
        else:
            combined_df = pd.DataFrame()
    else:
        print(f"Modeling skipped (RUN_MODELING=False) for perc={perc}")
        combined_path = os.path.join(perc_results_dir, "all_iterations.csv")
        if os.path.exists(combined_path):
            combined_df = pd.read_csv(combined_path)
            print(f"Loaded existing results: {combined_path} ({len(combined_df)} rows)")
        else:
            combined_df = pd.DataFrame()

    sweep_summary.append((perc, combined_df))

# =============================================================================
# 8. MECHANISM-LIFT SUMMARY BY BORUTA STRICTNESS (per feature window)
# =============================================================================
print("\n[8] Generating mechanism-lift summary (per window)...")

# Reference baseline for the mechanism-lift delta. Mechsig configs are built as
# baseline_ext + mechanism columns (see enhanced_mech_signals.build_enhanced_model_configs),
# so the genuine mechanism lift is mechsig - baseline_ext (Delta_3 in the
# decomposition antony -> ext -> mechsig). This reference logic is unchanged.
LIFT_REFERENCE = "baseline_ext"
MECHSIG_CONFIGS = ["mechsig_viral", "mechsig_immuno", "mechsig_endo", "mechsig_all"]

def _median_metrics(cdf, window, config):
    """Return (median AUROC, median AUPRC, median #features) for one (window, config), or None."""
    if cdf is None or cdf.empty:
        return None
    sub = cdf[(cdf["feature_window"] == window) & (cdf["config"] == config)]
    if sub.empty:
        return None
    auroc = sub["AUROC"].values if "AUROC" in sub.columns else sub["auroc"].values
    auprc = sub["AUPRC"].values if "AUPRC" in sub.columns else sub["auprc"].values
    nfeat = (np.median(sub["n_features_boruta"].values)
             if "n_features_boruta" in sub.columns else np.nan)
    return float(np.median(auroc)), float(np.median(auprc)), float(nfeat)

# Per-(perc, window, config) median metrics, indexed for both tables.
metrics = {}  # (perc, window, config) -> (auroc, auprc, nfeat)
all_configs_seen = []
all_windows_seen = []
for perc, cdf in sweep_summary:
    if cdf is None or cdf.empty:
        continue
    for win in cdf["feature_window"].unique():
        if win not in all_windows_seen:
            all_windows_seen.append(win)
    for cfg in cdf["config"].unique():
        if cfg not in all_configs_seen:
            all_configs_seen.append(cfg)
    for win in cdf["feature_window"].unique():
        for cfg in cdf["config"].unique():
            m = _median_metrics(cdf, win, cfg)
            if m is not None:
                metrics[(perc, win, cfg)] = m

# Order configs: baselines first, then mechsig set, then anything else.
_preferred = ["baseline_antony", "baseline_ext"] + MECHSIG_CONFIGS
ordered_configs = [c for c in _preferred if c in all_configs_seen]
ordered_configs += [c for c in all_configs_seen if c not in ordered_configs]

# Order windows by (start, end) so 0_21, 0_30, ... read naturally.
def _win_key(w):
    try:
        a, b = w.replace("w", "").split("_")
        return (int(a), int(b))
    except Exception:
        return (9999, 9999)
ordered_windows = sorted(all_windows_seen, key=_win_key)

percs = [p for p, _ in sweep_summary]

for win in ordered_windows:
    # --- Table 1: median AUROC / AUPRC / #feat, one column-group per perc ---
    print("\n" + "=" * 120)
    print(f"SUMMARY — median metrics by config x Boruta perc  [window={win} / {COHORT_MODE}]")
    print("=" * 120)
    header = f"{'config':>22s}"
    for p in percs:
        header += f" | perc{p}: {'AUROC':>7s} {'AUPRC':>7s} {'#feat':>6s}"
    print(header)
    print("-" * 120)
    for cfg in ordered_configs:
        row = f"{cfg:>22s}"
        for p in percs:
            m = metrics.get((p, win, cfg))
            if m is None:
                row += f" | {'-':>26s}"
            else:
                row += f" | {'':>7s}{m[0]:7.4f} {m[1]:7.4f} {m[2]:6.1f}"
        print(row)
    print("=" * 120)

    # --- Table 2: mechanism-lift delta (mechsig - baseline_ext), AUPRC primary ---
    print(f"\nMECHANISM-LIFT TABLE — (config - {LIFT_REFERENCE}), one column per perc  "
          f"[window={win}; AUPRC = primary]")
    print("-" * 120)
    header = f"{'mechsig config':>22s}"
    for p in percs:
        header += f" | perc{p}: {'dAUPRC':>8s} {'dAUROC':>8s}"
    print(header)
    print("-" * 120)
    for cfg in MECHSIG_CONFIGS:
        if cfg not in all_configs_seen:
            continue
        row = f"{cfg:>22s}"
        for p in percs:
            m_cfg = metrics.get((p, win, cfg))
            m_ref = metrics.get((p, win, LIFT_REFERENCE))
            if m_cfg is None or m_ref is None:
                row += f" | {'-':>19s}"
            else:
                d_auprc = m_cfg[1] - m_ref[1]   # primary
                d_auroc = m_cfg[0] - m_ref[0]
                row += f" | {d_auprc:+8.4f} {d_auroc:+8.4f}"
        print(row)
    print("-" * 120)
    print(f"Reference = {LIFT_REFERENCE} (median AUPRC / AUROC per perc) [window={win}]:")
    for p in percs:
        m_ref = metrics.get((p, win, LIFT_REFERENCE))
        if m_ref is not None:
            print(f"  perc{p}: AUPRC={m_ref[1]:.4f}  AUROC={m_ref[0]:.4f}")
    print("=" * 120)

print("\n" + "=" * 80)
print("BATCH JOB COMPLETE")
print("=" * 80)

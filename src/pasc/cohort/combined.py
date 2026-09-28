"""
Cohort construction (thesis 3.2): COVID-19-positive patients with the PASC label and follow-up.

Two modes:
  - 'strict':  COVID-positive patients only (reuses antony_cohort logic exactly)
  - 'relaxed': Strict + non-COVID PACS patients (synthetic index = first_pacs_date - 60d)

Controls are ALWAYS COVID-positive in both modes; only the case population differs.

Does NOT modify any existing modules -- calls antony_cohort functions as-is and
adds new SQL for the relaxed-mode extension.
"""

import pandas as pd
import os

# Fixed ascertainment horizon for this data pull. Used as the default data
# cutoff everywhere instead of an auto-derived latest-observation date, so the
# censoring / future-index filter is reproducible across rebuilds and re-loads.
DATA_CUTOFF = pd.Timestamp("2026-05-28")

from pasc.cohort.antony import (
    CDM_SCHEMA,
    PACS_CONCEPTS,
    INPATIENT_VISIT_CONCEPTS,
    build_covid_positive_base,
    add_long_covid_labels,
    classify_inpatient_outpatient,
    compute_acute_window,
    add_demographics,
    upload_antony_cohort_temp,
    apply_death_and_followup,
    fetch_death_dates,
    _fetch_df,
    _drop_temp,
)


# ============================================================================
# Relaxed mode: find PACS patients WITHOUT documented COVID
# ============================================================================

def _find_non_covid_pacs_patients(cur, covid_person_ids):
    """
    Find patients with PACS diagnosis who are NOT in the COVID+ base.

    Returns DataFrame with: person_id, first_pacs_date
    """
    pacs_ids = ", ".join(str(c) for c in PACS_CONCEPTS)

    sql = f"""
    SELECT
        person_id,
        MIN(condition_start_date) AS first_pacs_date
    FROM {CDM_SCHEMA}.condition_occurrence
    WHERE condition_concept_id IN ({pacs_ids})
    GROUP BY person_id
    """
    all_pacs = _fetch_df(cur, sql, "Querying all PACS patients")
    all_pacs["first_pacs_date"] = pd.to_datetime(all_pacs["first_pacs_date"])

    # Filter to those NOT in COVID+ base
    non_covid = all_pacs[~all_pacs["person_id"].isin(covid_person_ids)].copy()
    print(f"  PACS patients without documented COVID: {len(non_covid):,}")
    return non_covid


def _build_non_covid_pacs_cohort(cur, non_covid_pacs_df, acute_days=21):
    """
    Build cohort rows for non-COVID PACS patients.

    Synthetic index = first_pacs_date - (acute_days + 39) days.
    The 39-day gap ensures the same margin between acute_end and
    first_pacs_date regardless of the chosen acute window.
    Uses the same hospitalization/acute-window logic as antony_cohort.

    Returns DataFrame with same columns as the strict cohort.
    """
    synthetic_offset = acute_days + 39  # 21d -> 60d offset (original), 60d -> 99d offset
    df = non_covid_pacs_df.copy()
    df["covid_index_date"] = df["first_pacs_date"] - pd.Timedelta(days=synthetic_offset)
    print(f"  Synthetic index offset: first_pacs_date - {synthetic_offset}d (acute_days={acute_days})")
    df["label"] = 1  # All are PACS cases by definition
    df["has_documented_covid"] = 0

    if df.empty:
        return df

    # Upload to temp table for hospitalization query
    _drop_temp(cur, "#combined_noncovid_base")
    cur.execute("""
    CREATE LOCAL TEMPORARY COLUMN TABLE "#combined_noncovid_base" (
        person_id BIGINT,
        covid_index_date DATE
    );
    """)
    cur.connection.commit()

    insert_sql = 'INSERT INTO "#combined_noncovid_base" (person_id, covid_index_date) VALUES (?, ?)'
    rows = [
        (int(r.person_id), r.covid_index_date.date() if pd.notna(r.covid_index_date) else None)
        for r in df.itertuples(index=False)
    ]
    batch_size = 5000
    for i in range(0, len(rows), batch_size):
        cur.executemany(insert_sql, rows[i:i + batch_size])
    cur.connection.commit()

    # Classify inpatient/outpatient using synthetic index
    visit_ids = ", ".join(str(c) for c in INPATIENT_VISIT_CONCEPTS)
    hosp_sql = f"""
    SELECT
        ab.person_id,
        MAX(COALESCE(vo.visit_end_date, vo.visit_start_date)) AS discharge_date
    FROM "#combined_noncovid_base" ab
    JOIN {CDM_SCHEMA}.visit_occurrence vo
      ON vo.person_id = ab.person_id
     AND vo.visit_concept_id IN ({visit_ids})
     AND vo.visit_start_date <= ADD_DAYS(ab.covid_index_date, 16)
     AND COALESCE(vo.visit_end_date, vo.visit_start_date) >= ADD_DAYS(ab.covid_index_date, -1)
    GROUP BY ab.person_id
    """
    hosp = _fetch_df(cur, hosp_sql, "Querying inpatient visits for non-COVID PACS")
    hosp["discharge_date"] = pd.to_datetime(hosp["discharge_date"])

    df = df.merge(hosp, on="person_id", how="left")
    df["is_inpatient"] = df["discharge_date"].notna().astype(int)

    # Compute acute window (same logic as antony_cohort)
    df["acute_start"] = df["covid_index_date"]
    default_end = df["covid_index_date"] + pd.Timedelta(days=acute_days)
    df["acute_end"] = default_end
    inpatient_mask = df["is_inpatient"] == 1
    if inpatient_mask.any():
        df.loc[inpatient_mask, "acute_end"] = df.loc[inpatient_mask].apply(
            lambda r: max(r["covid_index_date"] + pd.Timedelta(days=acute_days), r["discharge_date"])
            if pd.notna(r["discharge_date"]) else r["covid_index_date"] + pd.Timedelta(days=acute_days),
            axis=1,
        )

    # Effect-based death handling (these are all PACS cases): exclude only
    # deaths up to acute_end (which compromise the feature window). Deaths
    # after the acute window do not invalidate an already-ascertained case.
    deaths = fetch_death_dates(cur)
    df = df.merge(deaths, on="person_id", how="left")
    df["death_date"] = pd.to_datetime(df["death_date"])
    drop_mask = df["death_date"].notna() & (df["death_date"] <= pd.to_datetime(df["acute_end"]))
    n_drop = int(drop_mask.sum())
    df = df[~drop_mask].copy()
    print(f"  Excluded {n_drop:,} non-COVID PACS patients with death <= acute_end")
    # These rows are positives (events); annotate for schema parity with the
    # strict cohort so concatenation does not introduce silent NaNs.
    df["insufficient_followup"] = False
    df["eligibility_status"] = "positive"
    df["death_bucket"] = "non_covid_pacs_case"

    # Demographics
    demo_sql = f"""
    SELECT
        ab.person_id,
        FLOOR(DAYS_BETWEEN(p.birth_datetime, ab.covid_index_date) / 365.25) AS age_at_index,
        p.gender_concept_id
    FROM "#combined_noncovid_base" ab
    JOIN {CDM_SCHEMA}.person p
      ON p.person_id = ab.person_id
    """
    demo = _fetch_df(cur, demo_sql, "Querying demographics for non-COVID PACS")
    df = df.merge(demo, on="person_id", how="left")
    df["age_at_index"] = pd.to_numeric(df["age_at_index"], errors="coerce")

    _drop_temp(cur, "#combined_noncovid_base")

    return df


# ============================================================================
# Leakage validation
# ============================================================================

def _validate_no_leakage(df, acute_days=21):
    """
    For non-COVID PACS patients, verify acute_end < first_pacs_date.
    Synthetic index = first_pacs_date - (acute_days + 39)d, so acute_end
    should be at most first_pacs_date - 39d (for outpatient with any window).
    """
    non_covid = df[df["has_documented_covid"] == 0].copy()
    if non_covid.empty:
        return

    non_covid["acute_end_dt"] = pd.to_datetime(non_covid["acute_end"])
    non_covid["pacs_dt"] = pd.to_datetime(non_covid["first_pacs_date"])
    leaking = non_covid[non_covid["acute_end_dt"] >= non_covid["pacs_dt"]]

    if len(leaking) > 0:
        print(f"  WARNING: {len(leaking)} non-COVID PACS patients have acute_end >= first_pacs_date!")
        print("           These patients may have temporal leakage and should be investigated.")
        # For inpatients with very long stays, the acute_end could extend past PACS date.
        # We cap it to first_pacs_date - 1 to prevent leakage.
        cap_mask = (df["has_documented_covid"] == 0) & (pd.to_datetime(df["acute_end"]) >= pd.to_datetime(df["first_pacs_date"]))
        if cap_mask.any():
            df.loc[cap_mask, "acute_end"] = pd.to_datetime(df.loc[cap_mask, "first_pacs_date"]) - pd.Timedelta(days=1)
            print(f"  Capped {cap_mask.sum()} patients' acute_end to first_pacs_date - 1")
    else:
        print("  Leakage check passed: all non-COVID PACS patients have acute_end < first_pacs_date")


def _validate_temporal_integrity(df, acute_days=21, washout_days=0, strict=True):
    """
    Target-trial-emulation (TTE) integrity checks on the assembled cohort.

    Implements the TTE checklist (Hernán & Robins, *What If* Ch. 22) translated
    to the structure of this cohort. All checks operate on the final cohort
    DataFrame (post date-filter, pre-feature-extraction). Pure validation —
    does not modify the DataFrame. On violation, prints a structured violation
    report and (if `strict=True`) raises `AssertionError`.

    Checks:
      1. Eligibility uniformity — every row has a valid `covid_index_date`
         on or after the COVID-19 emergence date (2020-01-01).
      2. Exposure / feature window strictly pre/peri-index —
         `acute_start == covid_index_date` for every patient
         (no patient's feature window starts after their index date).
      3. Outcome ascertainment uniformity —
         `(acute_end - covid_index_date).days` is constant within
         each is_inpatient stratum-level group OR equals `acute_days`
         (allows inpatient-extended windows but flags unexpected variance).
      4. Follow-up start = `covid_index_date + washout_days` —
         for cases (`label == 1`) the first PASC date is on or after
         `covid_index_date + washout_days` (no immortal-time inflation;
         no outcome events counted before follow-up begins).
      5. Outcome strictly post-acute — `first_pacs_date > acute_end` for
         every case (no outcome events inside the feature window).
      6. No silent eligibility drops — patients flagged with
         `insufficient_followup` (if column present) are retained in the
         returned df with a reason logged, not silently filtered.

    Args:
        df: Cohort DataFrame after `build_combined_cohorts` finalization.
        acute_days: Nominal acute-window length passed to cohort construction.
        washout_days: Days between `covid_index_date` and follow-up start.
            Default 0 to match current pipeline (no explicit washout); set
            higher in sensitivity analyses.
        strict: If True (default), raise `AssertionError` on any violation.
            Set False for diagnostic-only runs.

    Returns:
        violations: dict mapping check-name → list of violation records
        (empty dict if all checks pass).
    """
    violations = {}

    df_check = df.copy()
    df_check["covid_index_date"] = pd.to_datetime(df_check["covid_index_date"])
    df_check["acute_start"] = pd.to_datetime(df_check["acute_start"])
    df_check["acute_end"] = pd.to_datetime(df_check["acute_end"])
    if "first_pacs_date" in df_check.columns:
        df_check["first_pacs_date"] = pd.to_datetime(df_check["first_pacs_date"])

    # --- 1. Eligibility uniformity ---
    bad_idx = df_check[df_check["covid_index_date"].isna()
                       | (df_check["covid_index_date"] < pd.Timestamp("2020-01-01"))]
    if len(bad_idx) > 0:
        violations["eligibility_index_date"] = bad_idx[["person_id", "covid_index_date"]].head(10).to_dict("records")

    # --- 2. Exposure window strictly pre/peri-index ---
    bad_start = df_check[df_check["acute_start"] > df_check["covid_index_date"]]
    if len(bad_start) > 0:
        violations["exposure_window_starts_after_index"] = (
            bad_start[["person_id", "covid_index_date", "acute_start"]].head(10).to_dict("records")
        )

    # --- 3. Outcome ascertainment uniformity ---
    df_check["acute_span_days"] = (
        df_check["acute_end"] - df_check["covid_index_date"]
    ).dt.days
    # Allow inpatient-extended (>= acute_days). Flag only acute_span < acute_days.
    bad_span = df_check[df_check["acute_span_days"] < acute_days - 1]  # -1 for date-arith jitter
    if len(bad_span) > 0:
        violations["outcome_window_short"] = (
            bad_span[["person_id", "acute_span_days"]].head(10).to_dict("records")
        )

    # --- 4. Follow-up start = covid_index_date + washout ---
    if "first_pacs_date" in df_check.columns and "label" in df_check.columns:
        cases = df_check[df_check["label"] == 1].dropna(subset=["first_pacs_date"])
        if len(cases) > 0:
            follow_up_start = cases["covid_index_date"] + pd.Timedelta(days=washout_days)
            bad_followup = cases[cases["first_pacs_date"] < follow_up_start]
            if len(bad_followup) > 0:
                violations["outcome_before_followup_start"] = (
                    bad_followup[["person_id", "covid_index_date",
                                  "first_pacs_date"]].head(10).to_dict("records")
                )

    # --- 5. Outcome strictly post-acute ---
    if "first_pacs_date" in df_check.columns and "label" in df_check.columns:
        cases = df_check[df_check["label"] == 1].dropna(subset=["first_pacs_date"])
        # For non-COVID PACS in relaxed mode, _validate_no_leakage already
        # caps acute_end to first_pacs_date - 1, so this should hold post-cap.
        bad_outcome = cases[cases["first_pacs_date"] <= cases["acute_end"]]
        if len(bad_outcome) > 0:
            violations["outcome_inside_feature_window"] = (
                bad_outcome[["person_id", "acute_end",
                             "first_pacs_date"]].head(10).to_dict("records")
            )

    # --- 6. No silent drops ---
    if "insufficient_followup" in df_check.columns:
        n_flagged = int(df_check["insufficient_followup"].sum())
        if n_flagged > 0:
            print(f"  [TTE] {n_flagged} patients flagged with insufficient_followup "
                  f"(retained for explicit downstream handling).")

    # --- Report ---
    if violations:
        print("=" * 80)
        print("TTE TEMPORAL-INTEGRITY VIOLATIONS")
        print("=" * 80)
        for check, records in violations.items():
            print(f"  [{check}] {len(records)} example(s) (first 10):")
            for r in records:
                print(f"    {r}")
        print("=" * 80)
        if strict:
            raise AssertionError(
                f"_validate_temporal_integrity: {len(violations)} TTE check(s) failed "
                f"({list(violations)}). Re-run with strict=False for diagnostic-only."
            )
    else:
        print(f"  [TTE] Temporal-integrity checks passed "
              f"(acute_days={acute_days}, washout_days={washout_days}, N={len(df_check):,})")

    return violations


# ============================================================================
# Orchestrator
# ============================================================================

def build_combined_cohorts(
    cur,
    mode="strict",
    run_all_patients=True,
    run_inpatients=False,
    run_outpatients=False,
    save_dir=None,
    acute_days=21,
    validate_tte=True,
    washout_days=0,
    tte_strict=True,
    min_days_post_acute=90,
    min_followup_days=90,
    data_cutoff=None,
):
    """
    Build cohorts for the combined analysis.

    Args:
        cur: Database cursor
        mode: 'strict' (COVID+ only) or 'relaxed' (COVID+ plus non-COVID PACS)
        run_all_patients: Include all_patients sub-cohort
        run_inpatients: Include inpatients sub-cohort
        run_outpatients: Include outpatients sub-cohort
        save_dir: Directory for parquet output
        acute_days: Number of days for the acute phase window (default: 21)
        min_days_post_acute: Minimum days after index for a PASC diagnosis to
            count as label=1 (default: 90)
        min_followup_days: Minimum observed follow-up for a control to count as
            a *confirmed negative*; controls below this are censored
            (insufficient_followup=True) rather than treated as negatives
            (default: 90)
        data_cutoff: Ascertainment horizon used for censoring. If None, it is
            derived as the latest observation_period end date in the source.

    Returns:
        dict of sub-cohort name -> DataFrame
    """
    print("=" * 80)
    print(f"COMBINED COHORT CONSTRUCTION — mode={mode}, acute_days={acute_days}")
    print("=" * 80)

    # Determine the ascertainment horizon (data cutoff) once, up front, so it
    # is shared by the death/follow-up censoring and the future-index clip.
    # Defaults to the fixed DATA_CUTOFF (28 May 2026 data pull) for
    # reproducibility rather than an auto-derived latest-observation date.
    if data_cutoff is None:
        data_cutoff = DATA_CUTOFF
    data_cutoff = pd.Timestamp(data_cutoff)
    print(f"[cutoff] Ascertainment horizon (data_cutoff) = {data_cutoff.date()}")

    # Step 1: Build strict COVID+ cohort (reuse antony_cohort exactly).
    # Death handling is effect-based (apply_death_and_followup) instead of a
    # blanket exclusion, and requires labels + acute window to be present first.
    base = build_covid_positive_base(cur)
    base = add_long_covid_labels(cur, base, min_days_post_acute=min_days_post_acute)
    base = classify_inpatient_outpatient(cur, base)
    base = compute_acute_window(base, acute_days=acute_days)
    base = apply_death_and_followup(
        cur, base, acute_days=acute_days,
        min_followup_days=min_followup_days, data_cutoff=data_cutoff,
    )
    base = add_demographics(cur, base)
    base["has_documented_covid"] = 1

    covid_person_ids = set(base["person_id"].tolist())

    if mode == "relaxed":
        print("\n[Relaxed mode] Adding non-COVID PACS patients...")
        non_covid_pacs = _find_non_covid_pacs_patients(cur, covid_person_ids)

        if len(non_covid_pacs) > 0:
            extra = _build_non_covid_pacs_cohort(cur, non_covid_pacs, acute_days=acute_days)
            if len(extra) > 0:
                # Deduplicate: prefer COVID+ version
                extra = extra[~extra["person_id"].isin(covid_person_ids)].copy()
                base = pd.concat([base, extra], ignore_index=True)
                _validate_no_leakage(base, acute_days=acute_days)
                print(f"  Added {len(extra):,} non-COVID PACS patients")
        else:
            print("  No non-COVID PACS patients found")

    # Filter: exclude cases before 2020 (COVID-19 did not exist before 2020)
    # and after the ascertainment horizon (future-dated index = data artifact).
    base["covid_index_date"] = pd.to_datetime(base["covid_index_date"])
    n_before = len(base)
    base = base[
        (base["covid_index_date"] >= "2020-01-01")
        & (base["covid_index_date"] <= data_cutoff)
    ].copy()
    n_excluded = n_before - len(base)
    if n_excluded > 0:
        print(f"\n[Date filter] Excluded {n_excluded:,} patients with index date "
              f"before 2020 or after data cutoff {data_cutoff.date()}")
        print(f"  Remaining: {len(base):,} patients")

    # TTE temporal-integrity assertions (Hernán & Robins, What If Ch. 22).
    # Runs on the finalized cohort prior to feature extraction. Default strict
    # so leakage / immortal-time violations surface immediately. Set
    # tte_strict=False for diagnostic-only runs that should not raise.
    if validate_tte:
        print("\n[TTE] Validating temporal integrity...")
        _validate_temporal_integrity(
            base,
            acute_days=acute_days,
            washout_days=washout_days,
            strict=tte_strict,
        )

    # Upload to temp table (needed for feature extraction)
    upload_antony_cohort_temp(cur, base)

    # Build sub-cohorts
    cohorts = {}
    if run_all_patients:
        cohorts["all_patients"] = base.copy()
    if run_inpatients:
        cohorts["inpatients"] = base[base["is_inpatient"] == 1].copy().reset_index(drop=True)
    if run_outpatients:
        cohorts["outpatients"] = base[base["is_inpatient"] == 0].copy().reset_index(drop=True)

    # Summary
    print("\n" + "=" * 80)
    print(f"COHORT SUMMARY (mode={mode})")
    print("=" * 80)
    for name, df in cohorts.items():
        n_pos = df["label"].sum()
        n_neg = len(df) - n_pos
        prev = n_pos / len(df) * 100 if len(df) > 0 else 0
        n_covid = df["has_documented_covid"].sum() if "has_documented_covid" in df.columns else len(df)
        print(f"  {name:20s} | N={len(df):>10,} | pos={n_pos:>8,} | neg={n_neg:>10,} | "
              f"prev={prev:.2f}% | covid+={n_covid:,}")
    print("=" * 80)

    # Save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, df in cohorts.items():
            path = os.path.join(save_dir, f"combined_{mode}_{name}.parquet")
            df.to_parquet(path, index=False)
            print(f"  Saved: {path}")

    return cohorts

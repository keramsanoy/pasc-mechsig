"""
Cohort construction for the Antony et al. replication.

Replicates:
- COVID-positive base population (U07.1 dx + positive PCR/antigen tests)
- Death exclusion
- Long COVID label (PACS concepts 600588/600589 as proxy for U09.9)
- Inpatient vs outpatient classification (hospitalized within [-1, +16] of index)
- Acute phase window (outpatient: 0-21d; inpatient: 0-max(21, discharge))
- Three sub-cohorts: all_patients, inpatients, outpatients

Documented deviations from Antony et al.:
- Label: PACS concepts 600588/600589 instead of ICD-10-CM U09.9
- Single-site only (Mount Sinai); no site restriction logic needed
"""

import pandas as pd
import numpy as np
import os

from pasc.config.omop import CDM_SCHEMA  # OMOP schema name; set OMOP_CDM_SCHEMA to override

# Earliest plausible COVID-19 index date (disease did not exist before 2020).
COVID_EMERGENCE_DATE = "2020-01-01"

# Concept IDs
COVID_CONDITION_CONCEPT = 37311061          # COVID-19 (maps from U07.1)
PACS_CONCEPTS = (600588, 600589)            # Post-acute sequelae of COVID-19
POSITIVE_VALUE_CONCEPTS = (45884084, 45877985)  # Positive / Detected
SARS_COV2_TEST_CONCEPTS = (706169, 586526, 706170, 706163, 723476)
INPATIENT_VISIT_CONCEPTS = (9201, 262)      # Inpatient Visit, ER+Inpatient


def _execute_query(cur, sql, description=""):
    """Execute a SQL statement, printing description if provided."""
    if description:
        print(f"  {description}...", flush=True)
    cur.execute(sql)
    cur.connection.commit()


def _fetch_df(cur, sql, description=""):
    """Execute a SELECT and return a DataFrame with lowercased columns."""
    if description:
        print(f"  {description}...", flush=True)
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0].lower() for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def _drop_temp(cur, name):
    """Drop a HANA local temp table if it exists."""
    try:
        cur.execute(f'DROP TABLE "{name}";')
        cur.connection.commit()
    except Exception:
        cur.connection.rollback()


# ============================================================================
# Step 1: COVID-positive base population
# ============================================================================

def build_covid_positive_base(cur):
    """
    Build the COVID-positive base population.

    A patient is COVID-positive if they have:
      (a) at least one condition_occurrence with concept 37311061 (U07.1), OR
      (b) at least one positive SARS-CoV-2 PCR or antigen test.

    Index date = earliest COVID-positive date across both sources.

    Returns:
        DataFrame with columns: person_id, covid_index_date
    """
    print("\n[Step 1] Building COVID-positive base population...")

    test_concepts = ", ".join(str(c) for c in SARS_COV2_TEST_CONCEPTS)
    pos_values = ", ".join(str(c) for c in POSITIVE_VALUE_CONCEPTS)

    sql = f"""
    SELECT
        person_id,
        MIN(event_date) AS covid_index_date
    FROM (
        -- Source A: COVID-19 diagnosis
        SELECT person_id, condition_start_date AS event_date
        FROM {CDM_SCHEMA}.condition_occurrence
        WHERE condition_concept_id = {COVID_CONDITION_CONCEPT}

        UNION ALL

        -- Source B: positive SARS-CoV-2 lab test
        SELECT person_id, measurement_date AS event_date
        FROM {CDM_SCHEMA}.measurement
        WHERE measurement_concept_id IN ({test_concepts})
          AND value_as_concept_id IN ({pos_values})
    ) covid_events
    WHERE event_date IS NOT NULL
    GROUP BY person_id
    """

    base = _fetch_df(cur, sql, "Querying COVID-positive patients")
    base["covid_index_date"] = pd.to_datetime(base["covid_index_date"])
    print(f"  COVID-positive base population: {len(base):,} patients")
    return base


# ============================================================================
# Step 2: Exclude deceased patients
# ============================================================================

def exclude_deceased(cur, base_df):
    """
    Exclude patients with a death record in CDMPHI.death.

    Returns:
        Filtered DataFrame
    """
    print("\n[Step 2] Excluding deceased patients...")

    death_sql = f"SELECT DISTINCT person_id FROM {CDM_SCHEMA}.death"
    deceased = _fetch_df(cur, death_sql, "Querying death table")

    n_before = len(base_df)
    base_df = base_df[~base_df["person_id"].isin(deceased["person_id"])].copy()
    n_excluded = n_before - len(base_df)
    print(f"  Excluded {n_excluded:,} deceased patients")
    print(f"  Remaining: {len(base_df):,} patients")
    return base_df


# ============================================================================
# Step 2b: Effect-based death handling + follow-up / censoring annotation
# ============================================================================

def fetch_death_dates(cur):
    """First death date per person from CDMPHI.death (read-only).

    Detects whether the table exposes ``death_date`` or ``death_datetime``.
    Returns a DataFrame with columns: person_id, death_date.
    """
    cols = _fetch_df(
        cur,
        f"""
        SELECT COLUMN_NAME
        FROM SYS.TABLE_COLUMNS
        WHERE SCHEMA_NAME = '{CDM_SCHEMA}' AND TABLE_NAME = 'DEATH'
        """,
        "Inspecting death table columns",
    )
    names = {str(c).lower() for c in cols["column_name"]}
    date_col = "death_date" if "death_date" in names else "death_datetime"
    df = _fetch_df(
        cur,
        f"SELECT person_id, MIN({date_col}) AS death_date "
        f"FROM {CDM_SCHEMA}.death GROUP BY person_id",
        "Querying death dates",
    )
    df["death_date"] = pd.to_datetime(df["death_date"])
    return df


def fetch_observation_periods(cur):
    """Earliest start / latest end of observation_period per person (read-only).

    Returns a DataFrame with columns: person_id, obs_start, obs_end.
    """
    df = _fetch_df(
        cur,
        f"""
        SELECT person_id,
               MIN(observation_period_start_date) AS obs_start,
               MAX(observation_period_end_date)   AS obs_end
        FROM {CDM_SCHEMA}.observation_period
        GROUP BY person_id
        """,
        "Querying observation_period",
    )
    df["obs_start"] = pd.to_datetime(df["obs_start"])
    df["obs_end"] = pd.to_datetime(df["obs_end"])
    return df


def robust_data_cutoff(obs_end_series, upper_quantile=0.999):
    """Latest trustworthy observation-end date (ascertainment horizon).

    Ignores far-future artifacts (e.g. year-2098 placeholder dates that
    contaminate the raw maximum) by discarding values above ``upper_quantile``
    before taking the max. Returns a pandas Timestamp.
    """
    vals = pd.to_datetime(obs_end_series).dropna()
    if vals.empty:
        return pd.Timestamp.today().normalize()
    cap = vals.quantile(upper_quantile)
    trimmed = vals[vals <= cap]
    return trimmed.max() if not trimmed.empty else vals.max()


def apply_death_and_followup(cur, base_df, acute_days=21, min_followup_days=90,
                             data_cutoff=None):
    """Effect-based death exclusion + follow-up / censoring annotation.

    Replaces the blanket ``exclude_deceased`` drop. A death is treated as an
    *exclusion* only when it actually compromises eligibility or feature
    ascertainment; otherwise the patient is retained, and controls whose
    outcome could not be fully ascertained are *censored* (not silently
    labelled negative).

    Requires ``base_df`` to already contain: person_id, covid_index_date,
    acute_end, label, first_pacs_date.

    Death buckets (relative to index / acute window):
      a  death <= covid_index_date          -> EXCLUDE (ineligible)
      b  index < death <= acute_end         -> EXCLUDE (incomplete features)
      d  death after a qualifying PASC dx    -> KEEP (true positive)
      *  any other post-acute death          -> KEEP (censored if control with
                                                short follow-up; see below)

    Follow-up / censoring (applied to *retained* patients):
      last_obs_date = min(obs_end, death_date, data_cutoff)
      followup_days = (last_obs_date - covid_index_date).days
      A control (label==0) is a *confirmed negative* iff
      followup_days >= min_followup_days; otherwise it is *censored*
      (insufficient_followup=True) and must not be treated as a negative
      downstream. Cases (label==1) are always events.

    Adds columns: death_date, obs_end, last_obs_date, followup_days,
    death_bucket, insufficient_followup, eligibility_status.

    Returns the DataFrame with buckets a+b removed.
    """
    print("\n[Step 2b] Effect-based death handling + follow-up annotation...")

    df = base_df.copy()
    deaths = fetch_death_dates(cur)
    obs = fetch_observation_periods(cur)
    df = df.merge(deaths, on="person_id", how="left")
    df = df.merge(obs[["person_id", "obs_end"]], on="person_id", how="left")

    for c in ("covid_index_date", "acute_end", "first_pacs_date", "death_date", "obs_end"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])

    idx = df["covid_index_date"]
    acute_end = df["acute_end"]
    death = df["death_date"]
    pacs = df["first_pacs_date"]
    is_case = df["label"] == 1
    has_death = death.notna()

    # Ascertainment horizon (data cutoff). Default: robust latest observation
    # end (ignores far-future placeholder dates).
    if data_cutoff is None:
        data_cutoff = robust_data_cutoff(df["obs_end"])
    data_cutoff = pd.Timestamp(data_cutoff)
    print(f"  Data cutoff (ascertainment horizon): {data_cutoff.date()}")

    # --- Bucket assignment (priority order) ---
    bucket = pd.Series("alive", index=df.index, dtype="object")
    a = has_death & (death <= idx)
    b = has_death & (death > idx) & (death <= acute_end)
    d = has_death & is_case & pacs.notna() & (death >= pacs)
    rest_death = has_death & ~a & ~b & ~d
    bucket[a] = "a_pre_index_ineligible"
    bucket[b] = "b_in_acute_window"
    bucket[d] = "d_post_pasc_true_positive"
    bucket[rest_death] = "post_acute_death"
    df["death_bucket"] = bucket

    # --- Drop only justified exclusions (a + b) ---
    drop_mask = a | b
    n_a, n_b = int(a.sum()), int(b.sum())
    df = df[~drop_mask].copy()
    n_retained_dead = int(df["death_bucket"].isin(
        ["d_post_pasc_true_positive", "post_acute_death"]).sum())
    print(f"  Excluded (a) death <= index:        {n_a:,}")
    print(f"  Excluded (b) death in acute window: {n_b:,}")
    print(f"  Retained deceased (post-acute):     {n_retained_dead:,} "
          f"(vs blanket rule which would drop all {n_a + n_b + n_retained_dead:,})")

    # --- Follow-up / censoring on retained patients ---
    # last observed = earliest of observation end, death, and the data cutoff.
    cutoff_series = pd.Series(data_cutoff, index=df.index)
    candidates = pd.concat([df["obs_end"], df["death_date"], cutoff_series], axis=1)
    df["last_obs_date"] = candidates.min(axis=1)
    df["followup_days"] = (df["last_obs_date"] - df["covid_index_date"]).dt.days

    # Confirmed-negative gate: controls need >= min_followup_days of observation.
    is_control = df["label"] == 0
    df["insufficient_followup"] = (
        is_control & (df["followup_days"] < min_followup_days)
    ).fillna(False).astype(bool)

    status = pd.Series("confirmed_negative", index=df.index, dtype="object")
    status[df["label"] == 1] = "positive"
    status[df["insufficient_followup"]] = "censored"
    df["eligibility_status"] = status

    n_censored = int(df["insufficient_followup"].sum())
    n_pos = int((df["label"] == 1).sum())
    n_conf_neg = int((status == "confirmed_negative").sum())
    print(f"  Min follow-up for confirmed negative: {min_followup_days}d")
    print(f"  Positives (events):                  {n_pos:,}")
    print(f"  Confirmed negatives:                 {n_conf_neg:,}")
    print(f"  Censored controls (insufficient fu): {n_censored:,}")
    print(f"  Remaining after a+b drop:            {len(df):,}")
    return df


# ============================================================================
# Step 3: Add long COVID labels
# ============================================================================

def add_long_covid_labels(cur, base_df, min_days_post_acute=90):
    """
    Add long COVID labels.

    Label = 1 if patient has a PACS diagnosis (600588/600589)
    at least *min_days_post_acute* days after the covid_index_date.
    Label = 0 otherwise.

    NOTE: The Antony et al. replication uses 21 days (pass
    ``min_days_post_acute=21`` to reproduce the original paper).
    The default of 90 days is a deliberate departure: it prevents any
    temporal overlap between feature-extraction windows (≤ 90 d) and the
    PASC outcome window, eliminating indirect leakage that the 21-day
    threshold permits when feature windows extend past day 21.

    Parameters
    ----------
    cur : DB cursor
    base_df : DataFrame with person_id, covid_index_date
    min_days_post_acute : int
        Minimum days after COVID index for a PASC diagnosis to count as
        label=1.  Default 90 avoids temporal overlap between feature
        windows (≤ 90 d) and the PASC outcome window.  Use 21 to match
        the Antony et al. paper, or 30/60 for other sensitivity analyses.

    Returns:
        DataFrame with added columns: label, first_pacs_date
    """
    print(f"\n[Step 3] Adding long COVID labels (min_days={min_days_post_acute})...")

    pacs_ids = ", ".join(str(c) for c in PACS_CONCEPTS)
    sql = f"""
    SELECT
        person_id,
        MIN(condition_start_date) AS first_pacs_date
    FROM {CDM_SCHEMA}.condition_occurrence
    WHERE condition_concept_id IN ({pacs_ids})
    GROUP BY person_id
    """
    pacs = _fetch_df(cur, sql, "Querying PACS diagnoses")
    pacs["first_pacs_date"] = pd.to_datetime(pacs["first_pacs_date"])

    base_df = base_df.merge(pacs, on="person_id", how="left")

    # Calculate days between COVID and PASC diagnosis
    days_to_pasc = (base_df["first_pacs_date"] - base_df["covid_index_date"]).dt.days

    # Label=1 if PACS occurs >= min_days_post_acute after COVID index date
    base_df["label"] = (
        base_df["first_pacs_date"].notna()
        & (days_to_pasc >= min_days_post_acute)
    ).astype(int)

    # Statistics on excluded acute-phase diagnoses
    acute_phase_excluded = (
        (base_df["first_pacs_date"].notna())
        & (days_to_pasc < min_days_post_acute)
    ).sum()

    n_pos = base_df["label"].sum()
    n_neg = len(base_df) - n_pos
    prevalence = n_pos / len(base_df) * 100 if len(base_df) > 0 else 0
    print(f"  Total PACS diagnoses found: {base_df['first_pacs_date'].notna().sum():,}")
    print(f"  PACS during acute phase (<{min_days_post_acute}d) excluded: {acute_phase_excluded:,}")
    print(f"  Label=1 (long COVID): {n_pos:,}")
    print(f"  Label=0 (controls):   {n_neg:,}")
    print(f"  Prevalence:           {prevalence:.2f}%")
    return base_df


# ============================================================================
# Step 4: Classify inpatient vs outpatient
# ============================================================================

def classify_inpatient_outpatient(cur, base_df):
    """
    Classify patients as inpatient or outpatient.

    Inpatient: hospitalized from 1 day before through 16 days after
    the COVID index date. Outpatient: all others.

    Also computes discharge_date for inpatients.

    Returns:
        DataFrame with added columns: is_inpatient, discharge_date
    """
    print("\n[Step 4] Classifying inpatient vs outpatient...")

    # Upload person_id + covid_index_date to temp table for join
    _drop_temp(cur, "#antony_base")
    cur.execute("""
    CREATE LOCAL TEMPORARY COLUMN TABLE "#antony_base" (
        person_id BIGINT,
        covid_index_date DATE
    );
    """)
    cur.connection.commit()

    insert_sql = 'INSERT INTO "#antony_base" (person_id, covid_index_date) VALUES (?, ?)'
    rows = [
        (int(r.person_id), r.covid_index_date.date() if pd.notna(r.covid_index_date) else None)
        for r in base_df.itertuples(index=False)
    ]
    # Batch insert
    batch_size = 5000
    for i in range(0, len(rows), batch_size):
        cur.executemany(insert_sql, rows[i:i + batch_size])
    cur.connection.commit()
    print(f"  Uploaded {len(rows):,} patients to temp table")

    visit_ids = ", ".join(str(c) for c in INPATIENT_VISIT_CONCEPTS)
    sql = f"""
    SELECT
        ab.person_id,
        MAX(COALESCE(vo.visit_end_date, vo.visit_start_date)) AS discharge_date
    FROM "#antony_base" ab
    JOIN {CDM_SCHEMA}.visit_occurrence vo
      ON vo.person_id = ab.person_id
     AND vo.visit_concept_id IN ({visit_ids})
     AND vo.visit_start_date <= ADD_DAYS(ab.covid_index_date, 16)
     AND COALESCE(vo.visit_end_date, vo.visit_start_date) >= ADD_DAYS(ab.covid_index_date, -1)
    GROUP BY ab.person_id
    """
    hosp = _fetch_df(cur, sql, "Querying inpatient visits")
    hosp["discharge_date"] = pd.to_datetime(hosp["discharge_date"])

    base_df = base_df.merge(hosp, on="person_id", how="left")
    base_df["is_inpatient"] = base_df["discharge_date"].notna().astype(int)

    n_inp = base_df["is_inpatient"].sum()
    n_out = len(base_df) - n_inp
    print(f"  Inpatients:  {n_inp:,}")
    print(f"  Outpatients: {n_out:,}")
    return base_df


# ============================================================================
# Step 5: Compute acute phase window
# ============================================================================

def compute_acute_window(df, acute_days=21, window_start=0):
    """
    Compute the acute phase window per patient.

    Outpatients: index_date + window_start to index_date + acute_days
    Inpatients:  index_date + window_start to max(index_date + acute_days, discharge_date)

    Args:
        df: DataFrame with covid_index_date, is_inpatient, discharge_date
        acute_days: Number of days for the acute phase window end (default: 21)
        window_start: Day offset (relative to index) for the window start
            (default: 0). Use >0 for onset-shifted feature windows, e.g.
            window_start=30, acute_days=60 -> days 30..60 post index.

    Returns:
        DataFrame with added columns: acute_start, acute_end
    """
    print(f"\n[Step 5] Computing acute phase windows "
          f"(window_start={window_start}, acute_days={acute_days})...")

    df["acute_start"] = df["covid_index_date"] + pd.Timedelta(days=window_start)
    default_end = df["covid_index_date"] + pd.Timedelta(days=acute_days)

    # For inpatients: extend to discharge if longer than acute_days
    df["acute_end"] = default_end
    inpatient_mask = df["is_inpatient"] == 1
    if inpatient_mask.any():
        df.loc[inpatient_mask, "acute_end"] = df.loc[inpatient_mask].apply(
            lambda r: max(r["covid_index_date"] + pd.Timedelta(days=acute_days), r["discharge_date"])
            if pd.notna(r["discharge_date"]) else r["covid_index_date"] + pd.Timedelta(days=acute_days),
            axis=1,
        )

    acute_lengths = (df["acute_end"] - df["acute_start"]).dt.days
    print(f"  Acute window length — median: {acute_lengths.median():.0f}d, "
          f"max: {acute_lengths.max():.0f}d, min: {acute_lengths.min():.0f}d")
    return df


# ============================================================================
# Step 6: Add demographics (age + gender)
# ============================================================================

def add_demographics(cur, base_df):
    """
    Add age at COVID index date and gender_concept_id.

    Returns:
        DataFrame with added columns: age_at_index, gender_concept_id
    """
    print("\n[Step 6] Adding demographics...")

    sql = f"""
    SELECT
        ab.person_id,
        FLOOR(DAYS_BETWEEN(p.birth_datetime, ab.covid_index_date) / 365.25) AS age_at_index,
        p.gender_concept_id
    FROM "#antony_base" ab
    JOIN {CDM_SCHEMA}.person p
      ON p.person_id = ab.person_id
    """
    demo = _fetch_df(cur, sql, "Querying demographics")

    base_df = base_df.merge(demo, on="person_id", how="left")
    base_df["age_at_index"] = pd.to_numeric(base_df["age_at_index"], errors="coerce")

    print(f"  Age — median: {base_df['age_at_index'].median():.0f}, "
          f"mean: {base_df['age_at_index'].mean():.1f}")
    return base_df


# ============================================================================
# Step 7: Upload final cohort to temp table for feature extraction
# ============================================================================

def upload_antony_cohort_temp(cur, cohort_df):
    """
    Create temp table '#antony_cohort' with all fields needed by feature queries.

    Columns: person_id, covid_index_date, label, is_inpatient,
             acute_start, acute_end, discharge_date,
             age_at_index, gender_concept_id, first_pacs_date
    """
    _drop_temp(cur, "#antony_cohort")

    cur.execute("""
    CREATE LOCAL TEMPORARY COLUMN TABLE "#antony_cohort" (
        person_id         BIGINT,
        covid_index_date  DATE,
        label             INTEGER,
        is_inpatient      INTEGER,
        acute_start       DATE,
        acute_end         DATE,
        discharge_date    DATE,
        age_at_index      INTEGER,
        gender_concept_id INTEGER,
        first_pacs_date   DATE
    );
    """)
    cur.connection.commit()

    insert_sql = """
    INSERT INTO "#antony_cohort" (
        person_id, covid_index_date, label, is_inpatient,
        acute_start, acute_end, discharge_date,
        age_at_index, gender_concept_id, first_pacs_date
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    def _to_date(v):
        if pd.isna(v):
            return None
        return pd.Timestamp(v).date()

    def _to_int(v, default=None):
        if pd.isna(v):
            return default
        return int(v)

    rows = []
    for r in cohort_df.itertuples(index=False):
        rows.append((
            int(r.person_id),
            _to_date(r.covid_index_date),
            _to_int(r.label, 0),
            _to_int(r.is_inpatient, 0),
            _to_date(r.acute_start),
            _to_date(r.acute_end),
            _to_date(r.discharge_date),
            _to_int(r.age_at_index),
            _to_int(r.gender_concept_id),
            _to_date(r.first_pacs_date),
        ))

    batch_size = 5000
    for i in range(0, len(rows), batch_size):
        cur.executemany(insert_sql, rows[i:i + batch_size])
    cur.connection.commit()
    print(f"  Uploaded {len(rows):,} patients to #antony_cohort temp table")


# ============================================================================
# Orchestrator
# ============================================================================

def build_all_antony_cohorts(cur, save_dir=None,
                             min_days_post_acute=90):
    """
    Build all three Antony-style sub-cohorts.

    Args:
        cur: Database cursor.
        save_dir: Directory to save parquet files.  Pass *None* to skip.
        min_days_post_acute: Minimum days after COVID index for a PASC
            diagnosis to count as label=1.  Default 90 (thesis).
            Use 28 for strict Antony replication.

    Returns:
        dict with keys 'all_patients', 'inpatients', 'outpatients',
        each mapping to a DataFrame.
    """
    print("=" * 80)
    print("ANTONY ET AL. COHORT CONSTRUCTION")
    print("=" * 80)

    # 1. COVID-positive base
    base = build_covid_positive_base(cur)

    # 2. Exclude deceased
    base = exclude_deceased(cur, base)

    # 3. Long COVID labels
    base = add_long_covid_labels(cur, base, min_days_post_acute=min_days_post_acute)

    # 4. Inpatient/outpatient classification
    base = classify_inpatient_outpatient(cur, base)

    # 5. Acute phase windows
    base = compute_acute_window(base)

    # 6. Demographics
    base = add_demographics(cur, base)

    # 7. Upload to temp table for downstream feature queries
    upload_antony_cohort_temp(cur, base)

    # 8. Build sub-cohorts
    cohorts = {
        "all_patients": base.copy(),
        "inpatients": base[base["is_inpatient"] == 1].copy().reset_index(drop=True),
        "outpatients": base[base["is_inpatient"] == 0].copy().reset_index(drop=True),
    }

    print("\n" + "=" * 80)
    print("COHORT SUMMARY")
    print("=" * 80)
    for name, df in cohorts.items():
        n_pos = df["label"].sum()
        n_neg = len(df) - n_pos
        prev = n_pos / len(df) * 100 if len(df) > 0 else 0
        print(f"  {name:20s} | N={len(df):>10,} | pos={n_pos:>8,} | neg={n_neg:>10,} | prev={prev:.2f}%")
    print("=" * 80)

    # Save to parquet
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, df in cohorts.items():
            path = os.path.join(save_dir, f"antony_cohort_{name}.parquet")
            df.to_parquet(path, index=False)
            print(f"  Saved: {path}")

    return cohorts


# ============================================================================
# Cross-site: care-site assignment (stub for Antony Fig 6 analogue)
# ============================================================================

def extract_care_site(cur, cohort_df):
    """
    Assign a dominant care_site_id to each patient based on the most
    frequent inpatient/outpatient visit location around the COVID index.

    This enables a within-MSHS cross-site analysis analogous to
    Antony's Fig 6 (train on partner 1, test on partners 2–39).

    Returns:
        cohort_df with an added ``care_site_id`` column (int or NaN).
    """
    print("\n[Cross-site] Extracting dominant care_site per patient...")

    sql = f"""
    SELECT
        c.person_id,
        vo.care_site_id,
        COUNT(*) AS n_visits
    FROM "#antony_cohort" c
    JOIN {CDM_SCHEMA}.visit_occurrence vo
      ON vo.person_id = c.person_id
     AND vo.visit_start_date BETWEEN ADD_DAYS(c.covid_index_date, -365)
                                 AND ADD_DAYS(c.covid_index_date,  365)
     AND vo.care_site_id IS NOT NULL
    GROUP BY c.person_id, vo.care_site_id
    """
    visits = _fetch_df(cur, sql, "Querying care_site visits")

    if visits.empty:
        print("  WARNING: No care_site data found")
        cohort_df["care_site_id"] = np.nan
        return cohort_df

    # Keep the care_site with the most visits per patient
    idx = visits.groupby("person_id")["n_visits"].idxmax()
    dominant = visits.loc[idx, ["person_id", "care_site_id"]]

    cohort_df = cohort_df.merge(dominant, on="person_id", how="left")
    n_sites = cohort_df["care_site_id"].nunique()
    n_assigned = cohort_df["care_site_id"].notna().sum()
    print(f"  Assigned care_site to {n_assigned:,}/{len(cohort_df):,} patients "
          f"across {n_sites} sites")
    return cohort_df

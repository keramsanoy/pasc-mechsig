"""
Unified feature extraction: Antony et al. families A-E, extended covariates, engagement controls, mechanism indicators.

Wraps three layers:
  1. Antony baseline (families A-E) via antony_features.py functions (unchanged)
  2. Extended baseline features (BMI/BP/lifestyle, acute severity, variant wave,
     vaccination, additional symptoms, has_documented_covid flag) — new SQL
     referencing '#antony_cohort' temp table
  3. Mechanistic signals via mech_signals_common.build_signals (unchanged)

Does NOT modify any existing modules.
"""

import pandas as pd
import numpy as np

from antony_features import (
    extract_comorbidities,
    extract_acute_symptoms,
    extract_acute_symptoms_hpo,
    extract_acute_drugs,
    extract_demographics,
    extract_treatment_measures,
    _fetch_df,
)
from antony_cohort import upload_antony_cohort_temp
from mech_signals_common import prepare_temp_cohort, build_signals
from indicator_definitions import (
    VIRAL_INDICATOR_LIST,
    IMMUNO_INDICATOR_LIST,
    ENDO_INDICATOR_LIST,
)

from omop_config import CDM_SCHEMA  # OMOP schema name; set OMOP_CDM_SCHEMA to override

# Concept IDs (from feature_engineering.py — duplicated here to avoid modifying it)
BMI_CONCEPT_ID = 3038553
SBP_CONCEPT_ID = 3004249
DBP_CONCEPT_ID = 3012888
SMOKING_OBS_CONCEPT_ID = 1585856
ALCOHOL_OBS_CONCEPT_ID = 1586197
SPO2_MEAS_CONCEPT_ID = 3013502
VENT_MODE_OBS_CONCEPT_ID = 3004921
ICU_OBS_CONCEPT_ID = 1259883
ICU_MEAS_CONCEPT_ID = 706367
INPATIENT_VISIT_CONCEPT_IDS = (9201, 262)

COVID_VAX_ANY_CVX = [
    724907, 724906, 702866, 702678, 724904, 702676, 724905,
    702664, 702672, 702679, 702677, 702666, 905420
]
COVID_VAX_MRNA_CVX = [724907, 724906, 702678, 702676, 702677, 905420]
COVID_VAX_VECTOR_CVX = [702866, 724905]

# Serology (SARS-CoV-2 antibody) concept IDs
SARS_COV2_AB_LOINCS = [
    "94661-6",   # SARS-CoV-2 IgG+IgM (primary serostatus)
    "94563-4",   # SARS-CoV-2 IgG
    "94769-7",   # SARS-CoV-2 total Ab titer (quantitative)
    "94505-5",   # SARS-CoV-2 IgG titer (quantitative)
    "94762-2",   # SARS-CoV-2 Ab (secondary positivity)
    "94564-2",   # SARS-CoV-2 IgM
]
# Positive / Negative / Equivocal interpretation concept IDs (LOINC Answer,
# domain 'Meas Value'). Verified against CDMPHI.concept on 2026-04-25.
# Prior versions had 45877985 ("Detected") miscoded as Negative — fixed here.
SARS_COV2_AB_POSITIVE_CONCEPT_IDS = (
    45884084,  # Positive       (LA6576-8)
    45877985,  # Detected       (LA11882-0)
    45881802,  # Reactive       (LA15255-5)
)
SARS_COV2_AB_NEGATIVE_CONCEPT_IDS = (
    45878583,  # Negative       (LA6577-6)
    45880296,  # Not detected   (LA11883-8)
    45884092,  # Nonreactive    (LA15256-3)
)
SARS_COV2_AB_EQUIVOCAL_CONCEPT_IDS = (
    45884087,  # Equivocal      (LA11885-3)
    45884091,  # Indeterminate  (LA11884-6)
    45877990,  # Inconclusive   (LA9663-1)
)

# Anti-N (nucleocapsid) and anti-S (spike) specific LOINCs for N-vs-S contrast.
# Anti-N rises only after natural infection; anti-S rises after vaccination OR
# infection. High anti-N with low anti-S suggests recent/persistent antigen
# (unusual given most MSHS patients are vaccinated → high anti-S baseline).
# If a LOINC is absent from the local data, the feature will simply be zero
# with a missing indicator at the assembly step.
SARS_COV2_N_AB_LOINCS = [
    "94720-0",   # SARS-CoV-2 N Ab
    "94504-8",   # SARS-CoV-2 N IgG
    "94761-4",   # SARS-CoV-2 N IgG titer
    "94506-3",   # SARS-CoV-2 N IgM
    "96118-2",   # SARS-CoV-2 N IgG (Abbott Architect, widely used at MSHS)
]
SARS_COV2_S_AB_LOINCS = [
    "94509-7",   # SARS-CoV-2 S IgG
    "94551-9",   # SARS-CoV-2 S IgG titer
    "96831-0",   # SARS-CoV-2 S1 IgG
    "94507-1",   # SARS-CoV-2 S IgM
    "94505-5",   # SARS-CoV-2 IgG titer (S-based in most quant assays)
]


# ============================================================================
# Extended Feature: BMI / BP / Lifestyle (pre-index window)
# ============================================================================

def extract_ext_bmi_bp_lifestyle(cur):
    """
    Extract BMI, blood pressure, and lifestyle features from the pre-index
    window (on or before covid_index_date).

    References '#antony_cohort' temp table.

    Returns:
        DataFrame with person_id + feature columns.
    """
    print("\n[Ext] Extracting BMI/BP/lifestyle features...")

    sql = f"""
    WITH cohort AS (
        SELECT person_id, covid_index_date
        FROM "#antony_cohort"
    ),

    bmi_ranked AS (
        SELECT
            c.person_id,
            CAST(m.value_as_number AS DOUBLE) AS bmi_value,
            ROW_NUMBER() OVER (
                PARTITION BY c.person_id
                ORDER BY m.measurement_date DESC, m.measurement_id DESC
            ) AS rn
        FROM cohort c
        JOIN {CDM_SCHEMA}.measurement m
          ON m.person_id = c.person_id
         AND m.measurement_date <= c.covid_index_date
         AND m.measurement_concept_id = {BMI_CONCEPT_ID}
         AND m.value_as_number IS NOT NULL
    ),
    bmi_last AS (
        SELECT person_id, bmi_value AS f_ext_bmi_last
        FROM bmi_ranked WHERE rn = 1
    ),

    sbp_ranked AS (
        SELECT
            c.person_id,
            CAST(m.value_as_number AS DOUBLE) AS sbp_value,
            ROW_NUMBER() OVER (
                PARTITION BY c.person_id
                ORDER BY m.measurement_date DESC, m.measurement_id DESC
            ) AS rn
        FROM cohort c
        JOIN {CDM_SCHEMA}.measurement m
          ON m.person_id = c.person_id
         AND m.measurement_date <= c.covid_index_date
         AND m.measurement_concept_id = {SBP_CONCEPT_ID}
         AND m.value_as_number IS NOT NULL
    ),
    sbp_last AS (
        SELECT person_id, sbp_value AS f_ext_sbp_last
        FROM sbp_ranked WHERE rn = 1
    ),

    dbp_ranked AS (
        SELECT
            c.person_id,
            CAST(m.value_as_number AS DOUBLE) AS dbp_value,
            ROW_NUMBER() OVER (
                PARTITION BY c.person_id
                ORDER BY m.measurement_date DESC, m.measurement_id DESC
            ) AS rn
        FROM cohort c
        JOIN {CDM_SCHEMA}.measurement m
          ON m.person_id = c.person_id
         AND m.measurement_date <= c.covid_index_date
         AND m.measurement_concept_id = {DBP_CONCEPT_ID}
         AND m.value_as_number IS NOT NULL
    ),
    dbp_last AS (
        SELECT person_id, dbp_value AS f_ext_dbp_last
        FROM dbp_ranked WHERE rn = 1
    ),

    life_flags AS (
        SELECT
            c.person_id,
            MAX(CASE WHEN o.observation_concept_id = {SMOKING_OBS_CONCEPT_ID}
                       OR LOWER(o.observation_source_value) LIKE '%smok%'
                       OR LOWER(o.observation_source_value) LIKE '%tobacco%'
                 THEN 1 ELSE 0 END) AS f_ext_smoking_any,
            MAX(CASE WHEN o.observation_concept_id = {ALCOHOL_OBS_CONCEPT_ID}
                       OR LOWER(o.observation_source_value) LIKE '%alcohol%'
                 THEN 1 ELSE 0 END) AS f_ext_alcohol_any
        FROM cohort c
        LEFT JOIN {CDM_SCHEMA}.observation o
          ON o.person_id = c.person_id
         AND o.observation_date <= c.covid_index_date
         AND (o.observation_concept_id IN ({SMOKING_OBS_CONCEPT_ID}, {ALCOHOL_OBS_CONCEPT_ID})
              OR LOWER(o.observation_source_value) LIKE '%smok%'
              OR LOWER(o.observation_source_value) LIKE '%tobacco%'
              OR LOWER(o.observation_source_value) LIKE '%alcohol%')
        GROUP BY c.person_id
    )

    SELECT
        c.person_id,
        b.f_ext_bmi_last,
        s.f_ext_sbp_last,
        d.f_ext_dbp_last,
        lf.f_ext_smoking_any,
        lf.f_ext_alcohol_any
    FROM cohort c
    LEFT JOIN bmi_last b  ON b.person_id = c.person_id
    LEFT JOIN sbp_last s  ON s.person_id = c.person_id
    LEFT JOIN dbp_last d  ON d.person_id = c.person_id
    LEFT JOIN life_flags lf ON lf.person_id = c.person_id
    """

    df = _fetch_df(cur, sql, "Querying BMI/BP/lifestyle")
    print(f"  Extracted ext BMI/BP/lifestyle for {len(df):,} patients")
    return df


# ============================================================================
# Extended Feature: Acute Severity
# ============================================================================

def extract_ext_acute_severity(cur):
    """
    Extract acute COVID severity features (hospitalization, ICU, SpO2,
    ventilation) from the acute phase window.

    References '#antony_cohort' temp table.

    Returns:
        DataFrame with person_id + severity feature columns.
    """
    print("\n[Ext] Extracting acute severity features...")

    inp_visit_ids = ", ".join(str(x) for x in INPATIENT_VISIT_CONCEPT_IDS)

    sql = f"""
    WITH cohort AS (
        SELECT person_id, covid_index_date, acute_start, acute_end, is_inpatient
        FROM "#antony_cohort"
    ),

    acute_inpatient_visits AS (
        SELECT
            c.person_id,
            vo.visit_occurrence_id,
            vo.care_site_id,
            vo.visit_start_date,
            COALESCE(vo.visit_end_date, vo.visit_start_date) AS visit_end_date,
            (DAYS_BETWEEN(vo.visit_start_date, COALESCE(vo.visit_end_date, vo.visit_start_date)) + 1) AS los_days
        FROM cohort c
        JOIN {CDM_SCHEMA}.visit_occurrence vo
          ON vo.person_id = c.person_id
         AND vo.visit_concept_id IN ({inp_visit_ids})
         AND vo.visit_start_date <= c.acute_end
         AND COALESCE(vo.visit_end_date, vo.visit_start_date) >= c.acute_start
    ),

    hosp_agg AS (
        SELECT
            c.person_id,
            CASE WHEN COUNT(aiv.visit_occurrence_id) > 0 THEN 1 ELSE 0 END AS f_ext_hosp_acute,
            MAX(aiv.los_days) AS f_ext_los_max,
            COUNT(aiv.visit_occurrence_id) AS f_ext_n_admissions
        FROM cohort c
        LEFT JOIN acute_inpatient_visits aiv
          ON aiv.person_id = c.person_id
        GROUP BY c.person_id
    ),

    icu_obs AS (
        SELECT
            c.person_id,
            MAX(CASE WHEN o.observation_concept_id = {ICU_OBS_CONCEPT_ID}
                       OR LOWER(o.observation_source_value) LIKE '%icu%'
                 THEN 1 ELSE 0 END) AS f_ext_icu_obs
        FROM cohort c
        LEFT JOIN acute_inpatient_visits aiv
          ON aiv.person_id = c.person_id
        LEFT JOIN {CDM_SCHEMA}.observation o
          ON o.person_id = c.person_id
         AND o.observation_date BETWEEN aiv.visit_start_date AND aiv.visit_end_date
         AND (o.observation_concept_id = {ICU_OBS_CONCEPT_ID}
              OR LOWER(o.observation_source_value) LIKE '%icu%')
        GROUP BY c.person_id
    ),

    icu_care_site_meas AS (
        SELECT
            c.person_id,
            MAX(
                CASE
                    WHEN m.measurement_concept_id = {ICU_MEAS_CONCEPT_ID} THEN 1
                    ELSE 0
                END
            ) AS f_ext_icu_care_site
        FROM cohort c
        LEFT JOIN acute_inpatient_visits aiv
          ON aiv.person_id = c.person_id
        LEFT JOIN {CDM_SCHEMA}.measurement m
          ON m.person_id = c.person_id
         AND m.measurement_date BETWEEN aiv.visit_start_date AND aiv.visit_end_date
         AND m.measurement_concept_id = {ICU_MEAS_CONCEPT_ID}
        GROUP BY c.person_id
    ),

    icu_care_site_name AS (
        SELECT
            c.person_id,
            MAX(
                CASE
                    WHEN LOWER(cs.care_site_name) LIKE '%icu%'
                      OR LOWER(cs.care_site_name) LIKE '%intensive%'
                      OR LOWER(cs.care_site_name) LIKE '%critical care%'
                      OR LOWER(cs.care_site_name) LIKE '%micu%'
                      OR LOWER(cs.care_site_name) LIKE '%sicu%'
                      OR LOWER(cs.care_site_name) LIKE '%ccu%'
                    THEN 1 ELSE 0
                END
            ) AS f_ext_icu_site_kw
        FROM cohort c
        LEFT JOIN acute_inpatient_visits aiv
          ON aiv.person_id = c.person_id
        LEFT JOIN {CDM_SCHEMA}.visit_occurrence vo
          ON vo.visit_occurrence_id = aiv.visit_occurrence_id
        LEFT JOIN {CDM_SCHEMA}.care_site cs
          ON cs.care_site_id = vo.care_site_id
        GROUP BY c.person_id
    ),

    spo2_agg AS (
        SELECT
            c.person_id,
            MAX(CASE WHEN m.measurement_concept_id = {SPO2_MEAS_CONCEPT_ID} THEN 1 ELSE 0 END) AS f_ext_spo2_any,
            MIN(CAST(m.value_as_number AS DOUBLE)) AS f_ext_spo2_min
        FROM cohort c
        LEFT JOIN acute_inpatient_visits aiv ON aiv.person_id = c.person_id
        LEFT JOIN {CDM_SCHEMA}.measurement m
          ON m.person_id = c.person_id
         AND m.measurement_date BETWEEN aiv.visit_start_date AND aiv.visit_end_date
         AND m.measurement_concept_id = {SPO2_MEAS_CONCEPT_ID}
         AND m.value_as_number IS NOT NULL
        GROUP BY c.person_id
    ),

    vent_mode AS (
        SELECT
            c.person_id,
            MAX(CASE WHEN o.observation_concept_id = {VENT_MODE_OBS_CONCEPT_ID} THEN 1 ELSE 0 END) AS f_ext_vent_any
        FROM cohort c
        LEFT JOIN acute_inpatient_visits aiv ON aiv.person_id = c.person_id
        LEFT JOIN {CDM_SCHEMA}.observation o
          ON o.person_id = c.person_id
         AND o.observation_date BETWEEN aiv.visit_start_date AND aiv.visit_end_date
         AND o.observation_concept_id = {VENT_MODE_OBS_CONCEPT_ID}
        GROUP BY c.person_id
    )

    SELECT
        c.person_id,
        COALESCE(h.f_ext_hosp_acute, 0) AS f_ext_hosp_acute,
        COALESCE(h.f_ext_n_admissions, 0) AS f_ext_n_admissions,
        CASE WHEN COALESCE(h.f_ext_hosp_acute, 0) = 1 THEN h.f_ext_los_max ELSE 0 END AS f_ext_los_max,
        COALESCE(io.f_ext_icu_obs, 0) AS f_ext_icu_obs,
        COALESCE(ics.f_ext_icu_care_site, 0) AS f_ext_icu_care_site,
        COALESCE(icsk.f_ext_icu_site_kw, 0) AS f_ext_icu_site_kw,
        CASE
            WHEN COALESCE(io.f_ext_icu_obs, 0) = 1
              OR COALESCE(ics.f_ext_icu_care_site, 0) = 1
              OR COALESCE(icsk.f_ext_icu_site_kw, 0) = 1
            THEN 1 ELSE 0
        END AS f_ext_icu_any,
        COALESCE(sa.f_ext_spo2_any, 0) AS f_ext_spo2_any,
        sa.f_ext_spo2_min,
        COALESCE(vm.f_ext_vent_any, 0) AS f_ext_vent_any
    FROM cohort c
    LEFT JOIN hosp_agg h          ON h.person_id = c.person_id
    LEFT JOIN icu_obs io          ON io.person_id = c.person_id
    LEFT JOIN icu_care_site_meas ics ON ics.person_id = c.person_id
    LEFT JOIN icu_care_site_name icsk ON icsk.person_id = c.person_id
    LEFT JOIN spo2_agg sa         ON sa.person_id = c.person_id
    LEFT JOIN vent_mode vm        ON vm.person_id = c.person_id
    """

    df = _fetch_df(cur, sql, "Querying acute severity")
    df = df.drop_duplicates(subset=["person_id"])
    print(f"  Extracted ext acute severity for {len(df):,} patients")
    return df


# ============================================================================
# Extended Feature: Variant Wave (pure pandas, no SQL)
# ============================================================================

def add_ext_variant_wave(cohort_df):
    """
    Add variant wave proxy features based on covid_index_date.

    Wave periods based on NYC serosurvey data from:
      Carreño, J. M. et al. (2024). SARS-CoV-2 serosurvey across multiple waves
      of the COVID-19 pandemic in New York City between 2020–2023.
      Nature Communications, 15, 5847. https://doi.org/10.1038/s41467-024-50052-2

    | Wave | Period                    | Dominant Variants                      |
    |------|---------------------------|----------------------------------------|
    |  1   | Feb 9 – Aug 30, 2020      | Ancestral SARS-CoV-2, D614G           |
    |  2   | Aug 31, 2020 – Jun 20, 2021 | Iota (B.1.529), Alpha (B.1.1.7)    |
    |  3   | Jun 21 – Oct 31, 2021     | Delta (B.1.617.2)                      |
    |  4   | Nov 1, 2021 – Mar 6, 2022 | Delta (B.1.617.2), Omicron BA.1       |
    |  5   | Mar 7 – Jul 18, 2022      | Omicron BA.2, BA.5                     |
    | F/U  | Aug 21 – Oct 2, 2023      | Not wave-assigned (follow-up)          |
    """
    print("\n[Ext] Adding variant wave features...")

    df = cohort_df.copy()
    idx = pd.to_datetime(df["covid_index_date"], errors="coerce")

    # NOTE (2026-04 ablation): f_ext_wave_unclassified REMOVED.
    # Rationale: the five classified wave dummies only cover 2020-02-09 through
    # 2022-07-18, so f_ext_wave_unclassified == 1 is effectively "indexed after
    # mid-2022" (with a negligible pre-Feb-2020 sliver). That is not a variant
    # proxy; it conflates three label-correlated confounders:
    #   (a) shorter observable follow-up time for late-indexed patients,
    #   (b) U09.9 / PASC coding adoption (introduced Oct 2021, uptake rose
    #       through 2022–2023),
    #   (c) MSHS care-pattern era (telehealth mix, Long COVID clinics).
    # SHAP on baseline_ext at w0_21 ranked f_ext_wave_unclassified as the
    # #2 most important feature (mean |SHAP| ≈ 0.038, above f_age and every
    # mechanistic signal). Era/follow-up should be controlled via dedicated
    # continuous features (index_year, followup_days) rather than smuggled in
    # through an "other" wave bin. Patients outside the five defined wave
    # periods now simply have all wave dummies = 0.
    wave_cols = [
        "f_ext_wave_ancestral_1",
        "f_ext_wave_iota_alpha_2",
        "f_ext_wave_delta_3",
        "f_ext_wave_delta_omicron_4",
        "f_ext_wave_omicron_ba2_ba5_5",
    ]
    for c in wave_cols:
        df[c] = 0

    # Wave 1: Feb 9 – Aug 30, 2020  (Ancestral SARS-CoV-2, D614G)
    df.loc[(idx >= "2020-02-09") & (idx <= "2020-08-30"), "f_ext_wave_ancestral_1"] = 1
    # Wave 2: Aug 31, 2020 – Jun 20, 2021  (Iota B.1.529, Alpha B.1.1.7)
    df.loc[(idx >= "2020-08-31") & (idx <= "2021-06-20"), "f_ext_wave_iota_alpha_2"] = 1
    # Wave 3: Jun 21 – Oct 31, 2021  (Delta B.1.617.2)
    df.loc[(idx >= "2021-06-21") & (idx <= "2021-10-31"), "f_ext_wave_delta_3"] = 1
    # Wave 4: Nov 1, 2021 – Mar 6, 2022  (Delta B.1.617.2, Omicron BA.1)
    df.loc[(idx >= "2021-11-01") & (idx <= "2022-03-06"), "f_ext_wave_delta_omicron_4"] = 1
    # Wave 5: Mar 7 – Jul 18, 2022  (Omicron BA.2, BA.5)
    df.loc[(idx >= "2022-03-07") & (idx <= "2022-07-18"), "f_ext_wave_omicron_ba2_ba5_5"] = 1
    # Patients outside all five wave periods keep all wave dummies = 0
    # (no "unclassified" bin — see rationale comment above).

    # Continuous calendar-time control: index year + fractional month in a
    # single monotonic feature (e.g. Jan 2020 -> 2020.0, Jul 2021 -> 2021.5).
    # Unlike the five wave dummies (which only span 2020-02 .. 2022-07 and are
    # all-zero afterwards), this covers the full index range and gives the
    # model an explicit handle on era/follow-up/PASC-coding-adoption drift
    # rather than leaving late-indexed patients calendar-blind.
    df["f_ext_index_year_month"] = idx.dt.year + (idx.dt.month - 1) / 12.0
    # Guard against NaT index dates (should not occur given the >=2020-01-01
    # eligibility filter, but keep the model NaN-safe): fall back to the
    # cohort median calendar value.
    n_missing = int(df["f_ext_index_year_month"].isna().sum())
    if n_missing:
        median_cal = df["f_ext_index_year_month"].median()
        df["f_ext_index_year_month"] = df["f_ext_index_year_month"].fillna(median_cal)
        print(f"  [calendar] {n_missing} rows had missing index date; "
              f"filled f_ext_index_year_month with median {median_cal:.3f}")

    wave_cols = wave_cols + ["f_ext_index_year_month"]
    print(f"  Added {len(wave_cols)} variant wave + calendar features")
    return df, wave_cols


# ============================================================================
# Extended Feature: Vaccination
# ============================================================================

def extract_ext_vaccination(cur):
    """
    Extract pre-index vaccination features.
    References '#antony_cohort' temp table.
    """
    print("\n[Ext] Extracting vaccination features...")

    def _sql_in(int_list):
        return ", ".join(str(x) for x in int_list)

    sql = f"""
    WITH cohort AS (
        SELECT person_id, covid_index_date
        FROM "#antony_cohort"
    ),

    vax_any AS (
        SELECT concept_id FROM {CDM_SCHEMA}.concept
        WHERE concept_id IN ({_sql_in(COVID_VAX_ANY_CVX)})
    ),

    vax_mrna AS (
        SELECT concept_id FROM {CDM_SCHEMA}.concept
        WHERE concept_id IN ({_sql_in(COVID_VAX_MRNA_CVX)})
    ),

    vax_vector AS (
        SELECT concept_id FROM {CDM_SCHEMA}.concept
        WHERE concept_id IN ({_sql_in(COVID_VAX_VECTOR_CVX)})
    ),

    vax_raw AS (
        SELECT
            c.person_id,
            de.drug_exposure_start_date AS vax_date,
            CASE WHEN vm.concept_id IS NOT NULL
                   OR LOWER(de.drug_source_value) LIKE '%pfizer%'
                   OR LOWER(de.drug_source_value) LIKE '%biontech%'
                   OR LOWER(de.drug_source_value) LIKE '%comirnaty%'
                   OR LOWER(de.drug_source_value) LIKE '%moderna%'
                   OR LOWER(de.drug_source_value) LIKE '%spikevax%'
                 THEN 1 ELSE 0 END AS has_mrna,
            CASE WHEN vv.concept_id IS NOT NULL
                   OR LOWER(de.drug_source_value) LIKE '%janssen%'
                   OR LOWER(de.drug_source_value) LIKE '%johnson%'
                   OR LOWER(de.drug_source_value) LIKE '%astrazeneca%'
                   OR LOWER(de.drug_source_value) LIKE '%vaxzevria%'
                 THEN 1 ELSE 0 END AS has_vector
        FROM cohort c
        JOIN {CDM_SCHEMA}.drug_exposure de
          ON de.person_id = c.person_id
         AND de.drug_exposure_start_date < c.covid_index_date
         AND (de.drug_concept_id IN (SELECT concept_id FROM vax_any)
              OR LOWER(de.drug_source_value) LIKE '%covid%vaccin%'
              OR LOWER(de.drug_source_value) LIKE '%pfizer%'
              OR LOWER(de.drug_source_value) LIKE '%moderna%'
              OR LOWER(de.drug_source_value) LIKE '%comirnaty%'
              OR LOWER(de.drug_source_value) LIKE '%spikevax%'
              OR LOWER(de.drug_source_value) LIKE '%janssen%'
              OR LOWER(de.drug_source_value) LIKE '%johnson%'
              OR LOWER(de.drug_source_value) LIKE '%novavax%')
        LEFT JOIN vax_mrna vm   ON vm.concept_id = de.drug_concept_id
        LEFT JOIN vax_vector vv ON vv.concept_id = de.drug_concept_id
    ),

    dose_days AS (
        SELECT
            person_id, vax_date,
            MAX(has_mrna) AS has_mrna,
            MAX(has_vector) AS has_vector
        FROM vax_raw
        GROUP BY person_id, vax_date
    ),

    agg AS (
        SELECT
            c.person_id,
            CASE WHEN COUNT(dd.vax_date) > 0 THEN 1 ELSE 0 END AS f_ext_vax_any,
            COUNT(dd.vax_date) AS f_ext_vax_dose_count,
            CASE WHEN COUNT(dd.vax_date) >= 3 THEN 1 ELSE 0 END AS f_ext_vax_boosted,
            MAX(COALESCE(dd.has_mrna, 0)) AS f_ext_vax_mrna_any,
            MAX(COALESCE(dd.has_vector, 0)) AS f_ext_vax_vector_any
        FROM cohort c
        LEFT JOIN dose_days dd ON dd.person_id = c.person_id
        GROUP BY c.person_id
    )

    SELECT * FROM agg
    """

    df = _fetch_df(cur, sql, "Querying vaccination features")
    df = df.drop_duplicates(subset=["person_id"])
    print(f"  Extracted ext vaccination for {len(df):,} patients")
    return df


# ============================================================================
# Extended Feature: Additional Acute Symptoms (NOT in Antony Family B)
# ============================================================================

def extract_ext_additional_symptoms(cur):
    """
    Extract acute-phase symptoms beyond Antony Family B.
    Includes: palpitations, neurocognitive issues, insomnia, tachycardia.
    Window: acute_start to acute_end (same as Antony symptoms).
    """
    print("\n[Ext] Extracting additional acute symptoms...")

    sql = f"""
    WITH cohort AS (
        SELECT person_id, acute_start, acute_end
        FROM "#antony_cohort"
    ),
    acute_conditions AS (
        SELECT DISTINCT c.person_id, co.condition_concept_id
        FROM cohort c
        JOIN {CDM_SCHEMA}.condition_occurrence co
          ON co.person_id = c.person_id
         AND co.condition_start_date BETWEEN c.acute_start AND c.acute_end
    ),
    symptom_ancestors AS (
        SELECT ac.person_id, ca.ancestor_concept_id
        FROM acute_conditions ac
        JOIN {CDM_SCHEMA}.concept_ancestor ca
          ON ca.descendant_concept_id = ac.condition_concept_id
    )
    SELECT
        c.person_id,
        MAX(CASE WHEN sa.ancestor_concept_id IN (315078, 77670)          THEN 1 ELSE 0 END) AS f_ext_sym_palpitations,
        MAX(CASE WHEN sa.ancestor_concept_id IN (46271045, 4107230, 443432) THEN 1 ELSE 0 END) AS f_ext_sym_neurocog,
        MAX(CASE WHEN sa.ancestor_concept_id IN (436962, 435524)         THEN 1 ELSE 0 END) AS f_ext_sym_insomnia,
        MAX(CASE WHEN sa.ancestor_concept_id IN (444070)                 THEN 1 ELSE 0 END) AS f_ext_sym_tachycardia
    FROM cohort c
    LEFT JOIN symptom_ancestors sa ON sa.person_id = c.person_id
    GROUP BY c.person_id
    """

    df = _fetch_df(cur, sql, "Querying additional symptoms")
    print(f"  Extracted ext additional symptoms for {len(df):,} patients")
    return df


# ============================================================================
# Extended baseline: chronic viral hepatitis (pre-COVID comorbidities)
# ============================================================================

# OMOP ancestor concept IDs for Hep B/C — queried via concept_ancestor for broad
# matching (covers all descendant condition concepts).
_CHRONIC_INFECTION_ANCESTORS = {
    "f_ext_hepatitis_b": 4281232,   # Viral hepatitis type B
    "f_ext_hepatitis_c": 197494,    # Hepatitis C
}


def extract_ext_chronic_infections(cur):
    """
    Extract pre-COVID chronic viral hepatitis B/C as baseline comorbidities.

    Window: on or before covid_index_date (same as Antony comorbidity logic).
    Uses concept_ancestor hierarchy for broad matching.

    Returns:
        DataFrame with person_id + binary columns (f_ext_hepatitis_b, f_ext_hepatitis_c)
    """
    print("\n[Ext] Extracting chronic viral hepatitis (Hep B/C)...")

    case_exprs = []
    for col_name, ancestor_id in _CHRONIC_INFECTION_ANCESTORS.items():
        case_exprs.append(
            f"MAX(CASE WHEN ca.ancestor_concept_id = {ancestor_id} THEN 1 ELSE 0 END) AS {col_name}"
        )
    case_sql = ",\n        ".join(case_exprs)

    sql = f"""
    WITH cohort AS (
        SELECT person_id, covid_index_date
        FROM "#antony_cohort"
    ),
    baseline_conditions AS (
        SELECT DISTINCT
            c.person_id,
            co.condition_concept_id
        FROM cohort c
        JOIN {CDM_SCHEMA}.condition_occurrence co
          ON co.person_id = c.person_id
         AND co.condition_start_date <= c.covid_index_date
    ),
    cond_ancestors AS (
        SELECT
            bc.person_id,
            ca.ancestor_concept_id
        FROM baseline_conditions bc
        JOIN {CDM_SCHEMA}.concept_ancestor ca
          ON ca.descendant_concept_id = bc.condition_concept_id
    )
    SELECT
        c.person_id,
        {case_sql}
    FROM cohort c
    LEFT JOIN cond_ancestors ca
      ON ca.person_id = c.person_id
    GROUP BY c.person_id
    """

    df = _fetch_df(cur, sql, "Querying chronic Hep B/C comorbidities")
    print(f"  Extracted chronic infection features for {len(df):,} patients")
    return df


# ============================================================================
# Extended Feature: SARS-CoV-2 Serology (Baseline_ext)
# ============================================================================

def extract_ext_serology(cur, include_tested_flag=False):
    """
    Extract 6 SARS-CoV-2 serology baseline features from OMOP measurement.

    Extracts for all cohort patients from #antony_cohort, using post-index
    records to characterize baseline serological status.

    Measures:
      - f_ext_sars_cov2_ab_ever_positive: Any positive interpretation
      - f_ext_sars_cov2_ab_ever_negative: Any negative interpretation
      - f_ext_sars_cov2_ab_wave_at_first_positive: Wave category at first positive date
      - f_ext_sars_cov2_ab_titer_log_max: Max log-titer harmonized per assay
      - f_ext_sars_cov2_ab_titer_log_at_index: Log-titer closest to COVID index date
      - f_ext_sars_cov2_ab_n_tests: Count of serology measurements

    Gate: value_as_concept_id IS NOT NULL OR value_as_number IS NOT NULL

    Args:
        cur: HANA cursor with `"#antony_cohort"` populated.
        include_tested_flag: If True, emit an additional binary column
            `f_ext_sars_cov2_ab_tested` = 1 iff any baseline serology row was
            observed for the patient. This separates the *decision-to-measure*
            node from the *measured-value* nodes in the DAG (detection-bias
            control); RF can split on the testing-decision axis without
            leaning on the test-count covariate.

    Returns:
        DataFrame with person_id + 6 feature columns (+ `f_ext_sars_cov2_ab_tested`
        when `include_tested_flag=True`).
    """
    print("\n[Ext] Extracting SARS-CoV-2 serology features...")

    loinc_codes = ",".join(f"'{lc}'" for lc in SARS_COV2_AB_LOINCS)
    pos_ids = ",".join(str(x) for x in SARS_COV2_AB_POSITIVE_CONCEPT_IDS)
    neg_ids = ",".join(str(x) for x in SARS_COV2_AB_NEGATIVE_CONCEPT_IDS)

    # Query interpretation (serostatus) and basic aggregations.
    # Time-window guard: serology rows may come from any time up to the end of
    # the acute window (acute_end). This allows prior-history titers to be used
    # as baseline information but prevents leakage from follow-up after the
    # acute phase (e.g., post-PASC antibody draws).
    sql_interp = f"""
    WITH cohort AS (
        SELECT person_id, covid_index_date, acute_end
        FROM "#antony_cohort"
    ),

    serology_raw AS (
        SELECT
            c.person_id,
            m.measurement_date,
            m.measurement_source_concept_id,
            m.value_as_concept_id,
            CAST(m.value_as_number AS DOUBLE) AS value_num,
            ABS(DAYS_BETWEEN(m.measurement_date, c.covid_index_date)) AS days_from_index
        FROM cohort c
        JOIN {CDM_SCHEMA}.measurement m
          ON m.person_id = c.person_id
         AND (m.value_as_concept_id IS NOT NULL OR m.value_as_number IS NOT NULL)
         AND m.measurement_date <= c.acute_end
        JOIN {CDM_SCHEMA}.concept cm
          ON cm.concept_id = m.measurement_concept_id
         AND UPPER(cm.concept_code) IN ({loinc_codes})
    ),

    interpretation AS (
        SELECT
            person_id,
            CASE
                WHEN value_as_concept_id IN ({pos_ids}) THEN 1
                WHEN value_as_concept_id IN ({neg_ids}) THEN 0
                ELSE NULL
            END AS interp_code
        FROM serology_raw
    ),

    first_positive_date AS (
        SELECT
            person_id,
            MIN(measurement_date) AS first_pos_date
        FROM serology_raw s
        WHERE EXISTS (
            SELECT 1 FROM interpretation i
            WHERE i.person_id = s.person_id AND i.interp_code = 1
        )
        GROUP BY person_id
    ),

    wave_assignment AS (
        SELECT
            fp.person_id,
            CASE
                WHEN fp.first_pos_date >= '2020-02-09' AND fp.first_pos_date <= '2020-08-30' THEN 1
                WHEN fp.first_pos_date >= '2020-08-31' AND fp.first_pos_date <= '2021-06-20' THEN 2
                WHEN fp.first_pos_date >= '2021-06-21' AND fp.first_pos_date <= '2021-10-31' THEN 3
                WHEN fp.first_pos_date >= '2021-11-01' AND fp.first_pos_date <= '2022-03-06' THEN 4
                WHEN fp.first_pos_date >= '2022-03-07' AND fp.first_pos_date <= '2022-07-18' THEN 5
                ELSE 6
            END AS wave_category
        FROM first_positive_date fp
    ),

    interp_agg AS (
        SELECT
            person_id,
            MAX(CASE WHEN interp_code = 1 THEN 1 ELSE 0 END) AS has_positive,
            MAX(CASE WHEN interp_code = 0 THEN 1 ELSE 0 END) AS has_negative,
            COUNT(*) AS n_tests
        FROM interpretation
        WHERE interp_code IS NOT NULL
        GROUP BY person_id
    ),

    agg AS (
        SELECT
            c.person_id,
            COALESCE(ia.has_positive, 0) AS f_ext_sars_cov2_ab_ever_positive,
            COALESCE(ia.has_negative, 0) AS f_ext_sars_cov2_ab_ever_negative,
            COALESCE(wa.wave_category, 0) AS f_ext_sars_cov2_ab_wave_at_first_positive,
            COALESCE(ia.n_tests, 0) AS f_ext_sars_cov2_ab_n_tests
        FROM cohort c
        LEFT JOIN interp_agg ia ON ia.person_id = c.person_id
        LEFT JOIN wave_assignment wa ON wa.person_id = c.person_id
    )

    SELECT * FROM agg
    """

    df_interp = _fetch_df(cur, sql_interp, "Querying SARS-CoV-2 serology interpretation")
    df_interp.columns = [c.lower() for c in df_interp.columns]

    # Query titers separately for per-assay z-score computation in Python.
    # Same time-window guard as sql_interp: cap at acute_end to avoid leaking
    # post-acute follow-up serology into a baseline predictor.
    sql_titers = f"""
    WITH cohort AS (
        SELECT person_id, covid_index_date, acute_end
        FROM "#antony_cohort"
    ),

    serology_raw AS (
        SELECT
            c.person_id,
            m.measurement_date,
            m.measurement_source_concept_id,
            CAST(m.value_as_number AS DOUBLE) AS value_num,
            c.covid_index_date
        FROM cohort c
        JOIN {CDM_SCHEMA}.measurement m
          ON m.person_id = c.person_id
         AND m.value_as_number IS NOT NULL
         AND m.measurement_date <= c.acute_end
        JOIN {CDM_SCHEMA}.concept cm
          ON cm.concept_id = m.measurement_concept_id
         AND UPPER(cm.concept_code) IN ({loinc_codes})
    )

    SELECT
        person_id,
        measurement_date,
        measurement_source_concept_id,
        value_num,
        covid_index_date
    FROM serology_raw
    WHERE value_num > 0
    """

    df_titers = _fetch_df(cur, sql_titers, None)
    if not df_titers.empty:
        df_titers.columns = [c.lower() for c in df_titers.columns]
        df_titers["measurement_date"] = pd.to_datetime(df_titers["measurement_date"])
        df_titers["covid_index_date"] = pd.to_datetime(df_titers["covid_index_date"])
        df_titers["days_from_index"] = (
            df_titers["measurement_date"] - df_titers["covid_index_date"]
        ).dt.days.abs()
        df_titers["log_value"] = np.log(np.maximum(df_titers["value_num"].astype(float), 1e-10))

    # Compute per-assay z-scores in Python
    titer_max_by_person = {}
    titer_closest_by_person = {}
    titer_slope_by_person = {}
    titer_n_meas_by_person = {}

    if not df_titers.empty:
        # Signed days (negative = before index) for slope regression
        df_titers["signed_days"] = (
            df_titers["measurement_date"] - df_titers["covid_index_date"]
        ).dt.days.astype(float)

        for person_id, group in df_titers.groupby('person_id'):
            z_scores_all = []
            closest_z = 0.0

            for assay_id, assay_data in group.groupby('measurement_source_concept_id'):
                log_vals = assay_data['log_value'].values
                mean_log = np.mean(log_vals)
                std_log = np.std(log_vals, ddof=0)

                if std_log > 0:
                    z = (log_vals - mean_log) / std_log
                else:
                    z = np.zeros_like(log_vals)

                z_scores_all.extend(z)

                # Find closest to index
                idx_closest = assay_data['days_from_index'].idxmin()
                if idx_closest in assay_data.index:
                    val_closest = assay_data.loc[idx_closest, 'log_value']
                    closest_z = (val_closest - mean_log) / std_log if std_log > 0 else 0

            titer_max_by_person[person_id] = float(np.max(z_scores_all)) if z_scores_all else 0.0
            titer_closest_by_person[person_id] = closest_z

            # Titer count and slope (regression of log_value on signed_days)
            titer_n_meas_by_person[person_id] = int(len(group))
            if len(group) >= 2 and group["signed_days"].nunique() >= 2:
                try:
                    slope = float(np.polyfit(
                        group["signed_days"].values,
                        group["log_value"].values,
                        1,
                    )[0])
                except (np.linalg.LinAlgError, ValueError):
                    slope = 0.0
                titer_slope_by_person[person_id] = slope
            else:
                titer_slope_by_person[person_id] = 0.0

    # Add titer features to interpretation dataframe
    df_interp['f_ext_sars_cov2_ab_titer_log_max'] = df_interp['person_id'].map(
        lambda x: titer_max_by_person.get(x, 0.0)
    ).fillna(0.0)

    df_interp['f_ext_sars_cov2_ab_titer_log_at_index'] = df_interp['person_id'].map(
        lambda x: titer_closest_by_person.get(x, 0.0)
    ).fillna(0.0)

    df_interp['f_ext_sars_cov2_ab_titer_log_slope'] = df_interp['person_id'].map(
        lambda x: titer_slope_by_person.get(x, 0.0)
    ).fillna(0.0)

    df_interp['f_ext_sars_cov2_ab_n_titer_measurements'] = df_interp['person_id'].map(
        lambda x: titer_n_meas_by_person.get(x, 0)
    ).fillna(0).astype(int)

    # ---- Anti-N vs anti-S quantitative contrast ----
    # Per-assay z-score, averaged within N-LOINC group and S-LOINC group per
    # patient, then contrasted. If either LOINC group has no data for a patient
    # the corresponding feature is 0 (with the composite contrast treated
    # as missing-equivalent, i.e. 0).
    n_loincs = ",".join(f"'{lc}'" for lc in SARS_COV2_N_AB_LOINCS)
    s_loincs = ",".join(f"'{lc}'" for lc in SARS_COV2_S_AB_LOINCS)
    sql_ns_titers = f"""
    WITH cohort AS (
        SELECT person_id, covid_index_date, acute_end
        FROM "#antony_cohort"
    )
    SELECT
        c.person_id,
        m.measurement_source_concept_id AS assay_id,
        CAST(m.value_as_number AS DOUBLE) AS value_num,
        CASE WHEN UPPER(cm.concept_code) IN ({n_loincs}) THEN 'N'
             WHEN UPPER(cm.concept_code) IN ({s_loincs}) THEN 'S'
             ELSE NULL END AS ab_type
    FROM cohort c
    JOIN {CDM_SCHEMA}.measurement m
      ON m.person_id = c.person_id
     AND m.value_as_number IS NOT NULL
     AND m.measurement_date <= c.acute_end
    JOIN {CDM_SCHEMA}.concept cm
      ON cm.concept_id = m.measurement_concept_id
     AND UPPER(cm.concept_code) IN ({n_loincs}, {s_loincs})
    WHERE m.value_as_number > 0
    """
    df_ns = _fetch_df(cur, sql_ns_titers, "Querying anti-N / anti-S titers")
    if not df_ns.empty:
        df_ns.columns = [c.lower() for c in df_ns.columns]
        df_ns["log_value"] = np.log(np.maximum(df_ns["value_num"].astype(float), 1e-10))

        # Per-assay z across all patients (within-assay normalization)
        def _z_within_assay(g):
            mu = g["log_value"].mean()
            sd = g["log_value"].std(ddof=0)
            g = g.copy()
            g["z"] = (g["log_value"] - mu) / sd if sd and sd > 0 else 0.0
            return g

        df_ns = df_ns.groupby("assay_id", group_keys=False).apply(_z_within_assay)

        n_z = (df_ns[df_ns["ab_type"] == "N"]
               .groupby("person_id")["z"].mean()
               .rename("f_ext_sars_cov2_anti_n_igg_z"))
        s_z = (df_ns[df_ns["ab_type"] == "S"]
               .groupby("person_id")["z"].mean()
               .rename("f_ext_sars_cov2_anti_s_igg_z"))
        # Capture data-availability and N/S phenotype sets BEFORE the merge so
        # that "z==0.0 because no data" is distinguished from "z==0.0 because
        # the patient's value is at the per-assay mean."
        has_n_ids = set(n_z.index)
        has_s_ids = set(s_z.index)
        n_pos_ids = set(n_z.index[n_z > 0])  # N-titer above per-assay mean => infection-exposed
        s_pos_ids = set(s_z.index[s_z > 0])  # S-titer above per-assay mean
        df_interp = df_interp.merge(n_z, on="person_id", how="left")
        df_interp = df_interp.merge(s_z, on="person_id", how="left")
    else:
        df_interp["f_ext_sars_cov2_anti_n_igg_z"] = 0.0
        df_interp["f_ext_sars_cov2_anti_s_igg_z"] = 0.0
        has_n_ids = set()
        has_s_ids = set()
        n_pos_ids = set()
        s_pos_ids = set()

    df_interp["f_ext_sars_cov2_anti_n_igg_z"] = df_interp["f_ext_sars_cov2_anti_n_igg_z"].fillna(0.0)
    df_interp["f_ext_sars_cov2_anti_s_igg_z"] = df_interp["f_ext_sars_cov2_anti_s_igg_z"].fillna(0.0)
    df_interp["f_ext_sars_cov2_anti_n_minus_s_z"] = (
        df_interp["f_ext_sars_cov2_anti_n_igg_z"]
        - df_interp["f_ext_sars_cov2_anti_s_igg_z"]
    )

    # ---- Task 1: N-vs-S compositional immunoprofile flags ----
    # Motivated by Carreño et al. Nat Commun 2024: anti-NP vs anti-spike
    # divergence (vaccine-spike-only induction + faster NP waning) makes the
    # NP/spike composition informative above either marker alone.
    pid = df_interp["person_id"]
    df_interp["f_ext_sars_cov2_has_n_titer"] = pid.isin(has_n_ids).astype("int8")
    df_interp["f_ext_sars_cov2_has_s_titer"] = pid.isin(has_s_ids).astype("int8")

    ab_ever_pos = df_interp.get("f_ext_sars_cov2_ab_ever_positive",
                                pd.Series(0, index=df_interp.index)).fillna(0).astype(int)

    # Vax-only proxy: S-titer present AND no N-titer AND any positive antibody.
    # NB this is a proxy — absence of N-titer may mean truly NP-negative OR
    # that NP was simply never assayed. Ambiguity is unavoidable in EHR data.
    df_interp["f_ext_sars_cov2_immunoprofile_vax_only"] = (
        pid.isin(has_s_ids) & ~pid.isin(has_n_ids) & (ab_ever_pos == 1)
    ).astype("int8")

    # Infection-exposed: NP titer above per-assay mean (z > 0).
    df_interp["f_ext_sars_cov2_immunoprofile_infection"] = pid.isin(n_pos_ids).astype("int8")

    # Breakthrough: both N-titer and S-titer above per-assay mean.
    df_interp["f_ext_sars_cov2_immunoprofile_breakthrough"] = (
        pid.isin(n_pos_ids) & pid.isin(s_pos_ids)
    ).astype("int8")

    # ---- Task 2: wave-stratified antibody-positivity encoding ----
    # Wave bins reflect the three seroprevalence regimes in Carreño et al.:
    # (1–2) pre-vaccine low prevalence, (3–4) vaccine rollout + Delta,
    # (5–6) Omicron-era saturation. Augments (does not replace) ab_ever_positive.
    wave = df_interp.get("f_ext_sars_cov2_ab_wave_at_first_positive",
                         pd.Series(0, index=df_interp.index)).fillna(0).astype(int)
    is_pos = (ab_ever_pos == 1)
    df_interp["f_ext_sars_cov2_ab_early_wave_positive"] = (is_pos & wave.isin([1, 2])).astype("int8")
    df_interp["f_ext_sars_cov2_ab_mid_wave_positive"]   = (is_pos & wave.isin([3, 4])).astype("int8")
    df_interp["f_ext_sars_cov2_ab_late_wave_positive"]  = (is_pos & wave.isin([5, 6])).astype("int8")

    df = df_interp

    # Type hygiene
    for col in ["f_ext_sars_cov2_ab_ever_positive", "f_ext_sars_cov2_ab_ever_negative"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    for col in ["f_ext_sars_cov2_ab_wave_at_first_positive", "f_ext_sars_cov2_ab_n_tests",
                "f_ext_sars_cov2_ab_n_titer_measurements"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    for col in ["f_ext_sars_cov2_ab_titer_log_max", "f_ext_sars_cov2_ab_titer_log_at_index",
                "f_ext_sars_cov2_ab_titer_log_slope",
                "f_ext_sars_cov2_anti_n_igg_z", "f_ext_sars_cov2_anti_s_igg_z",
                "f_ext_sars_cov2_anti_n_minus_s_z"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Task 4: prevalence sanity check for the N-vs-S compositional +
    # wave-stratified features added per Carreño et al. 2024.
    _new_serology_cols = [
        "f_ext_sars_cov2_has_n_titer",
        "f_ext_sars_cov2_has_s_titer",
        "f_ext_sars_cov2_immunoprofile_vax_only",
        "f_ext_sars_cov2_immunoprofile_infection",
        "f_ext_sars_cov2_immunoprofile_breakthrough",
        "f_ext_sars_cov2_ab_early_wave_positive",
        "f_ext_sars_cov2_ab_mid_wave_positive",
        "f_ext_sars_cov2_ab_late_wave_positive",
    ]
    if len(df) > 0:
        print("  New serology feature prevalence (Carreño-motivated):")
        for c in _new_serology_cols:
            if c in df.columns:
                n_pos = int(df[c].sum())
                print(f"    {c}: {n_pos:,} / {len(df):,} ({100*n_pos/len(df):.2f}%)")

    # Optional: detection-bias indicator. Materialized as a separate binary
    # (1 iff any serology row exists) so the model can split on the
    # decision-to-measure axis independently of `f_ext_sars_cov2_ab_n_tests`.
    if include_tested_flag:
        if "f_ext_sars_cov2_ab_n_tests" in df.columns:
            df["f_ext_sars_cov2_ab_tested"] = (
                df["f_ext_sars_cov2_ab_n_tests"].fillna(0).astype(int) > 0
            ).astype("int8")
        else:
            df["f_ext_sars_cov2_ab_tested"] = 0
        n_tested = int(df["f_ext_sars_cov2_ab_tested"].sum())
        print(f"  Detection-bias indicator (f_ext_sars_cov2_ab_tested): "
              f"{n_tested:,} / {len(df):,} ({100*n_tested/max(len(df),1):.2f}%)")

    print(f"  Extracted serology features for {len(df):,} patients")
    return df



def extract_mechanistic_signals(cur, cohort_df):
    """
    Extract all three mechanistic signal families using the acute phase window.

    Maps acute_start/acute_end -> postcovid_window_start/postcovid_window_end
    for compatibility with mech_signals_common.

    Returns:
        tuple of (viral_df, immuno_df, endo_df, viral_cols, immuno_cols, endo_cols)
    """
    print("\n" + "=" * 80)
    print("MECHANISTIC SIGNAL EXTRACTION (acute phase window)")
    print("=" * 80)

    # Build the window DataFrame for mech_signals_common
    window_df = cohort_df[["person_id"]].copy()
    window_df["postcovid_window_start"] = pd.to_datetime(cohort_df["acute_start"])
    window_df["postcovid_window_end"] = pd.to_datetime(cohort_df["acute_end"])
    window_df = window_df.drop_duplicates(subset=["person_id"])

    # Upload to #cohort_temp
    prepare_temp_cohort(cur, window_df)

    # Viral persistence
    print("\n[MechSig] Extracting viral persistence signals...")
    viral_df = build_signals(cur, window_df, VIRAL_INDICATOR_LIST, mechanism_prefix="viral")
    viral_cols = [c for c in viral_df.columns if c.startswith("f_ind_viral_")]
    print(f"  Viral features: {len(viral_cols)}")

    # Immunoinflammatory
    print("\n[MechSig] Extracting immunoinflammatory signals...")
    immuno_df = build_signals(cur, window_df, IMMUNO_INDICATOR_LIST, mechanism_prefix="immuno")
    immuno_cols = [c for c in immuno_df.columns if c.startswith("f_ind_immuno_")]
    print(f"  Immuno features: {len(immuno_cols)}")

    # Endothelial dysfunction
    print("\n[MechSig] Extracting endothelial dysfunction signals...")
    endo_df = build_signals(cur, window_df, ENDO_INDICATOR_LIST, mechanism_prefix="endo")
    endo_cols = [c for c in endo_df.columns if c.startswith("f_ind_endo_")]
    print(f"  Endo features: {len(endo_cols)}")

    return viral_df, immuno_df, endo_df, viral_cols, immuno_cols, endo_cols


# ============================================================================
# Type hygiene
# ============================================================================

def _apply_type_hygiene(df, ext_cols, flag_cols):
    """Clean up types: flags to int, numeric with missing indicators."""
    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Numeric columns: coerce types (missing indicators disabled)
    numeric_ext = ["f_ext_bmi_last", "f_ext_sbp_last", "f_ext_dbp_last", "f_ext_spo2_min", "f_ext_los_max"]
    for col in numeric_ext:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # COMMENTED OUT: missingness indicators (*_missing) are healthcare-
            # engagement proxies, not mechanism features.  f_ext_bmi_last_missing
            # (SHAP 0.044) predicts U09.9 via "patient had MSHS contact", not
            # biology.  Healthcare engagement is now captured by dedicated
            # f_eng_* features (Step 5) and controlled for explicitly.
            # See Al-Aly 2021, Crosskey 2025 on ascertainment confounding.
            # miss_col = f"{col}_missing"
            # if miss_col not in df.columns:
            #     df[miss_col] = df[col].isna().astype(int)

    return df


# ============================================================================
# Step 5: Healthcare-engagement control features (ascertainment adjustment)
# ============================================================================

from omop_config import CDM_SCHEMA as CDM  # OMOP schema name; set OMOP_CDM_SCHEMA to override

# Primary-care provider specialty concepts (OMOP)
_PCP_SPECIALTY_CONCEPTS = (
    38004446,   # General Practice
    38004489,   # Internal Medicine
    38004479,   # Family Medicine
)

def extract_engagement_features(cur, cohort_df):
    """
    Extract healthcare-engagement control features (Step 5).

    These features soak up the ascertainment pathway so that mechanism
    features are not rewarded for encoding 'patient had MSHS contact'.
    Designed to be forced into all model configs (not Boruta-droppable).

    Features:
        f_eng_n_encounters_pre_index_1y  — MSHS visits in year before index
        f_eng_has_pcp_pre_index          — any PCP visit in pre-index year
        f_eng_insurance_commercial       — commercial insurance (one-hot)
        f_eng_insurance_medicaid         — Medicaid (one-hot)
        f_eng_insurance_medicare         — Medicare (one-hot)
        f_eng_insurance_self_pay         — self-pay / uninsured (one-hot)
        (REMOVED: f_eng_n_encounters_post_acute_1y — leaked by proxying PASC label)

    Args:
        cur: Active HANA cursor (with `"#antony_cohort"` temp table populated).
        cohort_df: DataFrame with at least `person_id` column.

    Returns:
        (engagement_df, engagement_cols)
    """
    print("\n  Extracting healthcare-engagement control features...")

    pids = cohort_df[["person_id"]].drop_duplicates()

    # --- f_eng_n_encounters_pre_index_1y ---
    sql_pre = f"""
    SELECT
        ac.person_id,
        COUNT(DISTINCT vo.visit_occurrence_id) AS f_eng_n_encounters_pre_index_1y
    FROM "#antony_cohort" ac
    JOIN {CDM}.visit_occurrence vo
      ON vo.person_id = ac.person_id
     AND vo.visit_start_date BETWEEN ADD_DAYS(ac.covid_index_date, -365) AND ADD_DAYS(ac.covid_index_date, -1)
    GROUP BY ac.person_id
    """
    print("    Pre-index encounters (1y)...")
    cur.execute(sql_pre)
    pre_enc = pd.DataFrame(cur.fetchall(), columns=[d[0].lower() for d in cur.description])

    # --- f_eng_has_pcp_pre_index ---
    # Join visit_occurrence → provider → care_site (or use provider specialty)
    sql_pcp = f"""
    SELECT DISTINCT ac.person_id, 1 AS f_eng_has_pcp_pre_index
    FROM "#antony_cohort" ac
    JOIN {CDM}.visit_occurrence vo
      ON vo.person_id = ac.person_id
     AND vo.visit_start_date BETWEEN ADD_DAYS(ac.covid_index_date, -365) AND ADD_DAYS(ac.covid_index_date, -1)
    JOIN {CDM}.provider pr
      ON pr.provider_id = vo.provider_id
    WHERE pr.specialty_concept_id IN ({', '.join(str(c) for c in _PCP_SPECIALTY_CONCEPTS)})
    """
    print("    PCP visits pre-index...")
    cur.execute(sql_pcp)
    pcp_df = pd.DataFrame(cur.fetchall(), columns=[d[0].lower() for d in cur.description])

    # --- Insurance type (from payer_plan_period) ---
    # Use name matching on plan_name since concept mapping varies by site
    sql_ins = f"""
    SELECT
        ac.person_id,
        pp.payer_source_value
    FROM "#antony_cohort" ac
    JOIN {CDM}.payer_plan_period pp
      ON pp.person_id = ac.person_id
     AND pp.payer_plan_period_start_date <= ac.covid_index_date
     AND (pp.payer_plan_period_end_date >= ac.covid_index_date
          OR pp.payer_plan_period_end_date IS NULL)
    """
    print("    Insurance type...")
    try:
        cur.execute(sql_ins)
        ins_raw = pd.DataFrame(cur.fetchall(), columns=[d[0].lower() for d in cur.description])
    except Exception as e:
        print(f"    WARNING: payer_plan_period query failed ({e}); insurance features will be 0")
        ins_raw = pd.DataFrame(columns=["person_id", "payer_source_value"])

    # Classify insurance by name matching
    ins_df = pids.copy()
    ins_df["f_eng_insurance_commercial"] = 0
    ins_df["f_eng_insurance_medicaid"] = 0
    ins_df["f_eng_insurance_medicare"] = 0
    ins_df["f_eng_insurance_self_pay"] = 0

    if not ins_raw.empty:
        ins_raw["payer_source_value"] = ins_raw["payer_source_value"].fillna("").str.upper()
        for pid, group in ins_raw.groupby("person_id"):
            vals = " ".join(group["payer_source_value"].tolist())
            mask = ins_df["person_id"] == pid
            if "MEDICAID" in vals:
                ins_df.loc[mask, "f_eng_insurance_medicaid"] = 1
            if "MEDICARE" in vals:
                ins_df.loc[mask, "f_eng_insurance_medicare"] = 1
            if any(k in vals for k in ("COMMERCIAL", "PPO", "HMO", "AETNA", "CIGNA",
                                        "UNITED", "BLUE", "ANTHEM", "EMPIRE")):
                ins_df.loc[mask, "f_eng_insurance_commercial"] = 1
            if any(k in vals for k in ("SELF", "UNINSURED", "CHARITY", "NO INSURANCE")):
                ins_df.loc[mask, "f_eng_insurance_self_pay"] = 1

    # NOTE: f_eng_n_encounters_post_acute_1y (day 60-425) REMOVED — it directly
    # proxies the PASC label (patients diagnosed with U09.9 have more visits in
    # exactly that window, creating textbook ascertainment leakage).

    # NOTE (2026-04 ablation): Acute-window utilization counts REMOVED
    # (f_eng_n_labs_acute, f_eng_n_distinct_meas_concepts_acute,
    #  f_eng_n_encounters_acute, f_eng_had_inpatient_acute).
    # Rationale: these were added as an "ascertainment sponge", but when
    # forced into the feature matrix they acted as near-sufficient statistics
    # for any mechanistic test/indicator (a patient with 50+ labs in the acute
    # window has trivially had CRP / D-dimer / NLR / etc. measured). SHAP on
    # baseline_ext at w0_21 showed f_eng_n_encounters_acute, f_eng_n_labs_acute,
    # and f_eng_n_distinct_meas_concepts_acute as the top-ranked features,
    # inflating baseline_ext AUROC from ~0.71 (baseline_antony) to ~0.82 and
    # leaving zero marginal lift for any mechsig_* config. They are dropped
    # entirely rather than merely un-forced because the prevalence filter
    # would not remove them (continuous counts) and Boruta keeps them since
    # they are strongly label-correlated via ascertainment, not biology.

    # --- Assemble ---
    result = pids.copy()
    result = result.merge(pre_enc, on="person_id", how="left")
    result = result.merge(pcp_df, on="person_id", how="left")
    result = result.merge(ins_df, on="person_id", how="left")

    engagement_cols = [
        "f_eng_n_encounters_pre_index_1y",
        "f_eng_has_pcp_pre_index",
        "f_eng_insurance_commercial",
        "f_eng_insurance_medicaid",
        "f_eng_insurance_medicare",
        "f_eng_insurance_self_pay",
    ]

    for col in engagement_cols:
        if col not in result.columns:
            result[col] = 0
        result[col] = result[col].fillna(0).astype(int)

    print(f"    Engagement features: {len(engagement_cols)} columns for {len(result):,} patients")
    for col in engagement_cols:
        n_nonzero = (result[col] > 0).sum()
        print(f"      {col:45s}: {n_nonzero:6,} non-zero ({100*n_nonzero/len(result):.1f}%)")

    return result[["person_id"] + engagement_cols], engagement_cols


# ============================================================================
# Orchestrator
# ============================================================================

def build_combined_feature_matrix(
    cur,
    cohort_df,
    include_treatment=True,
    include_extended=True,
    include_mechanistic=True,
    include_engagement=True,
    use_hpo_symptoms=False,
    include_tested_flag=False,
):
    """
    Build the complete feature matrix for one cohort.

    Expects '#antony_cohort' temp table to already be populated
    (done by combined_cohort.build_combined_cohorts).

    Args:
        cur: Database cursor
        cohort_df: DataFrame with person_id, label, has_documented_covid, etc.
        include_treatment: Include Family E treatment measures
        include_extended: Include extended baseline features
        include_mechanistic: Include mechanistic signal features
        include_engagement: Include healthcare-engagement control features (Step 5)
        use_hpo_symptoms: Map acute symptoms through OMOP2OBO HPO terms (thesis 3.4.1)
        include_tested_flag: Emit f_ext_sars_cov2_ab_tested (not used in the thesis)

    Returns:
        tuple of (feature_df, feature_families)
        where feature_families maps family name -> list of column names
    """
    print("\n" + "=" * 80)
    print("BUILDING COMBINED FEATURE MATRIX")
    print("=" * 80)

    result = cohort_df[["person_id", "label", "covid_index_date"]].copy()

    # ---- Antony Families A–E ----
    comorbidities = extract_comorbidities(cur)
    if use_hpo_symptoms:
        from omop2obo_mapping import (
            load_omop2obo_condition_mappings, build_concept_to_hpo_lookup,
            TRUSTED_CATEGORIES,
        )
        mapping_df = load_omop2obo_condition_mappings(quality_filter=TRUSTED_CATEGORIES)
        concept_to_hpo = build_concept_to_hpo_lookup(mapping_df)
        symptoms = extract_acute_symptoms_hpo(cur, concept_to_hpo)
    else:
        symptoms = extract_acute_symptoms(cur)
    drugs = extract_acute_drugs(cur)
    demographics = extract_demographics(cur)

    result = result.merge(comorbidities, on="person_id", how="left")
    result = result.merge(symptoms, on="person_id", how="left")
    result = result.merge(drugs, on="person_id", how="left")
    result = result.merge(demographics, on="person_id", how="left")

    comor_cols = [c for c in comorbidities.columns if c.startswith("f_comor_")]
    sym_cols = [c for c in symptoms.columns if c.startswith("f_sym_")]
    drug_cols = [c for c in drugs.columns if c.startswith("f_drug_")]
    demo_cols = [c for c in demographics.columns if c.startswith("f_")]
    treatment_cols = []

    if include_treatment:
        treatment = extract_treatment_measures(cur)
        result = result.merge(treatment, on="person_id", how="left")
        treatment_cols = [c for c in treatment.columns if c.startswith("f_tx_")]
        result[treatment_cols] = result[treatment_cols].fillna(0)

    # Fill NaNs for binary families
    for cols in [comor_cols, sym_cols, drug_cols]:
        result[cols] = result[cols].fillna(0).astype(int)

    # ---- Extended Features ----
    ext_bmi_cols = []
    ext_severity_cols = []
    ext_wave_cols = []
    ext_vax_cols = []
    ext_sym_cols = []
    ext_covid_flag_cols = []
    ext_chronic_inf_cols = []
    ext_serology_cols = []

    if include_extended:
        # BMI / BP / Lifestyle
        bmi_bp = extract_ext_bmi_bp_lifestyle(cur)
        result = result.merge(bmi_bp, on="person_id", how="left")
        ext_bmi_cols = [c for c in bmi_bp.columns if c.startswith("f_ext_")]

        # Acute severity
        severity = extract_ext_acute_severity(cur)
        result = result.merge(severity, on="person_id", how="left")
        ext_severity_cols = [c for c in severity.columns if c.startswith("f_ext_")]

        # Variant wave (pandas-only)
        result, ext_wave_cols = add_ext_variant_wave(result)

        # Vaccination
        vax = extract_ext_vaccination(cur)
        result = result.merge(vax, on="person_id", how="left")
        ext_vax_cols = [c for c in vax.columns if c.startswith("f_ext_")]

        # Additional acute symptoms
        add_sym = extract_ext_additional_symptoms(cur)
        result = result.merge(add_sym, on="person_id", how="left")
        ext_sym_cols = [c for c in add_sym.columns if c.startswith("f_ext_")]

        # Chronic viral hepatitis B/C (pre-COVID comorbidities → baseline_ext)
        chronic_inf = extract_ext_chronic_infections(cur)
        result = result.merge(chronic_inf, on="person_id", how="left")
        ext_chronic_inf_cols = [c for c in chronic_inf.columns if c.startswith("f_ext_")]

        # SARS-CoV-2 serology (baseline exposure / immunological history → baseline_ext)
        serology = extract_ext_serology(cur, include_tested_flag=include_tested_flag)
        result = result.merge(serology, on="person_id", how="left")
        ext_serology_cols = [c for c in serology.columns if c.startswith("f_ext_")]

        ext_covid_flag_cols = []

        # Type hygiene
        all_ext_flag_cols = (
            [c for c in ext_severity_cols if c != "f_ext_spo2_min" and c != "f_ext_los_max"]
            + ext_sym_cols
            + ext_chronic_inf_cols
        )
        result = _apply_type_hygiene(result, ext_bmi_cols, all_ext_flag_cols)

        # Fill binary ext features
        for cols in [ext_sym_cols, ext_vax_cols]:
            for c in cols:
                if c in result.columns:
                    result[c] = result[c].fillna(0).astype(int)

    # ---- Mechanistic Signals ----
    viral_cols = []
    immuno_cols = []
    endo_cols = []

    if include_mechanistic:
        viral_df, immuno_df, endo_df, viral_cols, immuno_cols, endo_cols = \
            extract_mechanistic_signals(cur, cohort_df)

        result = result.merge(viral_df, on="person_id", how="left")
        result = result.merge(immuno_df, on="person_id", how="left")
        result = result.merge(endo_df, on="person_id", how="left")

        for cols in [viral_cols, immuno_cols, endo_cols]:
            result[cols] = result[cols].fillna(0).astype(int)

    # ---- Build feature families dict ----
    all_ext_cols = ext_bmi_cols + ext_severity_cols + ext_wave_cols + ext_vax_cols + ext_sym_cols + ext_covid_flag_cols + ext_chronic_inf_cols + ext_serology_cols
    # Add missing indicator columns that were created by _apply_type_hygiene
    missing_ind_cols = [c for c in result.columns if c.endswith("_missing") and c.startswith("f_ext_")]
    all_ext_cols = list(dict.fromkeys(all_ext_cols + missing_ind_cols))  # deduplicate, preserve order

    # ---- Engagement controls (Step 5) ----
    eng_cols = []
    if include_engagement:
        eng_df, eng_cols = extract_engagement_features(cur, cohort_df)
        result = result.merge(eng_df, on="person_id", how="left")
        for c in eng_cols:
            result[c] = result[c].fillna(0).astype(int)

    feature_families = {
        "antony_comorbidities": comor_cols,
        "antony_symptoms": sym_cols,
        "antony_drugs": drug_cols,
        "antony_demographics": demo_cols,
        "antony_treatment": treatment_cols,
        "ext_features": all_ext_cols,
        "engagement_controls": eng_cols,
        "mechsig_viral": viral_cols,
        "mechsig_immuno": immuno_cols,
        "mechsig_endo": endo_cols,
    }

    # Summary
    print("\n" + "=" * 80)
    print("COMBINED FEATURE MATRIX SUMMARY")
    print("=" * 80)
    print(f"  Patients:       {len(result):,}")
    total = 0
    for family, cols in feature_families.items():
        if cols:
            print(f"    {family:30s}: {len(cols)} features")
            total += len(cols)
    print(f"    {'TOTAL':30s}: {total} features")
    print("=" * 80)

    result = result.drop(columns=["covid_index_date"], errors="ignore")

    return result, feature_families


# ============================================================================
# Model configuration builder
# ============================================================================

def build_model_configs(feature_families):
    """
    Build the 6 model configs from the feature families dict.

    Returns:
        dict of config_name -> list of feature column names
    """
    antony_families = ["antony_comorbidities", "antony_symptoms", "antony_drugs",
                       "antony_demographics", "antony_treatment"]
    antony_cols = []
    for fam in antony_families:
        antony_cols.extend(feature_families.get(fam, []))

    ext_cols = feature_families.get("ext_features", [])
    viral_cols = feature_families.get("mechsig_viral", [])
    immuno_cols = feature_families.get("mechsig_immuno", [])
    endo_cols = feature_families.get("mechsig_endo", [])

    baseline_ext = antony_cols + ext_cols

    configs = {
        "baseline_antony": antony_cols,
        "baseline_ext": baseline_ext,
        "mechsig_viral": baseline_ext + viral_cols,
        "mechsig_immuno": baseline_ext + immuno_cols,
        "mechsig_endo": baseline_ext + endo_cols,
        "mechsig_all": baseline_ext + viral_cols + immuno_cols + endo_cols,
    }

    print("\nModel configurations:")
    for name, cols in configs.items():
        print(f"  {name:25s}: {len(cols)} features")

    return configs

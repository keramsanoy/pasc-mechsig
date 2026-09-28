"""
Feature engineering for the Antony et al. replication.

Implements the five feature families from the paper:
  A. Pre-COVID comorbidities  (binary, on or before index date)
     - Diabetes split into uncomplicated / complicated (Antony top-30)
     - Standalone CAD feature (Ischemic heart disease, 4185932)
  B. Acute COVID symptoms     (binary, during acute phase)
     - strict_antony flag gates the 5 extra symptoms not in Antony's list
  C. Drugs during acute infection (binary per ingredient)
     - Pure percentage-based prevalence filter (min_prevalence=0.01,
       min_patient_count=0 by default — strict Antony parity)
     - Grouped f_drug_corticosteroid_any composite (Antony's #1 SHAP),
       built from pre-filter ingredient exposures so it survives even
       when individual steroids fall below the 1% threshold
     - f_drug_covid_regimen_corticosteroids (inpatient-restricted, ~3.13%)
  D. Demographics             (age numeric, gender one-hot)
  E. Treatment measures       (LOS numeric, IMV/ECMO/remdesivir binary; inpatient only)
     - WHO severity one-hot: Mild_no_ED, Mild_ED, Moderate_hosp,
       Severe_ICU_vent, Dead (placeholder — dead excluded at cohort build)

All SQL references the '#antony_cohort' temp table created by antony_cohort.py.

Documented deviations from Antony et al.:
- Symptoms default to SNOMED concept_ancestor; set use_hpo_symptoms=True
  to use HPO via OMOP2OBO (requires mapping artifact in data/omop2obo/).
- Comorbidity list is an approximation of Charlson + CDC severe-COVID conditions.
"""

import pandas as pd
import numpy as np

from omop_config import CDM_SCHEMA  # OMOP schema name; set OMOP_CDM_SCHEMA to override


def _fetch_df(cur, sql, description=""):
    """Execute a SELECT and return a DataFrame with lowercased columns."""
    if description:
        print(f"  {description}...", flush=True)
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0].lower() for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


# ============================================================================
# Family A: Pre-COVID Comorbidities
# ============================================================================

# Charlson Comorbidity Index conditions + CDC severe-COVID risk conditions
# Mapped to OMOP ancestor concept IDs
#
# Antony alignment notes:
#   - Diabetes split into uncomplicated (201820 minus complicated descendants)
#     and complicated (443767) — Antony's top-30 SHAP features report both.
#   - CAD (4185932, Ischemic heart disease) added as standalone — Antony top-30
#     at ~4.90%, distinct from broad cardiovascular (134057) and MI (4329847).
#     Concept audit (2026-05-01): swapped from 321318 (Angina pectoris, single
#     descendant; +4,343 cohort patients gained) to 4185932 which is the true
#     CAD ancestor and properly subsumes angina, MI, and chronic IHD codes.
# Concept audit (2026-05-01): rheumatic value-set canonical IDs.
# Single ancestor 257628 captured only ~30% of true rheumatic patients;
# expanded to a curated multi-ancestor set covering RA, SLE, scleroderma,
# Sjogren, polymyositis, dermatomyositis, plus a curated 8-id vasculitis
# subset (avoids the over-broad 81893 ancestor whose 449 descendants are
# dominated by phlebitis/varicose vein conditions).
RHEUMATIC_ANCESTOR_IDS = (
    80809,    # RA
    257628,   # Rheumatic disease (broad)
    134442,   # Dermatomyositis
    80182,    # SLE
    80800,    # Systemic sclerosis / scleroderma
    254443,   # Sjogren syndrome
    255348,   # Polymyositis
    313219,   # GPA (Wegener)
    4101602,  # IgA vasculitis (HSP)
    436642,   # Behcet disease
    314963,   # Giant cell arteritis
    4290976,  # Temporal arteritis
    314381,   # Kawasaki disease
    42535714, # ANCA-positive vasculitis
    196431,   # Hypersensitivity vasculitis
)

COMORBIDITY_ANCESTORS = {
    # Charlson
    "f_comor_mi":              4329847,   # Myocardial infarction
    "f_comor_chf":             316139,    # Heart failure
    "f_comor_pvd":             321052,    # Peripheral vascular disease
    "f_comor_cerebrovascular": 381591,    # Cerebrovascular disease
    "f_comor_dementia":        4182210,   # Dementia
    "f_comor_chronic_pulm":    4063381,   # Chronic lower respiratory disease
    "f_comor_rheumatic":       RHEUMATIC_ANCESTOR_IDS,  # was: 257628 (RA only) — missed ~70% of cohort rheumatic patients
    "f_comor_peptic_ulcer":    4247120,   # Peptic ulcer disease
    "f_comor_liver_mild":      4212540,   # Chronic liver disease
    "f_comor_diabetes_complicated": 443767, # Diabetes mellitus with complication (Charlson complicated)
    "f_comor_hemiplegia":      374022,    # Hemiplegia/paraplegia (updated)
    "f_comor_renal":           198124,    # Chronic kidney disease
    "f_comor_cancer":          443392,    # Malignant neoplasm
    "f_comor_hiv":             439727,    # HIV/AIDS
    # CDC severe-COVID risk
    "f_comor_cardiovascular":  134057,    # Disorder of cardiovascular system
    "f_comor_cad":             4185932,   # Ischemic heart disease (was: 321318 Angina pectoris — concept-audit swap, +4,343 patients)
    "f_comor_obesity":         433736,    # Obesity
    "f_comor_depression":      440383,    # Depressive disorder
    "f_comor_anxiety":         442077,    # Anxiety disorder
    "f_comor_autoimmune":      434621,    # Autoimmune disease
    "f_comor_hypertension":    316866,    # Hypertensive disorder
}

# Diabetes uncomplicated uses the broad DM ancestor (201820) but excludes
# descendants of the complicated-diabetes ancestor (443767).  Handled with
# special SQL logic in extract_comorbidities() rather than a simple CASE.
DIABETES_UNCOMPLICATED_ANCESTOR = 201820   # Diabetes mellitus (broad)
DIABETES_COMPLICATED_ANCESTOR   = 443767   # Diabetes mellitus with complication


def extract_comorbidities(cur):
    """
    Extract pre-COVID comorbidity features.

    Window: on or before covid_index_date.
    Uses concept_ancestor hierarchy for broad matching.

    Special handling for diabetes: ``f_comor_diabetes_uncomplicated`` uses
    the broad DM ancestor (201820) but *excludes* any condition that is
    also a descendant of the complicated-diabetes ancestor (443767).

    Returns:
        DataFrame with person_id + binary comorbidity columns.
    """
    print("\n[Family A] Extracting pre-COVID comorbidities...")

    # Build CASE expressions for each comorbidity in the standard dict.
    # Values may be a single int (single ancestor) or a tuple of ints
    # (multi-ancestor union, e.g. f_comor_rheumatic post-audit).
    case_exprs = []
    for col_name, ancestor_id in COMORBIDITY_ANCESTORS.items():
        if isinstance(ancestor_id, (tuple, list)):
            ids_csv = ", ".join(str(x) for x in ancestor_id)
            case_exprs.append(
                f"MAX(CASE WHEN ca.ancestor_concept_id IN ({ids_csv}) THEN 1 ELSE 0 END) AS {col_name}"
            )
        else:
            case_exprs.append(
                f"MAX(CASE WHEN ca.ancestor_concept_id = {ancestor_id} THEN 1 ELSE 0 END) AS {col_name}"
            )

    # Diabetes-uncomplicated: ancestor 201820 BUT condition NOT a descendant of 443767
    case_exprs.append(
        f"MAX(CASE WHEN ca.ancestor_concept_id = {DIABETES_UNCOMPLICATED_ANCESTOR} "
        f"AND bc.condition_concept_id NOT IN "
        f"(SELECT descendant_concept_id FROM {CDM_SCHEMA}.concept_ancestor "
        f"WHERE ancestor_concept_id = {DIABETES_COMPLICATED_ANCESTOR}) "
        f"THEN 1 ELSE 0 END) AS f_comor_diabetes_uncomplicated"
    )

    case_sql = ",\n        ".join(case_exprs)

    # We need the original condition_concept_id for the diabetes-uncomplicated
    # exclusion, so carry it through from baseline_conditions.
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
            bc.condition_concept_id,
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
    LEFT JOIN baseline_conditions bc
      ON bc.person_id = ca.person_id
     AND bc.condition_concept_id = ca.condition_concept_id
    GROUP BY c.person_id
    """

    n_features = len(COMORBIDITY_ANCESTORS) + 1  # +1 for diabetes_uncomplicated
    df = _fetch_df(cur, sql, f"Querying {n_features} comorbidity ancestors")
    print(f"  Extracted comorbidity features for {len(df):,} patients")
    return df


# ============================================================================
# Family B: Acute COVID Symptoms
# ============================================================================

# Ancestor concept IDs for acute symptoms (SNOMED hierarchy)
SYMPTOM_ANCESTORS = {
    "f_sym_fever":         437663,    # Fever
    "f_sym_cough":         254761,    # Cough
    "f_sym_fatigue":       4223659,   # Fatigue
    "f_sym_dyspnea":       312437,    # Dyspnea
    "f_sym_myalgia":       442752,    # Myalgia
    "f_sym_headache":      378253,    # Headache
    "f_sym_anosmia":       4185711,   # Anosmia
    "f_sym_ageusia":       4289517,  # Ageusia (updated)
    "f_sym_diarrhea":      196523,    # Diarrhea
    "f_sym_nausea":        31967,     # Nausea
    "f_sym_sore_throat":   4147326,    # Sore throat / pharyngitis (updated)
    "f_sym_chest_pain":    77670,     # Chest pain
    "f_sym_abdominal_pain": 200219,   # Abdominal pain
    "f_sym_rhinorrhea":    4276172,   # Nasal discharge (was 4100065 = "Disease caused by Coronaviridae" — wrong!)
    "f_sym_arthralgia":    77074,     # Arthralgia (updated)
    "f_sym_vomiting":      441408,    # Vomiting
    "f_sym_malaise":       4272240,   # Malaise
    "f_sym_dizziness":     4223938,   # Dizziness
    "f_sym_congestion":    4195085,   # Nasal congestion
}

# Five symptoms added beyond Antony's original SNOMED list.
# When strict_antony=True these are excluded from Family B.
ANTONY_EXTRA_SYMPTOMS = {
    "f_sym_abdominal_pain",
    "f_sym_dizziness",
    "f_sym_malaise",
    "f_sym_rhinorrhea",
    "f_sym_congestion",
}


def extract_acute_symptoms(cur, strict_antony=False):
    """
    Extract acute-phase symptom features.

    Window: acute_start to acute_end (patient-specific).
    Uses SNOMED concept_ancestor hierarchy for symptom matching.

    Args:
        cur: Database cursor.
        strict_antony: If True, exclude the five symptoms added beyond
            Antony's original list (abdominal pain, dizziness, malaise,
            rhinorrhea, congestion).

    Returns:
        DataFrame with person_id + binary symptom columns.
    """
    print("\n[Family B] Extracting acute-phase symptoms...")

    if strict_antony:
        symptoms = {k: v for k, v in SYMPTOM_ANCESTORS.items()
                    if k not in ANTONY_EXTRA_SYMPTOMS}
        print(f"  strict_antony=True → using {len(symptoms)}/{ len(SYMPTOM_ANCESTORS)} symptoms")
    else:
        symptoms = SYMPTOM_ANCESTORS

    case_exprs = []
    for col_name, ancestor_id in symptoms.items():
        case_exprs.append(
            f"MAX(CASE WHEN sa.ancestor_concept_id = {ancestor_id} THEN 1 ELSE 0 END) AS {col_name}"
        )
    case_sql = ",\n        ".join(case_exprs)

    sql = f"""
    WITH cohort AS (
        SELECT person_id, acute_start, acute_end
        FROM "#antony_cohort"
    ),
    acute_conditions AS (
        SELECT DISTINCT
            c.person_id,
            co.condition_concept_id
        FROM cohort c
        JOIN {CDM_SCHEMA}.condition_occurrence co
          ON co.person_id = c.person_id
         AND co.condition_start_date BETWEEN c.acute_start AND c.acute_end
    ),
    symptom_ancestors AS (
        SELECT
            ac.person_id,
            ca.ancestor_concept_id
        FROM acute_conditions ac
        JOIN {CDM_SCHEMA}.concept_ancestor ca
          ON ca.descendant_concept_id = ac.condition_concept_id
    )
    SELECT
        c.person_id,
        {case_sql}
    FROM cohort c
    LEFT JOIN symptom_ancestors sa
      ON sa.person_id = c.person_id
    GROUP BY c.person_id
    """

    df = _fetch_df(cur, sql, f"Querying {len(symptoms)} symptom ancestors")
    print(f"  Extracted symptom features for {len(df):,} patients")
    return df


def extract_acute_symptoms_hpo(cur, concept_to_hpo, strict_antony=False):
    """
    Extract acute-phase symptom features using HPO via OMOP2OBO mappings.

    Instead of joining concept_ancestor in SQL, this pulls raw
    condition_concept_id values and maps them to HPO terms in Python
    using the pre-built OMOP2OBO lookup.

    Args:
        cur: Database cursor.
        concept_to_hpo: Dict mapping OMOP concept_id (int) → set of HPO IDs.
            Built by omop2obo_mapping.build_concept_to_hpo_lookup().
        strict_antony: If True, use only Antony's 14 core HPO symptom terms.
            If False, include 5 extra symptoms (same gating as SNOMED version).

    Returns:
        DataFrame with person_id + binary f_sym_hpo_* columns.
    """
    from omop2obo_mapping import (
        HPO_SYMPTOM_TERMS, HPO_ALL_SYMPTOM_TERMS, HPO_EXTRA_SYMPTOM_TERMS,
    )

    print("\n[Family B] Extracting acute-phase symptoms (HPO via OMOP2OBO)...")

    if strict_antony:
        target_terms = HPO_SYMPTOM_TERMS
        print(f"  strict_antony=True → using {len(target_terms)} core HPO terms")
    else:
        target_terms = HPO_ALL_SYMPTOM_TERMS
        print(f"  Using all {len(target_terms)} HPO symptom terms "
              f"(14 core + {len(HPO_EXTRA_SYMPTOM_TERMS)} extra)")

    # Build inverse: HPO term → feature column name
    hpo_to_feature = {hpo: col for col, hpo in target_terms.items()}
    target_hpo_ids = set(target_terms.values())

    # Pull all acute-window condition concept IDs per patient
    sql = f"""
    SELECT DISTINCT
        c.person_id,
        co.condition_concept_id
    FROM "#antony_cohort" c
    JOIN {CDM_SCHEMA}.condition_occurrence co
      ON co.person_id = c.person_id
     AND co.condition_start_date BETWEEN c.acute_start AND c.acute_end
    """
    raw = _fetch_df(cur, sql, "Querying acute conditions for HPO mapping")

    # Get all person_ids from cohort (for LEFT JOIN semantics)
    cohort_sql = 'SELECT person_id FROM "#antony_cohort"'
    cohort_ids = _fetch_df(cur, cohort_sql)

    # Map condition_concept_id → HPO terms → feature columns
    # For each patient, OR across all condition records
    patient_features = {pid: {col: 0 for col in target_terms}
                        for pid in cohort_ids["person_id"]}

    n_mapped_events = 0
    n_total_events = len(raw)

    for _, row in raw.iterrows():
        pid = row["person_id"]
        cid = int(row["condition_concept_id"])
        hpo_ids = concept_to_hpo.get(cid, set())
        for hpo_id in hpo_ids:
            if hpo_id in target_hpo_ids:
                col = hpo_to_feature[hpo_id]
                if pid in patient_features:
                    patient_features[pid][col] = 1
                    n_mapped_events += 1

    df = pd.DataFrame.from_dict(patient_features, orient="index")
    df.index.name = "person_id"
    df = df.reset_index()

    # Ensure int dtype
    for col in target_terms:
        if col in df.columns:
            df[col] = df[col].astype(int)

    print(f"  Mapped {n_mapped_events:,} of {n_total_events:,} condition events "
          f"to target HPO terms")
    print(f"  Extracted HPO symptom features for {len(df):,} patients")
    return df


# ============================================================================
# Family C: Drugs during acute infection
# ============================================================================

# Ingredients comprising the systemic-corticosteroid composite (Item 3).
# Antony's #1 SHAP feature is "Systemic Corticosteroids" as a class.
CORTICOSTEROID_INGREDIENTS = {
    "prednisone", "dexamethasone", "methylprednisolone",
    "hydrocortisone", "prednisolone", "betamethasone", "triamcinolone",
}


def extract_acute_drugs(cur, min_patient_count=0, min_prevalence=0.01):
    """
    Extract acute-phase drug features grouped by active ingredient.

    Window: acute_start to acute_end.
    Groups drugs by RxNorm ingredient using concept_ancestor.

    Filtering: Antony drops drugs below ``min_prevalence`` (1% by default)
    of the cohort.  For strict parity the default uses a pure
    percentage-based threshold, i.e. ``max(1, ceil(min_prevalence * N))``.
    ``min_patient_count`` is retained as an optional absolute floor but
    defaults to 0 (disabled) to match Antony exactly; set it to a positive
    integer if you need an additional safety floor on very small cohorts.

    After the per-ingredient pivot a grouped ``f_drug_corticosteroid_any``
    composite column is added (OR across the standard systemic
    corticosteroid ingredients).  The composite is computed from the
    *unfiltered* ingredient exposures so that a class-level signal is
    preserved even when individual steroid ingredients fall below the
    prevalence threshold — this matches Antony's top-ranked
    "Systemic Corticosteroids" class feature.  Individual ingredient
    columns that pass the prevalence filter are retained in parallel so
    SHAP can resolve either level.

    Returns:
        DataFrame with person_id + binary drug ingredient columns (f_drug_<ingredient_name>).
    """
    # Compute effective threshold (Antony: pure 1% prevalence filter)
    n_patients_sql = 'SELECT COUNT(*) AS n FROM "#antony_cohort"'
    cur.execute(n_patients_sql)
    n_patients = cur.fetchone()[0]
    prevalence_threshold = max(1, int(np.ceil(min_prevalence * n_patients)))
    threshold = max(min_patient_count, prevalence_threshold)

    floor_note = (
        f", floor={min_patient_count}" if min_patient_count > 0 else ""
    )
    print(f"\n[Family C] Extracting acute-phase drugs "
          f"(threshold={threshold}, {min_prevalence:.0%} of {n_patients:,}"
          f"{floor_note})...")

    # Step 1: Get drug exposures in acute window, resolve to ingredients
    sql = f"""
    WITH cohort AS (
        SELECT person_id, acute_start, acute_end
        FROM "#antony_cohort"
    ),
    acute_drugs AS (
        SELECT DISTINCT
            c.person_id,
            de.drug_concept_id
        FROM cohort c
        JOIN {CDM_SCHEMA}.drug_exposure de
          ON de.person_id = c.person_id
         AND de.drug_exposure_start_date BETWEEN c.acute_start AND c.acute_end
         AND de.drug_concept_id > 0
    ),
    drug_to_ingredient AS (
        SELECT DISTINCT
            ad.person_id,
            ca.ancestor_concept_id AS ingredient_concept_id
        FROM acute_drugs ad
        JOIN {CDM_SCHEMA}.concept_ancestor ca
          ON ca.descendant_concept_id = ad.drug_concept_id
        JOIN {CDM_SCHEMA}.concept con
          ON con.concept_id = ca.ancestor_concept_id
         AND con.concept_class_id = 'Ingredient'
         AND con.vocabulary_id = 'RxNorm'
    )
    SELECT
        person_id,
        ingredient_concept_id
    FROM drug_to_ingredient
    """

    drug_df = _fetch_df(cur, sql, "Resolving drugs to ingredients")
    print(f"  Raw patient-ingredient pairs: {len(drug_df):,}")

    if drug_df.empty:
        print("  WARNING: No drug exposures found in acute windows")
        # Return just person_ids with no drug columns
        sql_ids = 'SELECT person_id FROM "#antony_cohort"'
        return _fetch_df(cur, sql_ids)

    # Step 2: Get ingredient names
    unique_ingredients = drug_df["ingredient_concept_id"].unique()
    ing_list = ", ".join(str(int(x)) for x in unique_ingredients)

    name_sql = f"""
    SELECT concept_id, concept_name
    FROM {CDM_SCHEMA}.concept
    WHERE concept_id IN ({ing_list})
    """
    names = _fetch_df(cur, name_sql, "Fetching ingredient names")
    id_to_name = dict(zip(names["concept_id"], names["concept_name"]))

    # Step 3a: Identify patients on ANY systemic corticosteroid BEFORE
    # prevalence filtering.  This preserves Antony's #1 SHAP class-level
    # feature "Systemic Corticosteroids" even when individual steroid
    # ingredients (e.g. triamcinolone, betamethasone) fall below the 1%
    # prevalence threshold and are dropped from the per-ingredient columns.
    drug_df["_ingredient_name_lower"] = (
        drug_df["ingredient_concept_id"].map(id_to_name).fillna("").str.lower()
    )
    steroid_mask = drug_df["_ingredient_name_lower"].apply(
        lambda nm: any(ing in nm for ing in CORTICOSTEROID_INGREDIENTS)
    )
    steroid_patients = set(
        drug_df.loc[steroid_mask, "person_id"].unique().tolist()
    )
    steroid_ingredients_seen = sorted(
        drug_df.loc[steroid_mask, "_ingredient_name_lower"].unique().tolist()
    )
    drug_df = drug_df.drop(columns=["_ingredient_name_lower"])

    # Step 3b: Filter by effective threshold (percentage-based, Antony parity)
    patient_counts = drug_df.groupby("ingredient_concept_id")["person_id"].nunique()
    qualifying = patient_counts[patient_counts >= threshold].index.tolist()
    drug_df = drug_df[drug_df["ingredient_concept_id"].isin(qualifying)].copy()

    print(f"  Ingredients with >= {threshold} patients: {len(qualifying)}")

    # Step 4: Pivot to wide format
    drug_df["ingredient_name"] = drug_df["ingredient_concept_id"].map(id_to_name)
    drug_df["ingredient_name"] = drug_df["ingredient_name"].fillna("unknown")

    # Clean ingredient names for column names
    def _clean_name(name):
        return "f_drug_" + (
            str(name)
            .lower()
            .replace(" ", "_")
            .replace(",", "")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace(".", "")
            .replace("'", "")
        )[:60]

    drug_df["col_name"] = drug_df["ingredient_name"].apply(_clean_name)
    drug_df["value"] = 1

    # Pivot: person_id x col_name
    pivot = drug_df.pivot_table(
        index="person_id", columns="col_name", values="value",
        aggfunc="max", fill_value=0,
    ).reset_index()
    pivot.columns.name = None

    # Merge with full cohort to ensure all patients present
    all_ids = _fetch_df(cur, 'SELECT person_id FROM "#antony_cohort"')
    result = all_ids.merge(pivot, on="person_id", how="left")

    # Fill NaN with 0 for drug columns
    drug_cols = [c for c in result.columns if c.startswith("f_drug_")]
    result[drug_cols] = result[drug_cols].fillna(0).astype(int)

    # Step 5: Add grouped corticosteroid composite (Antony's #1 SHAP feature).
    # Computed from the pre-filter ingredient exposures (steroid_patients) so
    # the class-level signal is preserved regardless of per-ingredient
    # prevalence filtering.
    result["f_drug_corticosteroid_any"] = (
        result["person_id"].isin(steroid_patients).astype(int)
    )
    drug_cols.append("f_drug_corticosteroid_any")
    n_steroid = int(result["f_drug_corticosteroid_any"].sum())
    if steroid_ingredients_seen:
        print(f"  Corticosteroid composite: OR across "
              f"{len(steroid_ingredients_seen)} pre-filter ingredients "
              f"({', '.join(steroid_ingredients_seen)}) → "
              f"{n_steroid:,} patients "
              f"({100.0 * n_steroid / max(1, len(result)):.2f}%)")
    else:
        print("  WARNING: No systemic corticosteroid ingredients found for composite")

    print(f"  Final drug features: {len(drug_cols)} columns (incl. corticosteroid composite)")
    return result


# ============================================================================
# Family C addendum: COVID-regimen corticosteroids (Item 4)
# ============================================================================

# RxNorm ingredient concept IDs for systemic corticosteroids
CORTICOSTEROID_RXNORM_ANCESTORS = (
    1518254,   # Prednisone
    1518978,   # Dexamethasone
    1506270,   # Methylprednisolone
    1507705,   # Hydrocortisone (systemic)
    1550557,   # Prednisolone
    920458,    # Betamethasone
    903963,    # Triamcinolone
)


def extract_covid_regimen_corticosteroids(cur):
    """
    Extract 'COVID Regimen Corticosteroids' feature (Antony top-30, ~3.13%).

    Distinct from the general systemic-corticosteroid composite: this is
    restricted to corticosteroid ingredients administered during the acute
    window *and* during an inpatient visit concurrent with the COVID
    admission.

    Returns:
        DataFrame with person_id + binary ``f_drug_covid_regimen_corticosteroids``.
    """
    print("\n[Family C+] Extracting COVID-regimen corticosteroids (inpatient-restricted)...")

    steroid_ancestors = ", ".join(str(c) for c in CORTICOSTEROID_RXNORM_ANCESTORS)
    inp_visits = ", ".join(str(c) for c in (9201, 262))   # Inpatient Visit, ER+Inpatient

    sql = f"""
    WITH inp_cohort AS (
        SELECT person_id, acute_start, acute_end, covid_index_date, discharge_date
        FROM "#antony_cohort"
        WHERE is_inpatient = 1
    ),
    covid_inpatient_visits AS (
        SELECT DISTINCT vo.person_id, vo.visit_occurrence_id,
               vo.visit_start_date, COALESCE(vo.visit_end_date, vo.visit_start_date) AS visit_end
        FROM inp_cohort ic
        JOIN {CDM_SCHEMA}.visit_occurrence vo
          ON vo.person_id = ic.person_id
         AND vo.visit_concept_id IN ({inp_visits})
         AND vo.visit_start_date <= ADD_DAYS(ic.covid_index_date, 16)
         AND COALESCE(vo.visit_end_date, vo.visit_start_date) >= ADD_DAYS(ic.covid_index_date, -1)
    ),
    regimen_steroids AS (
        SELECT DISTINCT de.person_id
        FROM inp_cohort ic
        JOIN {CDM_SCHEMA}.drug_exposure de
          ON de.person_id = ic.person_id
         AND de.drug_exposure_start_date BETWEEN ic.acute_start AND ic.acute_end
        JOIN {CDM_SCHEMA}.concept_ancestor ca
          ON ca.descendant_concept_id = de.drug_concept_id
         AND ca.ancestor_concept_id IN ({steroid_ancestors})
        JOIN covid_inpatient_visits civ
          ON civ.person_id = de.person_id
         AND de.drug_exposure_start_date BETWEEN civ.visit_start_date AND civ.visit_end
    )
    SELECT
        c.person_id,
        CASE WHEN rs.person_id IS NOT NULL THEN 1 ELSE 0 END
            AS f_drug_covid_regimen_corticosteroids
    FROM "#antony_cohort" c
    LEFT JOIN regimen_steroids rs ON rs.person_id = c.person_id
    """

    df = _fetch_df(cur, sql, "Querying COVID-regimen corticosteroids")
    n_pos = df["f_drug_covid_regimen_corticosteroids"].sum()
    print(f"  COVID-regimen corticosteroids: {n_pos:,} patients positive")
    return df


# ============================================================================
# Family E addendum: WHO severity one-hot (Item 5)
# ============================================================================

# ED visit concept IDs
ED_VISIT_CONCEPTS = (9203, 262)   # Emergency Room Visit, ER+Inpatient


def extract_who_severity(cur):
    """
    Emit an explicit WHO severity category one-hot encoding.

    Antony's top-30 SHAP output includes severity as one-hot:
      ``Mild_no_ED`` (85.79%), ``Mild_ED``, ``Moderate_hosp``,
      ``Severe_ICU_vent``, ``Dead``.

    Categories are mutually exclusive.  Because the cohort already
    excludes deceased patients, ``f_tx_severity_who_dead`` is always 0
    (retained for schema completeness).

    Derivation:
      - Dead:           always 0 (excluded at cohort build)
      - Severe_ICU_vent: inpatient + IMV or ventilation
      - Moderate_hosp:  inpatient, not severe
      - Mild_ED:        outpatient with ED visit in acute window
      - Mild_no_ED:     outpatient without ED visit in acute window

    Returns:
        DataFrame with person_id + 5 binary severity columns.
    """
    print("\n[Family E+] Extracting WHO severity one-hot...")

    imv_ancestors = ", ".join(str(c) for c in IMV_PROCEDURE_ANCESTORS)
    ed_visits = ", ".join(str(c) for c in ED_VISIT_CONCEPTS)

    sql = f"""
    WITH cohort AS (
        SELECT person_id, acute_start, acute_end, is_inpatient
        FROM "#antony_cohort"
    ),

    -- Detect IMV / ventilation (same logic as extract_treatment_measures)
    imv_flag AS (
        SELECT DISTINCT c.person_id
        FROM cohort c
        JOIN {CDM_SCHEMA}.procedure_occurrence po
          ON po.person_id = c.person_id
         AND po.procedure_date BETWEEN c.acute_start AND c.acute_end
        JOIN {CDM_SCHEMA}.concept_ancestor ca
          ON ca.descendant_concept_id = po.procedure_concept_id
         AND ca.ancestor_concept_id IN ({imv_ancestors})
        WHERE c.is_inpatient = 1

        UNION

        SELECT DISTINCT c.person_id
        FROM cohort c
        JOIN {CDM_SCHEMA}.observation o
          ON o.person_id = c.person_id
         AND o.observation_date BETWEEN c.acute_start AND c.acute_end
         AND o.observation_concept_id = {VENT_MODE_OBS_CONCEPT_ID}
        WHERE c.is_inpatient = 1
    ),

    -- Detect ED visits during acute window (for outpatients)
    ed_flag AS (
        SELECT DISTINCT c.person_id
        FROM cohort c
        JOIN {CDM_SCHEMA}.visit_occurrence vo
          ON vo.person_id = c.person_id
         AND vo.visit_concept_id IN ({ed_visits})
         AND vo.visit_start_date BETWEEN c.acute_start AND c.acute_end
        WHERE c.is_inpatient = 0
    )

    SELECT
        c.person_id,
        0 AS f_tx_severity_who_dead,
        CASE WHEN c.is_inpatient = 1 AND imv.person_id IS NOT NULL
             THEN 1 ELSE 0 END AS f_tx_severity_who_severe_icu_vent,
        CASE WHEN c.is_inpatient = 1 AND imv.person_id IS NULL
             THEN 1 ELSE 0 END AS f_tx_severity_who_moderate_hosp,
        CASE WHEN c.is_inpatient = 0 AND ed.person_id IS NOT NULL
             THEN 1 ELSE 0 END AS f_tx_severity_who_mild_ed,
        CASE WHEN c.is_inpatient = 0 AND ed.person_id IS NULL
             THEN 1 ELSE 0 END AS f_tx_severity_who_mild_no_ed
    FROM cohort c
    LEFT JOIN imv_flag imv ON imv.person_id = c.person_id
    LEFT JOIN ed_flag ed   ON ed.person_id = c.person_id
    """

    df = _fetch_df(cur, sql, "Computing WHO severity categories")
    severity_cols = [c for c in df.columns if c.startswith("f_tx_severity_who_")]
    for col in severity_cols:
        n = df[col].sum()
        pct = n / len(df) * 100 if len(df) > 0 else 0
        print(f"  {col}: {n:,} ({pct:.1f}%)")
    return df


# ============================================================================
# Family D: Demographics
# ============================================================================

def extract_demographics(cur):
    """
    Extract demographic features.

    - Age at COVID index date (numeric)
    - Gender: one-hot encoded (female, male, unknown)

    Returns:
        DataFrame with person_id, f_age, f_sex_female, f_sex_male, f_sex_unknown
    """
    print("\n[Family D] Extracting demographics...")

    sql = """
    SELECT person_id, age_at_index, gender_concept_id
    FROM "#antony_cohort"
    """
    df = _fetch_df(cur, sql, "Fetching demographics from temp table")

    df["f_age"] = pd.to_numeric(df["age_at_index"], errors="coerce")
    df["f_sex_female"] = (df["gender_concept_id"] == 8532).astype(int)
    df["f_sex_male"] = (df["gender_concept_id"] == 8507).astype(int)
    df["f_sex_unknown"] = (
        (~df["gender_concept_id"].isin([8532, 8507])) | df["gender_concept_id"].isna()
    ).astype(int)

    df = df[["person_id", "f_age", "f_sex_female", "f_sex_male", "f_sex_unknown"]]
    print(f"  Extracted demographics for {len(df):,} patients")
    return df


# ============================================================================
# Family E: Treatment Measures (inpatient only)
# ============================================================================

# Ventilation / IMV concepts
IMV_PROCEDURE_ANCESTORS = (4230167,)   # Artificial respiration (mechanical ventilation ancestor)
ECMO_PROCEDURE_ANCESTORS = (4052536,)  # Extracorporeal membrane oxygenation
VENT_MODE_OBS_CONCEPT_ID = 3004921     # Ventilation mode observation

# Remdesivir ingredient concept
REMDESIVIR_ANCESTORS = (37499271,)     # Remdesivir (ancestor for all formulations)


def extract_treatment_measures(cur):
    """
    Extract treatment-measure features for inpatients.

    - Length of hospital stay (numeric)
    - IMV indicator (mechanical ventilation)
    - ECMO indicator
    - Remdesivir during hospitalization indicator

    Only computed for inpatient patients.

    Returns:
        DataFrame with person_id + treatment columns.
    """
    print("\n[Family E] Extracting treatment measures (inpatient only)...")

    imv_ancestors = ", ".join(str(c) for c in IMV_PROCEDURE_ANCESTORS)
    ecmo_ancestors = ", ".join(str(c) for c in ECMO_PROCEDURE_ANCESTORS)
    rds_ancestors = ", ".join(str(c) for c in REMDESIVIR_ANCESTORS)

    sql = f"""
    WITH inp_cohort AS (
        SELECT person_id, covid_index_date, acute_start, acute_end, discharge_date
        FROM "#antony_cohort"
        WHERE is_inpatient = 1
    ),

    -- Length of stay
    los AS (
        SELECT
            person_id,
            DAYS_BETWEEN(covid_index_date, discharge_date) AS f_tx_los
        FROM inp_cohort
    ),

    -- IMV: procedures that are descendants of mechanical ventilation
    imv_via_proc AS (
        SELECT DISTINCT c.person_id, 1 AS flag
        FROM inp_cohort c
        JOIN {CDM_SCHEMA}.procedure_occurrence po
          ON po.person_id = c.person_id
         AND po.procedure_date BETWEEN c.acute_start AND c.acute_end
        JOIN {CDM_SCHEMA}.concept_ancestor ca
          ON ca.descendant_concept_id = po.procedure_concept_id
         AND ca.ancestor_concept_id IN ({imv_ancestors})
    ),

    -- IMV: also check observation table (ventilation mode)
    imv_via_obs AS (
        SELECT DISTINCT c.person_id, 1 AS flag
        FROM inp_cohort c
        JOIN {CDM_SCHEMA}.observation o
          ON o.person_id = c.person_id
         AND o.observation_date BETWEEN c.acute_start AND c.acute_end
         AND o.observation_concept_id = {VENT_MODE_OBS_CONCEPT_ID}
    ),

    -- ECMO: procedures that are descendants of ECMO
    ecmo AS (
        SELECT DISTINCT c.person_id, 1 AS flag
        FROM inp_cohort c
        JOIN {CDM_SCHEMA}.procedure_occurrence po
          ON po.person_id = c.person_id
         AND po.procedure_date BETWEEN c.acute_start AND c.acute_end
        JOIN {CDM_SCHEMA}.concept_ancestor ca
          ON ca.descendant_concept_id = po.procedure_concept_id
         AND ca.ancestor_concept_id IN ({ecmo_ancestors})
    ),

    -- Remdesivir during hospitalization
    remdesivir AS (
        SELECT DISTINCT c.person_id, 1 AS flag
        FROM inp_cohort c
        JOIN {CDM_SCHEMA}.drug_exposure de
          ON de.person_id = c.person_id
         AND de.drug_exposure_start_date BETWEEN c.acute_start AND c.acute_end
        JOIN {CDM_SCHEMA}.concept_ancestor ca
          ON ca.descendant_concept_id = de.drug_concept_id
         AND ca.ancestor_concept_id IN ({rds_ancestors})
    )

    SELECT
        c.person_id,
        COALESCE(l.f_tx_los, 0) AS f_tx_los,
        CASE WHEN ip.flag = 1 OR io.flag = 1 THEN 1 ELSE 0 END AS f_tx_imv,
        COALESCE(e.flag, 0)  AS f_tx_ecmo,
        COALESCE(r.flag, 0)  AS f_tx_remdesivir
    FROM inp_cohort c
    LEFT JOIN los l            ON l.person_id = c.person_id
    LEFT JOIN imv_via_proc ip  ON ip.person_id = c.person_id
    LEFT JOIN imv_via_obs io   ON io.person_id = c.person_id
    LEFT JOIN ecmo e           ON e.person_id = c.person_id
    LEFT JOIN remdesivir r     ON r.person_id = c.person_id
    """

    df = _fetch_df(cur, sql, "Querying treatment measures")
    print(f"  Extracted treatment features for {len(df):,} inpatients")
    return df


# ============================================================================
# Orchestrator: Build full feature matrix
# ============================================================================

def build_antony_feature_matrix(cur, cohort_df, include_treatment=True,
                                strict_antony=False, use_hpo_symptoms=False):
    """
    Build the complete feature matrix for an Antony-style cohort.

    Args:
        cur: Database cursor
        cohort_df: DataFrame with at least person_id, label, is_inpatient
        include_treatment: Whether to include Family E (treatment measures).
                          Set False for outpatient-only cohorts.
        strict_antony: If True, exclude the five extra acute symptoms and
            enable all strict-replication alignments.
        use_hpo_symptoms: If True, use HPO-based symptom extraction via
            OMOP2OBO mappings instead of SNOMED concept_ancestor.
            Requires data/omop2obo/ mapping artifact.

    Returns:
        Tuple of:
        - feature_df: DataFrame with person_id + all feature columns + label
        - feature_families: dict mapping family name -> list of column names
    """
    mode_parts = []
    if strict_antony:
        mode_parts.append("strict")
    if use_hpo_symptoms:
        mode_parts.append("HPO")
    mode_str = " (" + ", ".join(mode_parts) + ")" if mode_parts else ""

    print("\n" + "=" * 80)
    print(f"BUILDING ANTONY FEATURE MATRIX{mode_str}")
    print("=" * 80)

    # Extract each family
    comorbidities = extract_comorbidities(cur)

    if use_hpo_symptoms:
        from omop2obo_mapping import (
            load_omop2obo_condition_mappings,
            build_concept_to_hpo_lookup,
            TRUSTED_CATEGORIES,
        )
        mapping_df = load_omop2obo_condition_mappings(
            quality_filter=TRUSTED_CATEGORIES)
        concept_to_hpo = build_concept_to_hpo_lookup(mapping_df)
        symptoms = extract_acute_symptoms_hpo(
            cur, concept_to_hpo, strict_antony=strict_antony)
    else:
        symptoms = extract_acute_symptoms(cur, strict_antony=strict_antony)

    drugs = extract_acute_drugs(cur)
    covid_regimen = extract_covid_regimen_corticosteroids(cur)
    demographics = extract_demographics(cur)

    # Merge all onto cohort
    result = cohort_df[["person_id", "label"]].copy()
    result = result.merge(comorbidities, on="person_id", how="left")
    result = result.merge(symptoms, on="person_id", how="left")
    result = result.merge(drugs, on="person_id", how="left")
    result = result.merge(covid_regimen, on="person_id", how="left")
    result = result.merge(demographics, on="person_id", how="left")

    # Build feature family mapping
    comor_cols = [c for c in comorbidities.columns if c.startswith("f_comor_")]
    sym_cols = [c for c in symptoms.columns if c.startswith("f_sym_")]
    drug_cols = [c for c in drugs.columns if c.startswith("f_drug_")]
    # COVID-regimen corticosteroids goes into the drug family
    if "f_drug_covid_regimen_corticosteroids" in result.columns:
        drug_cols.append("f_drug_covid_regimen_corticosteroids")
    demo_cols = [c for c in demographics.columns if c.startswith("f_")]
    treatment_cols = []

    if include_treatment:
        treatment = extract_treatment_measures(cur)
        who_severity = extract_who_severity(cur)
        result = result.merge(treatment, on="person_id", how="left")
        result = result.merge(who_severity, on="person_id", how="left")
        treatment_cols = [c for c in treatment.columns if c.startswith("f_tx_")]
        treatment_cols += [c for c in who_severity.columns if c.startswith("f_tx_")]
        # Fill treatment NaN with 0 for outpatients in all-patient cohort
        result[treatment_cols] = result[treatment_cols].fillna(0)

    # Fill remaining NaNs
    binary_groups = [comor_cols, sym_cols, drug_cols]
    if "f_drug_covid_regimen_corticosteroids" in result.columns:
        result["f_drug_covid_regimen_corticosteroids"] = (
            result["f_drug_covid_regimen_corticosteroids"].fillna(0).astype(int))
    for cols in binary_groups:
        present = [c for c in cols if c in result.columns]
        result[present] = result[present].fillna(0).astype(int)

    feature_families = {
        "comorbidities": comor_cols,
        "symptoms": sym_cols,
        "drugs": drug_cols,
        "demographics": demo_cols,
        "treatment": treatment_cols,
    }

    all_features = comor_cols + sym_cols + drug_cols + demo_cols + treatment_cols

    print("\n" + "=" * 80)
    print("FEATURE MATRIX SUMMARY")
    print("=" * 80)
    print(f"  Patients:       {len(result):,}")
    print(f"  Total features: {len(all_features)}")
    for family, cols in feature_families.items():
        print(f"    {family:20s}: {len(cols)} features")
    print("=" * 80)

    return result, feature_families


# ============================================================================
# baseline_antony_strict convenience wrapper (Item 15)
# ============================================================================

def build_antony_strict_feature_matrix(cur, cohort_df, include_treatment=True,
                                       use_hpo_symptoms=False):
    """
    Build the strict Antony-replication feature matrix.

    Bundles all strict-replication alignments (Items 1–5, 8):
      - Split diabetes (uncomplicated / complicated)
      - Standalone CAD feature
      - Corticosteroid composites (general + COVID-regimen)
      - WHO severity one-hot
      - Strict symptoms (excludes 5 extensions)
      - Percentage-based drug prevalence filter

    This is the ``baseline_antony_strict`` config; call
    ``build_antony_feature_matrix(strict_antony=False)`` for the thesis
    default ``baseline_antony``.

    Args:
        cur: Database cursor
        cohort_df: DataFrame with person_id, label, is_inpatient
        include_treatment: Whether to include Family E.
        use_hpo_symptoms: If True, use HPO symptom extraction via OMOP2OBO.

    Returns:
        Same as :func:`build_antony_feature_matrix`.
    """
    return build_antony_feature_matrix(
        cur, cohort_df,
        include_treatment=include_treatment,
        strict_antony=True,
        use_hpo_symptoms=use_hpo_symptoms,
    )

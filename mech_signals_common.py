"""
Shared utilities for mechanistic signal extraction notebooks.

This module provides:
- Unified data classes (ConceptSpec, IndicatorSpec, IndicatorKind)
- Concept resolution with caching
- Cohort loading and temp table preparation
- SQL template constants and compilation
- Validation utilities

Usage:
    from mech_signals_common import *
    
    # Define indicators
    class MyIndicator(Enum):
        TEST = IndicatorSpec(...)
    
    # Extract signals
    signals = build_signals(cur, cohort, [MyIndicator.TEST])
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Literal, Union, Tuple
import pandas as pd
import numpy as np

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

OmopTable = Literal[
    "CDMPHI.measurement",
    "CDMPHI.observation",
    "CDMPHI.condition_occurrence",
    "CDMPHI.procedure_occurrence",
    "CDMPHI.drug_exposure",
    "CDMPHI.visit_occurrence",
]


class ConceptSourceKind(str, Enum):
    """How to resolve concept IDs from the specification."""
    DIRECT_IDS = "DIRECT_IDS"          # Already standard concept_ids
    LOINC_CODES = "LOINC_CODES"        # Resolve via LOINC codes in CDMPHI.concept
    DESCENDANTS = "DESCENDANTS"        # Resolve via CDMPHI.concept_ancestor
    MIXED = "MIXED"                    # Combine DIRECT_IDS + LOINC_CODES
    NAME_PATTERN = "NAME_PATTERN"      # Match concept_name with LIKE patterns


@dataclass(frozen=True)
class ConceptSpec:
    """
    Defines how to resolve concept_ids for WHERE clauses.
    
    Examples:
        # Direct IDs
        ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(12345, 67890))
        
        # LOINC codes
        ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("1988-5", "30522-7"))
        
        # Descendants of ancestor concept
        ConceptSpec(kind=ConceptSourceKind.DESCENDANTS, ancestor_ids=(4098244,))
        
        # Name pattern matching
        ConceptSpec(kind=ConceptSourceKind.NAME_PATTERN, name_patterns=("%THROMBOSIS%",))
    """
    kind: ConceptSourceKind
    direct_ids: tuple[int, ...] = ()
    loinc_codes: tuple[str, ...] = ()
    ancestor_ids: tuple[int, ...] = ()       # Multiple ancestors supported
    name_patterns: tuple[str, ...] = ()      # For LIKE matching
    domain_id: Optional[str] = None          # Filter by domain (e.g., "Measurement", "Condition")
    
    def validate(self) -> None:
        if self.kind == ConceptSourceKind.DIRECT_IDS and not self.direct_ids:
            raise ValueError("DIRECT_IDS requires direct_ids")
        if self.kind == ConceptSourceKind.LOINC_CODES and not self.loinc_codes:
            raise ValueError("LOINC_CODES requires loinc_codes")
        if self.kind == ConceptSourceKind.DESCENDANTS and not self.ancestor_ids:
            raise ValueError("DESCENDANTS requires ancestor_ids")
        if self.kind == ConceptSourceKind.NAME_PATTERN and not self.name_patterns:
            raise ValueError("NAME_PATTERN requires name_patterns")
        if self.kind == ConceptSourceKind.MIXED and not (self.direct_ids or self.loinc_codes):
            raise ValueError("MIXED requires at least one of direct_ids or loinc_codes")


class IndicatorKind(str, Enum):
    """
    Query pattern types for different mechanistic signal extraction strategies.
    """
    # Measurement patterns
    MEAS_ANY_RECORDED = "MEAS_ANY_RECORDED"                       # Any measurement row
    MEAS_NUMERIC_ANY = "MEAS_NUMERIC_ANY"                         # Any numeric value
    MEAS_NUMERIC_THRESHOLD = "MEAS_NUMERIC_THRESHOLD"             # Numeric >= threshold
    MEAS_ELEVATED_OR_ANY = "MEAS_ELEVATED_OR_ANY"                 # > range_high or any value
    MEAS_PAIRED_SAME_DATE_NUMERIC = "MEAS_PAIRED_SAME_DATE_NUMERIC"  # e.g., NLR computation
    MEAS_REPEAT_AT_LEAST_N = "MEAS_REPEAT_AT_LEAST_N"             # >= N rows per person
    MEAS_ELEV_EPISODES_GAP = "MEAS_ELEV_EPISODES_GAP"             # Persistent elevation
    MEAS_PERSISTENT_POSITIVITY = "MEAS_PERSISTENT_POSITIVITY"     # positive run without intervening negative
    
    # Observation patterns
    OBS_ANY_RECORDED = "OBS_ANY_RECORDED"                         # Any observation row
    
    # Measurement + Observation union
    MEAS_OR_OBS_ANY = "MEAS_OR_OBS_ANY"                           # Either domain
    
    # Condition patterns
    COND_ANY_RECORDED = "COND_ANY_RECORDED"                       # Any condition row
    COND_EPISODES_GAP = "COND_EPISODES_GAP"                       # >= N episodes with gap
    COND_NAME_PATTERN = "COND_NAME_PATTERN"                       # concept_name LIKE pattern
    
    # Procedure patterns
    PROC_ANY_RECORDED = "PROC_ANY_RECORDED"                       # Any procedure row
    PROC_NAME_PATTERN = "PROC_NAME_PATTERN"                       # concept_name LIKE pattern
    
    # Drug patterns
    DRUG_ANY_RECORDED = "DRUG_ANY_RECORDED"                       # Any drug exposure
    DRUG_INGREDIENT_DESC = "DRUG_INGREDIENT_DESC"                 # RxNorm ingredient descendants
    DRUG_REPEAT_AT_LEAST_N = "DRUG_REPEAT_AT_LEAST_N"             # >= N exposures
    DRUG_EPISODES_GAP = "DRUG_EPISODES_GAP"                       # >= N episodes with gap
    
    # Visit patterns
    VISIT_SPECIALTY_NAME = "VISIT_SPECIALTY_NAME"                 # Provider specialty match


@dataclass(frozen=True)
class IndicatorSpec:
    """
    Fully defines a mechanistic signal indicator.
    
    Contains all optional parameters for flexibility across different query patterns.
    """
    label: str
    kind: IndicatorKind
    table: OmopTable
    
    # Concept resolution (most indicators)
    concepts_any_of: tuple[ConceptSpec, ...] = ()
    
    # Measurement-specific
    value_concept_ids: tuple[int, ...] = ()        # Filter by value_as_concept_id
    require_any_value: bool = False                # Require numeric OR concept value
    require_numeric_value: bool = False            # Require numeric value
    threshold_ge: Optional[float] = None           # Numeric threshold (>=)
    
    # Paired measurement (e.g., NLR = neutrophils / lymphocytes)
    left_concepts: Optional[ConceptSpec] = None
    right_concepts: Optional[ConceptSpec] = None
    
    # Episode logic
    episode_gap_days: Optional[int] = None
    min_episodes: Optional[int] = None
    
    # Repeat count
    min_count_per_person: Optional[int] = None

    # Persistent-positivity (MEAS_PERSISTENT_POSITIVITY): require at least one
    # positive in the unbroken run with days_from_window_start >= this value.
    # For w0_* windows this equals days from COVID index; for offset windows
    # (wM_N with M>0) the constraint is auto-satisfied because window_start
    # already starts >=M days post-index.
    min_post_acute_day: Optional[int] = None

    # Persistent-positivity: if True, the COVID index date is treated as the
    # first positive in the run. The handler then requires only one in-window
    # positive at days_from_window_start >= min_post_acute_day, in the very
    # first run (i.e. before any in-window negative). Captures "index pos -> ...
    # -> pos >=Nd later, no negative between" without needing two in-window
    # positives.
    index_is_anchor_positive: bool = False
    
    # Condition/Procedure name patterns
    concept_name_patterns: tuple[str, ...] = ()
    
    # Drug exposure
    rxnorm_ingredient_names: tuple[str, ...] = ()
    
    # Observation-specific
    obs_concept_ids: tuple[int, ...] = ()          # For MEAS_OR_OBS_ANY pattern
    
    # Visit-specific
    specialty_name_patterns: tuple[str, ...] = ()  # Provider specialty LIKE patterns
    
    # Source-value fallback patterns (LIKE patterns for *_source_value columns)
    source_value_patterns: tuple[str, ...] = ()    # e.g., ("%platelet%", "%plt%")
    
    # Special cases
    constant_zero: bool = False                    # Not feasible, always zero
    
    # Documentation
    notes: str = ""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_query(cur, query: str) -> pd.DataFrame:
    """Execute SQL query and return results as DataFrame."""
    cur.execute(query)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def _ints_to_sql_in(values: Sequence[int]) -> str:
    """Convert sequence of integers to SQL IN clause values."""
    return ", ".join(str(int(v)) for v in values)


def _strings_to_sql_in(values: Sequence[str]) -> str:
    """Convert sequence of strings to SQL IN clause values."""
    return ", ".join(f"'{v}'" for v in values)


def _date_column_for_table(table: OmopTable) -> str:
    """Get the date column name for a given OMOP table."""
    if table.endswith("measurement"):
        return "measurement_date"
    if table.endswith("observation"):
        return "observation_date"
    if table.endswith("condition_occurrence"):
        return "condition_start_date"
    if table.endswith("procedure_occurrence"):
        return "procedure_date"
    if table.endswith("drug_exposure"):
        return "drug_exposure_start_date"
    if table.endswith("visit_occurrence"):
        return "visit_start_date"
    raise ValueError(f"Unsupported table: {table}")


def _concept_column_for_table(table: OmopTable) -> str:
    """Get the concept column name for a given OMOP table."""
    if table.endswith("measurement"):
        return "measurement_concept_id"
    if table.endswith("observation"):
        return "observation_concept_id"
    if table.endswith("condition_occurrence"):
        return "condition_concept_id"
    if table.endswith("procedure_occurrence"):
        return "procedure_concept_id"
    if table.endswith("drug_exposure"):
        return "drug_concept_id"
    raise ValueError(f"Unsupported table: {table}")


def _source_value_column_for_table(table: OmopTable) -> str:
    """Get the source_value column name for a given OMOP table."""
    if table.endswith("measurement"):
        return "measurement_source_value"
    if table.endswith("observation"):
        return "observation_source_value"
    if table.endswith("condition_occurrence"):
        return "condition_source_value"
    if table.endswith("procedure_occurrence"):
        return "procedure_source_value"
    if table.endswith("drug_exposure"):
        return "drug_source_value"
    raise ValueError(f"Unsupported table for source_value: {table}")


def _build_source_value_filter(spec: IndicatorSpec, alias: str = "t") -> str:
    """Build source-value LIKE filter clause from spec.source_value_patterns.

    Returns an OR clause like:
        OR (LOWER(t.measurement_source_value) LIKE '%platelet%'
            OR LOWER(t.measurement_source_value) LIKE '%plt%')
    Returns empty string if no patterns defined.
    """
    if not spec.source_value_patterns:
        return ""
    sv_col = _source_value_column_for_table(spec.table)
    likes = " OR ".join(
        f"LOWER({alias}.{sv_col}) LIKE '{pat.lower()}'" for pat in spec.source_value_patterns
    )
    return f"OR ({likes})"


# =============================================================================
# CONCEPT RESOLUTION
# =============================================================================

_CONCEPT_ID_CACHE: dict[ConceptSpec, tuple[int, ...]] = {}


def resolve_concept_ids(cur, spec: ConceptSpec) -> tuple[int, ...]:
    """
    Resolve ConceptSpec to tuple of concept_ids with caching.
    
    Supports:
    - DIRECT_IDS: Return as-is
    - LOINC_CODES: Resolve via CDMPHI.concept
    - DESCENDANTS: Resolve via CDMPHI.concept_ancestor
    - MIXED: Combine DIRECT_IDS + LOINC_CODES
    - NAME_PATTERN: Match concept_name with LIKE (returns matching concept_ids)
    """
    spec.validate()
    
    # Check cache
    cached = _CONCEPT_ID_CACHE.get(spec)
    if cached is not None:
        return cached
    
    result: tuple[int, ...] = tuple()
    
    if spec.kind == ConceptSourceKind.DIRECT_IDS:
        result = tuple(int(x) for x in spec.direct_ids)
    
    elif spec.kind == ConceptSourceKind.LOINC_CODES:
        sql = f"""
        SELECT c.concept_id
        FROM CDMPHI.concept c
        WHERE c.vocabulary_id = 'LOINC'
          AND c.domain_id = 'Measurement'
          AND c.standard_concept = 'S'
          AND c.concept_code IN ({_strings_to_sql_in(spec.loinc_codes)})
        """
        df = run_query(cur, sql)
        result = tuple(int(x) for x in df["CONCEPT_ID"].tolist()) if not df.empty else tuple()
    
    elif spec.kind == ConceptSourceKind.DESCENDANTS:
        sql = f"""
        SELECT DISTINCT ca.descendant_concept_id AS concept_id
        FROM CDMPHI.concept_ancestor ca
        WHERE ca.ancestor_concept_id IN ({_ints_to_sql_in(spec.ancestor_ids)})
        """
        df = run_query(cur, sql)
        result = tuple(int(x) for x in df["CONCEPT_ID"].tolist()) if not df.empty else tuple()
    
    elif spec.kind == ConceptSourceKind.MIXED:
        direct = tuple(int(x) for x in spec.direct_ids) if spec.direct_ids else tuple()
        loinc_ids: tuple[int, ...] = tuple()
        
        if spec.loinc_codes:
            sql = f"""
            SELECT c.concept_id
            FROM CDMPHI.concept c
            WHERE c.vocabulary_id = 'LOINC'
              AND c.domain_id = 'Measurement'
              AND c.standard_concept = 'S'
              AND c.concept_code IN ({_strings_to_sql_in(spec.loinc_codes)})
            """
            df = run_query(cur, sql)
            loinc_ids = tuple(int(x) for x in df["CONCEPT_ID"].tolist()) if not df.empty else tuple()
        
        # Combine and deduplicate
        result = tuple(dict.fromkeys(list(direct) + list(loinc_ids)))
    
    elif spec.kind == ConceptSourceKind.NAME_PATTERN:
        like_clauses = [f"UPPER(c.concept_name) LIKE '{p.upper()}'" for p in spec.name_patterns]
        like_clause = " OR ".join(like_clauses) if like_clauses else "1=0"
        
        domain_filter = ""
        if spec.domain_id:
            domain_filter = f"AND c.domain_id = '{spec.domain_id}'"
        
        sql = f"""
        SELECT c.concept_id
        FROM CDMPHI.concept c
        WHERE c.standard_concept = 'S'
          {domain_filter}
          AND ({like_clause})
        """
        df = run_query(cur, sql)
        result = tuple(int(x) for x in df["CONCEPT_ID"].tolist()) if not df.empty else tuple()
    
    else:
        raise ValueError(f"Unsupported ConceptSourceKind: {spec.kind}")
    
    # Cache result (including empty tuples to avoid repeated DB hits)
    _CONCEPT_ID_CACHE[spec] = result
    return result


# =============================================================================
# COHORT LOADING & TEMP TABLE PREPARATION
# =============================================================================

def load_model_cohort_from_parquet(path: str) -> pd.DataFrame:
    """
    Load cohort from parquet with window-based schema validation.
    
    Required columns: person_id, postcovid_window_start, postcovid_window_end
    
    Returns:
        DataFrame with normalized types and validated window bounds
    """
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    
    required = {"person_id", "postcovid_window_start", "postcovid_window_end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Cohort parquet is missing required columns: {sorted(missing)}. "
            f"Found columns: {sorted(df.columns)}"
        )
    
    # Normalize types
    df = df.copy()
    df["person_id"] = df["person_id"].astype("int64")
    
    # Ensure date-compatible format
    df["postcovid_window_start"] = pd.to_datetime(df["postcovid_window_start"]).dt.date
    df["postcovid_window_end"] = pd.to_datetime(df["postcovid_window_end"]).dt.date
    
    # Validate window bounds
    bad = df[df["postcovid_window_start"] > df["postcovid_window_end"]]
    if len(bad) > 0:
        raise ValueError(
            f"Found {len(bad)} rows where postcovid_window_start > postcovid_window_end. "
            "Fix cohort generation before feature extraction."
        )
    
    return df[["person_id", "postcovid_window_start", "postcovid_window_end"]]


def prepare_temp_cohort(cur, cohort_df: pd.DataFrame, temp_name: str = "#cohort_temp") -> None:
    """
    Create and populate HANA local temporary table with cohort.
    
    Table schema: (person_id, postcovid_window_start, postcovid_window_end)
    
    Safe to rerun - drops temp table if it exists.
    """
    required = {"person_id", "postcovid_window_start", "postcovid_window_end"}
    if not required.issubset(cohort_df.columns):
        raise ValueError(f"cohort_df must contain columns: {sorted(required)}")
    
    tmp = cohort_df[["person_id", "postcovid_window_start", "postcovid_window_end"]].copy()
    tmp["person_id"] = tmp["person_id"].astype("int64")
    tmp["postcovid_window_start"] = pd.to_datetime(tmp["postcovid_window_start"]).dt.date
    tmp["postcovid_window_end"] = pd.to_datetime(tmp["postcovid_window_end"]).dt.date
    tmp = tmp.drop_duplicates(subset=["person_id"], keep="first")
    
    # Drop if exists
    try:
        cur.execute(f"DROP TABLE {temp_name}")
    except Exception:
        pass
    
    # Create temp table
    cur.execute(f"""
        CREATE LOCAL TEMPORARY TABLE {temp_name} (
            person_id BIGINT,
            postcovid_window_start DATE,
            postcovid_window_end DATE
        )
    """)
    
    # Bulk insert
    insert_sql = f"""
        INSERT INTO {temp_name} (person_id, postcovid_window_start, postcovid_window_end)
        VALUES (?, ?, ?)
    """
    data = [
        (int(r.person_id), r.postcovid_window_start, r.postcovid_window_end)
        for r in tmp.itertuples(index=False)
    ]
    cur.executemany(insert_sql, data)
    
    print(f"Prepared {temp_name} with {len(tmp):,} rows")


# =============================================================================
# SQL COMPILATION
# =============================================================================

def compile_indicator_sql(cur, indicator_enum, feature_name: str) -> Optional[str]:
    """
    Compile SQL query for an indicator that returns (person_id, <feature_name>).
    
    Args:
        cur: Database cursor (for concept resolution)
        indicator_enum: Enum member with .value = IndicatorSpec
        feature_name: Column name for the feature (e.g., "f_ind_viral_repeated_covid_dx")
    
    Returns:
        SQL query string or None if indicator has no matching patients
    """
    spec: IndicatorSpec = indicator_enum.value
    
    # Handle constant zero indicators
    if spec.constant_zero:
        return None
    
    # Get table-specific metadata
    date_col = _date_column_for_table(spec.table)
    
    # Build WHERE clause components
    date_filter = f"t.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end"
    
    # Dispatch to specific pattern handlers
    if spec.kind == IndicatorKind.MEAS_ANY_RECORDED:
        return _compile_meas_any_recorded(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.MEAS_NUMERIC_ANY:
        return _compile_meas_numeric_any(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.MEAS_NUMERIC_THRESHOLD:
        return _compile_meas_numeric_threshold(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.MEAS_ELEVATED_OR_ANY:
        return _compile_meas_elevated_or_any(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.MEAS_PAIRED_SAME_DATE_NUMERIC:
        return _compile_meas_paired_same_date_numeric(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.MEAS_REPEAT_AT_LEAST_N:
        return _compile_meas_repeat_at_least_n(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.MEAS_ELEV_EPISODES_GAP:
        return _compile_meas_elev_episodes_gap(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.MEAS_PERSISTENT_POSITIVITY:
        return _compile_meas_persistent_positivity(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.OBS_ANY_RECORDED:
        return _compile_obs_any_recorded(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.MEAS_OR_OBS_ANY:
        return _compile_meas_or_obs_any(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.COND_ANY_RECORDED:
        return _compile_cond_any_recorded(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.COND_EPISODES_GAP:
        return _compile_cond_episodes_gap(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.COND_NAME_PATTERN:
        return _compile_cond_name_pattern(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.PROC_ANY_RECORDED:
        return _compile_proc_any_recorded(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.PROC_NAME_PATTERN:
        return _compile_proc_name_pattern(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.DRUG_ANY_RECORDED:
        return _compile_drug_any_recorded(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.DRUG_INGREDIENT_DESC:
        return _compile_drug_ingredient_desc(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.DRUG_REPEAT_AT_LEAST_N:
        return _compile_drug_repeat_at_least_n(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.DRUG_EPISODES_GAP:
        return _compile_drug_episodes_gap(cur, spec, feature_name, date_col, date_filter)
    
    elif spec.kind == IndicatorKind.VISIT_SPECIALTY_NAME:
        return _compile_visit_specialty_name(cur, spec, feature_name, date_col, date_filter)
    
    else:
        raise ValueError(f"Unsupported IndicatorKind: {spec.kind}")


# =============================================================================
# SQL PATTERN IMPLEMENTATIONS
# =============================================================================

def _build_concept_filter(cur, spec: IndicatorSpec, alias: str = "t") -> str:
    """Build concept ID filter clause, with optional source_value fallback."""
    concept_col = _concept_column_for_table(spec.table) if spec.concepts_any_of else None
    sv_filter = _build_source_value_filter(spec, alias)

    if not spec.concepts_any_of and not sv_filter:
        return ""

    # Resolve concept IDs
    all_ids = []
    if spec.concepts_any_of:
        for cs in spec.concepts_any_of:
            all_ids.extend(resolve_concept_ids(cur, cs))

    if all_ids and sv_filter:
        # Concept IDs OR source_value patterns
        return f"AND ({alias}.{concept_col} IN ({_ints_to_sql_in(all_ids)}) {sv_filter})"
    elif all_ids:
        return f"AND {alias}.{concept_col} IN ({_ints_to_sql_in(all_ids)})"
    elif sv_filter:
        # Only source_value patterns (strip leading OR)
        sv_clause = sv_filter[3:]  # remove leading "OR "
        return f"AND ({sv_clause})"
    else:
        return "AND 1=0"


def _compile_meas_any_recorded(cur, spec, feature_name, date_col, date_filter):
    concept_filter = _build_concept_filter(cur, spec, alias="m")
    
    value_filter = ""
    if spec.require_any_value:
        value_filter = "AND (m.value_as_number IS NOT NULL OR m.value_as_concept_id IS NOT NULL)"
    if spec.require_numeric_value:
        value_filter = "AND m.value_as_number IS NOT NULL"
    if spec.value_concept_ids:
        value_filter += f" AND m.value_as_concept_id IN ({_ints_to_sql_in(spec.value_concept_ids)})"
    
    return f"""
    SELECT DISTINCT
        m.person_id,
        1 AS {feature_name}
    FROM CDMPHI.measurement m
    JOIN #cohort_temp c ON c.person_id = m.person_id
    WHERE m.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
        {concept_filter}
        {value_filter}
    """


def _compile_meas_numeric_any(cur, spec, feature_name, date_col, date_filter):
    concept_filter = _build_concept_filter(cur, spec, alias="m")
    return f"""
    SELECT DISTINCT
        m.person_id,
        1 AS {feature_name}
    FROM CDMPHI.measurement m
    JOIN #cohort_temp c ON c.person_id = m.person_id
    WHERE m.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
        {concept_filter}
        AND m.value_as_number IS NOT NULL
    """


def _compile_meas_numeric_threshold(cur, spec, feature_name, date_col, date_filter):
    if spec.threshold_ge is None:
        raise ValueError("MEAS_NUMERIC_THRESHOLD requires threshold_ge")
    
    concept_filter = _build_concept_filter(cur, spec, alias="m")
    return f"""
    SELECT DISTINCT
        m.person_id,
        1 AS {feature_name}
    FROM CDMPHI.measurement m
    JOIN #cohort_temp c ON c.person_id = m.person_id
    WHERE m.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
        {concept_filter}
        AND m.value_as_number IS NOT NULL
        AND m.value_as_number >= {float(spec.threshold_ge)}
    """


def _compile_meas_elevated_or_any(cur, spec, feature_name, date_col, date_filter):
    concept_filter = _build_concept_filter(cur, spec, alias="m")
    return f"""
    SELECT DISTINCT
        m.person_id,
        1 AS {feature_name}
    FROM CDMPHI.measurement m
    JOIN #cohort_temp c ON c.person_id = m.person_id
    WHERE m.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
        {concept_filter}
        AND (
            (m.value_as_number IS NOT NULL AND m.range_high IS NOT NULL AND m.value_as_number > m.range_high)
            OR (m.value_as_number IS NOT NULL AND m.range_high IS NULL)
            OR (m.value_as_concept_id IS NOT NULL)
        )
    """


def _compile_meas_paired_same_date_numeric(cur, spec, feature_name, date_col, date_filter):
    if spec.left_concepts is None or spec.right_concepts is None:
        raise ValueError("MEAS_PAIRED_SAME_DATE_NUMERIC requires left_concepts and right_concepts")
    
    left_ids = resolve_concept_ids(cur, spec.left_concepts)
    right_ids = resolve_concept_ids(cur, spec.right_concepts)
    
    if not left_ids or not right_ids:
        return None
    
    return f"""
    WITH left_meas AS (
        SELECT m.person_id, m.{date_col}, m.value_as_number AS left_val
        FROM CDMPHI.measurement m
        JOIN #cohort_temp c ON c.person_id = m.person_id
        WHERE m.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
            AND m.measurement_concept_id IN ({_ints_to_sql_in(left_ids)})
            AND m.value_as_number IS NOT NULL
    ),
    right_meas AS (
        SELECT m.person_id, m.{date_col}, m.value_as_number AS right_val
        FROM CDMPHI.measurement m
        JOIN #cohort_temp c ON c.person_id = m.person_id
        WHERE m.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
            AND m.measurement_concept_id IN ({_ints_to_sql_in(right_ids)})
            AND m.value_as_number IS NOT NULL
    )
    SELECT DISTINCT
        r.person_id,
        1 AS {feature_name}
    FROM right_meas r
    JOIN left_meas l ON r.person_id = l.person_id AND r.{date_col} = l.{date_col}
    WHERE l.left_val > 0
    """


def _compile_meas_repeat_at_least_n(cur, spec, feature_name, date_col, date_filter):
    if spec.min_count_per_person is None:
        raise ValueError("MEAS_REPEAT_AT_LEAST_N requires min_count_per_person")
    
    concept_filter = _build_concept_filter(cur, spec, alias="m")
    # Defect-2 fix: honour spec.value_concept_ids (previously dropped, so
    # REPEATED_POS_SARS_COV2_TEST counted repeat *testing*, not repeat
    # *positivity*).  Mirrors _compile_meas_any_recorded.
    value_filter = ""
    if spec.value_concept_ids:
        value_filter = f"AND m.value_as_concept_id IN ({_ints_to_sql_in(spec.value_concept_ids)})"
    return f"""
    SELECT
        m.person_id,
        1 AS {feature_name}
    FROM CDMPHI.measurement m
    JOIN #cohort_temp c ON c.person_id = m.person_id
    WHERE m.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
        {concept_filter}
        {value_filter}
    GROUP BY m.person_id
    HAVING COUNT(*) >= {int(spec.min_count_per_person)}
    """


def _compile_meas_elev_episodes_gap(cur, spec, feature_name, date_col, date_filter):
    if spec.episode_gap_days is None or spec.min_episodes is None:
        raise ValueError("MEAS_ELEV_EPISODES_GAP requires episode_gap_days and min_episodes")
    
    concept_filter = _build_concept_filter(cur, spec, alias="m")
    gap_days = int(spec.episode_gap_days)
    min_episodes = int(spec.min_episodes)
    
    return f"""
    WITH elev AS (
        SELECT
            m.person_id,
            m.{date_col},
            CASE
                WHEN (m.value_as_number IS NOT NULL AND m.range_high IS NOT NULL AND m.value_as_number > m.range_high) THEN 1
                WHEN (m.value_as_number IS NOT NULL AND m.range_high IS NULL) THEN 1
                ELSE 0
            END AS is_elev
        FROM CDMPHI.measurement m
        JOIN #cohort_temp c ON c.person_id = m.person_id
        WHERE m.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
            {concept_filter}
            AND m.value_as_number IS NOT NULL
    ),
    elev_only AS (
        SELECT person_id, {date_col}
        FROM elev
        WHERE is_elev = 1
    ),
    lagged AS (
        SELECT
            person_id,
            {date_col},
            LAG({date_col}) OVER (PARTITION BY person_id ORDER BY {date_col}) AS prev_date
        FROM elev_only
    ),
    episodes AS (
        SELECT
            person_id,
            SUM(
                CASE
                    WHEN prev_date IS NULL OR {date_col} >= ADD_DAYS(prev_date, {gap_days})
                    THEN 1 ELSE 0
                END
            ) AS n_episodes
        FROM lagged
        GROUP BY person_id
    )
    SELECT
        e.person_id,
        1 AS {feature_name}
    FROM episodes e
    WHERE e.n_episodes >= {min_episodes}
    """


def _compile_meas_persistent_positivity(cur, spec, feature_name, date_col, date_filter):
    """
    Persistent SARS-CoV-2 positivity: >=2 positive results with NO intervening
    negative for the same person within the postcovid window.

    Operationalizes the viral-reservoir hypothesis (Proal 2023, Chen 2023) as
    "positive -> positive without an intervening negative" -- distinct from
    reinfection (positive -> negative -> positive) which would NOT qualify.

    Key handler-specific parameter semantics:
      - ``value_concept_ids`` MUST contain BOTH 45884084 (Positive) and
        45877985 (Negative). This is a *reinterpretation* of the field for
        this kind only -- elsewhere it is a positive-filter list. Negatives
        are required so the handler can detect them and break runs.
      - ``min_post_acute_day`` (optional): require at least one positive in
        the qualifying run at days_from_window_start >= N. For w0_* windows
        this equals days from COVID index. For offset windows (wM_N with
        M>0) the constraint is auto-satisfied because window_start is
        already M days post-index.

    Result-value capture (loosened):
      The handler interprets a row as positive/negative by inspecting
      ``value_as_concept_id`` first and falling back to
      ``value_source_value`` / ``measurement_source_value`` text patterns
      (case-insensitive). This is essential at MSH (CDMPHI) where PCR
      results are frequently stored as text ("Detected" / "Not Detected"
      / "Positive" / "Negative") in ``value_source_value`` without
      ``value_as_concept_id`` being populated. CDMPHI does NOT have a
      ``value_as_string`` column. The "no intervening negative" invariant
      is preserved -- text-detected negatives still break runs. Negatives
      are matched first so "not detected" is never misclassified as
      "detected".
    """
    if not spec.concepts_any_of:
        raise ValueError("MEAS_PERSISTENT_POSITIVITY requires concepts_any_of (PCR/antigen concept IDs)")
    # NOTE: spec.value_concept_ids is no longer used by this handler. The
    # Pos/Neg value-concept allowlist is hardcoded below based on a verified
    # MSH DB-wide audit (see check_value_concept_dist.py). The field is kept
    # in the spec for backward compat but ignored here.

    # Resolve concept IDs (PCR/antigen tests)
    all_ids = []
    for cs in spec.concepts_any_of:
        all_ids.extend(resolve_concept_ids(cur, cs))
    if not all_ids:
        return None
    concept_in = _ints_to_sql_in(all_ids)

    if spec.min_post_acute_day is not None:
        anchor = int(spec.min_post_acute_day)
        min_post_acute_clause = f"AND run_max_days_from_start >= {anchor}"
    else:
        min_post_acute_clause = ""

    # Index-as-anchor mode: COVID index date already counts as a confirmed
    # positive (cohort entry criterion). Therefore we only need ONE in-window
    # positive in the very first run (run_id = 0, i.e. before any in-window
    # negative) at days_from_window_start >= min_post_acute_day. This
    # naturally enforces "index pos -> ... -> pos >=Nd later, no negative
    # between".
    if spec.index_is_anchor_positive:
        n_pos_required = 1
        run_id_clause = "AND run_id = 0"
    else:
        n_pos_required = 2
        run_id_clause = ""

    # Text patterns used as a fallback when value_as_concept_id is NULL.
    # Negatives are matched before positives so "not detected" wins over
    # "detected". Patterns are LIKE-style and applied to LOWER() of both
    # value_source_value and measurement_source_value.
    #
    # Long-word patterns wrap with '%' to tolerate trailing qualifiers
    # ("Positive (high)"). Short stems ("pos", "neg") use exact match to
    # avoid incidental substring hits against words like "supposed" or
    # "negate". value_source_value for SARS-CoV-2 PCR/antigen at MSH is
    # short and dominated by the result word.
    neg_patterns = (
        "%negative%", "neg",
        "%not detected%", "%not-detected%",
        "%nondetected%", "%non-detected%",
        "%not reactive%", "%non-reactive%",
    )
    pos_patterns = (
        "%positive%", "pos",
        "%detected%", "%reactive%",
    )
    # 36715206 'Presumptive positive' is also stored as text; "%positive%" would
    # otherwise catch it.  Exclude it from the text-based positive call so the
    # persistence signal counts only confirmed positives (decision 2026-07-17).
    presumptive_patterns = (
        "%presumptive%",
    )

    def _any_like(col_expr: str, pats: tuple) -> str:
        return "(" + " OR ".join(f"LOWER({col_expr}) LIKE '{p}'" for p in pats) + ")"

    neg_text = (
        f"({_any_like('m.value_source_value', neg_patterns)} "
        f"OR {_any_like('m.measurement_source_value', neg_patterns)})"
    )
    pos_text = (
        f"({_any_like('m.value_source_value', pos_patterns)} "
        f"OR {_any_like('m.measurement_source_value', pos_patterns)})"
    )
    presumptive_text = (
        f"({_any_like('m.value_source_value', presumptive_patterns)} "
        f"OR {_any_like('m.measurement_source_value', presumptive_patterns)})"
    )

    # MSH (CDMPHI) value_as_concept_id distribution for SARS-CoV-2 NAA/antigen
    # tests (verified in a DB-wide audit of the MSHS extract):
    #   NEG: 45880296 'Not detected' (1.53M), 45878583 'Negative' (2.4K),
    #        1261264 'Neg' (605)
    #   POS: 45877985 'Detected' (110K), 45884084 'Positive' (467),
    #        36715206 'Presumptive positive' (1.3K)
    # NOTE: 45877985 means 'Detected' = POSITIVE in MSH usage. Earlier code
    # in this handler treated it as Negative; that was wrong and is the
    # reason persistent-positivity prevalence appeared to be ~0.
    neg_value_ids = (45880296, 45878583, 1261264)
    # 36715206 'Presumptive positive' EXCLUDED (decision 2026-07-17): an
    # unconfirmed positive is not evidence the virus failed to clear.
    pos_value_ids = (45877985, 45884084)
    neg_value_in = ",".join(str(i) for i in neg_value_ids)
    pos_value_in = ",".join(str(i) for i in pos_value_ids)
    all_value_in = ",".join(str(i) for i in (*neg_value_ids, *pos_value_ids))

    return f"""
    WITH tests AS (
        SELECT
            m.person_id,
            m.{date_col} AS test_date,
            CASE
                WHEN m.value_as_concept_id IN ({neg_value_in}) THEN 0
                WHEN m.value_as_concept_id IN ({pos_value_in}) THEN 1
                WHEN m.value_as_concept_id IS NULL AND {neg_text} THEN 0
                WHEN m.value_as_concept_id IS NULL AND {pos_text} AND NOT {presumptive_text} THEN 1
            END AS is_positive,
            DAYS_BETWEEN(c.postcovid_window_start, m.{date_col}) AS days_from_start
        FROM CDMPHI.measurement m
        JOIN #cohort_temp c ON c.person_id = m.person_id
        WHERE m.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
            AND m.measurement_concept_id IN ({concept_in})
            AND (
                m.value_as_concept_id IN ({all_value_in})
                OR (m.value_as_concept_id IS NULL AND ({pos_text} OR {neg_text}))
            )
    ),
    ordered AS (
        SELECT
            person_id, test_date, is_positive, days_from_start,
            SUM(CASE WHEN is_positive = 0 THEN 1 ELSE 0 END)
                OVER (PARTITION BY person_id ORDER BY test_date
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS neg_count_so_far
        FROM tests
    ),
    positive_runs AS (
        SELECT
            person_id,
            neg_count_so_far AS run_id,
            COUNT(*) AS n_pos_in_run,
            MAX(days_from_start) AS run_max_days_from_start
        FROM ordered
        WHERE is_positive = 1
        GROUP BY person_id, neg_count_so_far
    )
    SELECT DISTINCT
        person_id,
        1 AS {feature_name}
    FROM positive_runs
    WHERE n_pos_in_run >= {n_pos_required}
        {run_id_clause}
        {min_post_acute_clause}
    """


def _compile_obs_any_recorded(cur, spec, feature_name, date_col, date_filter):
    concept_filter = _build_concept_filter(cur, spec, alias="o")
    
    value_filter = ""
    if spec.require_any_value:
        value_filter = "AND (o.value_as_number IS NOT NULL OR o.value_as_concept_id IS NOT NULL OR o.value_as_string IS NOT NULL)"
    
    return f"""
    SELECT DISTINCT
        o.person_id,
        1 AS {feature_name}
    FROM CDMPHI.observation o
    JOIN #cohort_temp c ON c.person_id = o.person_id
    WHERE o.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
        {concept_filter}
        {value_filter}
    """


def _compile_meas_or_obs_any(cur, spec, feature_name, date_col, date_filter):
    meas_concept_filter = _build_concept_filter(cur, spec, alias="m")
    
    obs_filter = "AND 1=0"
    if spec.obs_concept_ids:
        obs_filter = f"AND o.observation_concept_id IN ({_ints_to_sql_in(spec.obs_concept_ids)})"
    
    return f"""
    WITH meas_pts AS (
        SELECT DISTINCT m.person_id
        FROM CDMPHI.measurement m
        JOIN #cohort_temp c ON c.person_id = m.person_id
        WHERE m.measurement_date BETWEEN c.postcovid_window_start AND c.postcovid_window_end
            {meas_concept_filter}
            AND (m.value_as_number IS NOT NULL OR m.value_as_concept_id IS NOT NULL)
    ),
    obs_pts AS (
        SELECT DISTINCT o.person_id
        FROM CDMPHI.observation o
        JOIN #cohort_temp c ON c.person_id = o.person_id
        WHERE o.observation_date BETWEEN c.postcovid_window_start AND c.postcovid_window_end
            {obs_filter}
            AND (o.value_as_number IS NOT NULL OR o.value_as_concept_id IS NOT NULL OR o.value_as_string IS NOT NULL)
    )
    SELECT DISTINCT person_id, 1 AS {feature_name}
    FROM (
        SELECT person_id FROM meas_pts
        UNION
        SELECT person_id FROM obs_pts
    )
    """


def _compile_cond_any_recorded(cur, spec, feature_name, date_col, date_filter):
    concept_filter = _build_concept_filter(cur, spec, alias="co")
    return f"""
    SELECT DISTINCT
        co.person_id,
        1 AS {feature_name}
    FROM CDMPHI.condition_occurrence co
    JOIN #cohort_temp c ON c.person_id = co.person_id
    WHERE co.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
        {concept_filter}
    """


def _compile_cond_episodes_gap(cur, spec, feature_name, date_col, date_filter):
    if spec.episode_gap_days is None or spec.min_episodes is None:
        raise ValueError("COND_EPISODES_GAP requires episode_gap_days and min_episodes")
    
    concept_filter = _build_concept_filter(cur, spec, alias="co")
    gap_days = int(spec.episode_gap_days)
    min_episodes = int(spec.min_episodes)
    
    return f"""
    WITH lagged AS (
        SELECT
            co.person_id,
            co.{date_col},
            LAG(co.{date_col}) OVER (PARTITION BY co.person_id ORDER BY co.{date_col}) AS prev_date
        FROM CDMPHI.condition_occurrence co
        JOIN #cohort_temp c ON c.person_id = co.person_id
        WHERE co.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
            {concept_filter}
    ),
    episodes AS (
        SELECT
            person_id,
            SUM(
                CASE
                    WHEN prev_date IS NULL OR {date_col} >= ADD_DAYS(prev_date, {gap_days})
                    THEN 1 ELSE 0
                END
            ) AS n_episodes
        FROM lagged
        GROUP BY person_id
    )
    SELECT
        e.person_id,
        1 AS {feature_name}
    FROM episodes e
    WHERE e.n_episodes >= {min_episodes}
    """


def _compile_cond_name_pattern(cur, spec, feature_name, date_col, date_filter):
    if not spec.concept_name_patterns:
        raise ValueError("COND_NAME_PATTERN requires concept_name_patterns")
    
    like_clauses = [f"UPPER(c.concept_name) LIKE '{p.upper()}'" for p in spec.concept_name_patterns]
    like_clause = " OR ".join(like_clauses)
    
    return f"""
    WITH concepts AS (
        SELECT c.concept_id
        FROM CDMPHI.concept c
        WHERE c.domain_id = 'Condition'
            AND c.standard_concept = 'S'
            AND ({like_clause})
    )
    SELECT DISTINCT
        co.person_id,
        1 AS {feature_name}
    FROM CDMPHI.condition_occurrence co
    JOIN #cohort_temp c ON c.person_id = co.person_id
    WHERE co.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
        AND co.condition_concept_id IN (SELECT concept_id FROM concepts)
    """


def _compile_proc_any_recorded(cur, spec, feature_name, date_col, date_filter):
    concept_filter = _build_concept_filter(cur, spec, alias="po")
    return f"""
    SELECT DISTINCT
        po.person_id,
        1 AS {feature_name}
    FROM CDMPHI.procedure_occurrence po
    JOIN #cohort_temp c ON c.person_id = po.person_id
    WHERE po.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
        {concept_filter}
    """


def _compile_proc_name_pattern(cur, spec, feature_name, date_col, date_filter):
    if not spec.concept_name_patterns:
        raise ValueError("PROC_NAME_PATTERN requires concept_name_patterns")
    
    like_clauses = [f"UPPER(c.concept_name) LIKE '{p.upper()}'" for p in spec.concept_name_patterns]
    like_clause = " OR ".join(like_clauses)
    
    return f"""
    WITH concepts AS (
        SELECT c.concept_id
        FROM CDMPHI.concept c
        WHERE c.domain_id = 'Procedure'
            AND c.standard_concept = 'S'
            AND ({like_clause})
    )
    SELECT DISTINCT
        po.person_id,
        1 AS {feature_name}
    FROM CDMPHI.procedure_occurrence po
    JOIN #cohort_temp c ON c.person_id = po.person_id
    WHERE po.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
        AND po.procedure_concept_id IN (SELECT concept_id FROM concepts)
    """


def _compile_drug_any_recorded(cur, spec, feature_name, date_col, date_filter):
    concept_filter = _build_concept_filter(cur, spec, alias="de")
    return f"""
    SELECT DISTINCT
        de.person_id,
        1 AS {feature_name}
    FROM CDMPHI.drug_exposure de
    JOIN #cohort_temp c ON c.person_id = de.person_id
    WHERE de.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
        {concept_filter}
    """


def _compile_drug_ingredient_desc(cur, spec, feature_name, date_col, date_filter):
    if not spec.rxnorm_ingredient_names:
        raise ValueError("DRUG_INGREDIENT_DESC requires rxnorm_ingredient_names")
    
    ingr_in = _strings_to_sql_in([n.upper() for n in spec.rxnorm_ingredient_names])
    
    # Build optional source_value fallback
    sv_union = ""
    if spec.source_value_patterns:
        sv_col = _source_value_column_for_table(spec.table)
        likes = " OR ".join(
            f"LOWER(de2.{sv_col}) LIKE '{pat.lower()}'" for pat in spec.source_value_patterns
        )
        sv_union = f"""
    UNION
    SELECT DISTINCT
        de2.person_id,
        1 AS {feature_name}
    FROM CDMPHI.drug_exposure de2
    JOIN #cohort_temp c ON c.person_id = de2.person_id
    WHERE de2.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
        AND ({likes})
    """

    return f"""
    WITH ingredients AS (
        SELECT c.concept_id
        FROM CDMPHI.concept c
        WHERE c.vocabulary_id = 'RxNorm'
            AND c.domain_id = 'Drug'
            AND c.standard_concept = 'S'
            AND c.concept_class_id = 'Ingredient'
            AND UPPER(c.concept_name) IN ({ingr_in})
    ),
    drug_descendants AS (
        SELECT DISTINCT ca.descendant_concept_id AS drug_concept_id
        FROM CDMPHI.concept_ancestor ca
        WHERE ca.ancestor_concept_id IN (SELECT concept_id FROM ingredients)
    )
    SELECT DISTINCT
        de.person_id,
        1 AS {feature_name}
    FROM CDMPHI.drug_exposure de
    JOIN #cohort_temp c ON c.person_id = de.person_id
    WHERE de.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
        AND de.drug_concept_id IN (SELECT drug_concept_id FROM drug_descendants)
    {sv_union}
    """


def _compile_drug_repeat_at_least_n(cur, spec, feature_name, date_col, date_filter):
    if spec.min_count_per_person is None:
        raise ValueError("DRUG_REPEAT_AT_LEAST_N requires min_count_per_person")
    
    if not spec.rxnorm_ingredient_names:
        raise ValueError("DRUG_REPEAT_AT_LEAST_N requires rxnorm_ingredient_names")
    
    ingr_in = _strings_to_sql_in([n.upper() for n in spec.rxnorm_ingredient_names])
    min_count = int(spec.min_count_per_person)
    
    # Build optional source_value fallback for the drug_matches CTE
    sv_clause = ""
    if spec.source_value_patterns:
        sv_col = _source_value_column_for_table(spec.table)
        likes = " OR ".join(
            f"LOWER(de2.{sv_col}) LIKE '{pat.lower()}'" for pat in spec.source_value_patterns
        )
        sv_clause = f"""
    UNION ALL
    SELECT de2.person_id, de2.{date_col} AS drug_date
    FROM CDMPHI.drug_exposure de2
    JOIN #cohort_temp c ON c.person_id = de2.person_id
    WHERE de2.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
        AND ({likes})
    """

    return f"""
    WITH ingredients AS (
        SELECT c.concept_id
        FROM CDMPHI.concept c
        WHERE c.vocabulary_id = 'RxNorm'
            AND c.domain_id = 'Drug'
            AND c.standard_concept = 'S'
            AND c.concept_class_id = 'Ingredient'
            AND UPPER(c.concept_name) IN ({ingr_in})
    ),
    drug_descendants AS (
        SELECT DISTINCT ca.descendant_concept_id AS drug_concept_id
        FROM CDMPHI.concept_ancestor ca
        WHERE ca.ancestor_concept_id IN (SELECT concept_id FROM ingredients)
    ),
    drug_matches AS (
        SELECT de.person_id, de.{date_col} AS drug_date
        FROM CDMPHI.drug_exposure de
        JOIN #cohort_temp c ON c.person_id = de.person_id
        WHERE de.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
            AND de.drug_concept_id IN (SELECT drug_concept_id FROM drug_descendants)
        {sv_clause}
    )
    SELECT
        dm.person_id,
        1 AS {feature_name}
    FROM drug_matches dm
    GROUP BY dm.person_id
    HAVING COUNT(*) >= {min_count}
    """


def _compile_drug_episodes_gap(cur, spec, feature_name, date_col, date_filter):
    if spec.episode_gap_days is None or spec.min_episodes is None:
        raise ValueError("DRUG_EPISODES_GAP requires episode_gap_days and min_episodes")
    
    if not spec.rxnorm_ingredient_names:
        raise ValueError("DRUG_EPISODES_GAP requires rxnorm_ingredient_names")
    
    ingr_in = _strings_to_sql_in([n.upper() for n in spec.rxnorm_ingredient_names])
    gap_days = int(spec.episode_gap_days)
    min_episodes = int(spec.min_episodes)
    
    return f"""
    WITH ingredients AS (
        SELECT c.concept_id
        FROM CDMPHI.concept c
        WHERE c.vocabulary_id = 'RxNorm'
            AND c.domain_id = 'Drug'
            AND c.standard_concept = 'S'
            AND c.concept_class_id = 'Ingredient'
            AND UPPER(c.concept_name) IN ({ingr_in})
    ),
    drug_descendants AS (
        SELECT DISTINCT ca.descendant_concept_id AS drug_concept_id
        FROM CDMPHI.concept_ancestor ca
        WHERE ca.ancestor_concept_id IN (SELECT concept_id FROM ingredients)
    ),
    lagged AS (
        SELECT
            de.person_id,
            de.{date_col},
            LAG(de.{date_col}) OVER (PARTITION BY de.person_id ORDER BY de.{date_col}) AS prev_date
        FROM CDMPHI.drug_exposure de
        JOIN #cohort_temp c ON c.person_id = de.person_id
        WHERE de.{date_col} BETWEEN c.postcovid_window_start AND c.postcovid_window_end
            AND de.drug_concept_id IN (SELECT drug_concept_id FROM drug_descendants)
    ),
    episodes AS (
        SELECT
            person_id,
            SUM(
                CASE
                    WHEN prev_date IS NULL OR {date_col} >= ADD_DAYS(prev_date, {gap_days})
                    THEN 1 ELSE 0
                END
            ) AS n_episodes
        FROM lagged
        GROUP BY person_id
    )
    SELECT
        e.person_id,
        1 AS {feature_name}
    FROM episodes e
    WHERE e.n_episodes >= {min_episodes}
    """


def _compile_visit_specialty_name(cur, spec, feature_name, date_col, date_filter):
    if not spec.specialty_name_patterns:
        raise ValueError("VISIT_SPECIALTY_NAME requires specialty_name_patterns")
    
    like_clauses = [f"LOWER(c.concept_name) LIKE '{p.lower()}'" for p in spec.specialty_name_patterns]
    like_clause = " OR ".join(like_clauses)

    sv_clauses = [f"LOWER(p.specialty_source_value) LIKE '{p.lower()}'" for p in spec.specialty_name_patterns]
    sv_clause = " OR ".join(sv_clauses)

    cs_clauses = [f"LOWER(cs.care_site_name) LIKE '{p.lower()}'" for p in spec.specialty_name_patterns]
    cs_clause = " OR ".join(cs_clauses)
    
    return f"""
    WITH specialist_providers AS (
        SELECT p.provider_id
        FROM CDMPHI.provider p
        LEFT JOIN CDMPHI.concept c ON p.specialty_concept_id = c.concept_id
        WHERE ({like_clause})
           OR ({sv_clause})
    ),
    specialist_sites AS (
        SELECT cs.care_site_id
        FROM CDMPHI.care_site cs
        WHERE {cs_clause}
    )
    SELECT DISTINCT
        vo.person_id,
        1 AS {feature_name}
    FROM CDMPHI.visit_occurrence vo
    JOIN #cohort_temp ct ON ct.person_id = vo.person_id
    WHERE vo.{date_col} BETWEEN ct.postcovid_window_start AND ct.postcovid_window_end
        AND (vo.provider_id IN (SELECT provider_id FROM specialist_providers)
             OR vo.care_site_id IN (SELECT care_site_id FROM specialist_sites))
    """


# =============================================================================
# SIGNAL EXTRACTION
# =============================================================================

def build_signals(cur, full_cohort: pd.DataFrame, indicators: Sequence, mechanism_prefix: str) -> pd.DataFrame:
    """
    Extract mechanistic signals for a cohort.
    
    Args:
        cur: Database cursor
        full_cohort: DataFrame with person_id, postcovid_window_start, postcovid_window_end
        indicators: List of indicator enum members
        mechanism_prefix: Short name for mechanism (e.g., "viral", "immuno", "endo")
    
    Returns:
        DataFrame with person_id and binary feature columns (f_ind_{mechanism}_{name})
    """
    # Initialize signals table
    signals = full_cohort[["person_id"]].drop_duplicates().copy()
    signals["person_id"] = signals["person_id"].astype("int64")
    
    # Process each indicator
    for ind in indicators:
        feat_col = f"f_ind_{mechanism_prefix}_{ind.name.lower()}"
        
        # Handle constant zero indicators
        if ind.value.constant_zero:
            signals[feat_col] = 0
            continue
        
        print(f"  Extracting: {ind.name}")
        
        try:
            sql = compile_indicator_sql(cur, ind, feat_col)
            
            if sql:
                df_hits = run_query(cur, sql)
                df_hits.columns = [c.lower() for c in df_hits.columns]
                
                if not df_hits.empty:
                    signals = signals.merge(df_hits[["person_id", feat_col]], on="person_id", how="left")
                else:
                    signals[feat_col] = 0
            else:
                signals[feat_col] = 0
        
        except Exception as e:
            print(f"    Error on {ind.name}: {e}")
            signals[feat_col] = 0
    
    # Fill NaNs and cast to int
    feat_cols = [c for c in signals.columns if c.startswith("f_ind_")]
    signals[feat_cols] = signals[feat_cols].fillna(0).astype("int8")
    
    return signals


# =============================================================================
# VALIDATION
# =============================================================================

def validate_signals_against_cohort(
    signals: pd.DataFrame,
    cohort: pd.DataFrame,
    signals_name: str = "signals",
) -> None:
    """
    Validate signals DataFrame structure and alignment with cohort.
    
    Checks:
    - Required person_id column
    - No duplicate person_ids
    - person_id sets match exactly
    - Feature columns follow naming convention (f_*)
    - All features are binary (0/1) with no NaNs
    - Integer dtypes
    
    Raises ValueError on any validation failure.
    """
    # Required column
    if "person_id" not in signals.columns:
        raise ValueError(f"{signals_name}: missing required column 'person_id'")
    
    # Duplicate checks
    n_dup_signals = int(signals["person_id"].duplicated().sum())
    n_dup_cohort = int(cohort["person_id"].duplicated().sum())
    
    if n_dup_signals:
        raise ValueError(f"{signals_name}: has {n_dup_signals} duplicate person_id rows")
    if n_dup_cohort:
        raise ValueError(f"cohort: has {n_dup_cohort} duplicate person_id rows")
    
    # Set equality
    cohort_ids = set(cohort["person_id"].astype("int64").tolist())
    signal_ids = set(signals["person_id"].astype("int64").tolist())
    
    missing_in_signals = cohort_ids - signal_ids
    extra_in_signals = signal_ids - cohort_ids
    
    if missing_in_signals:
        sample = list(sorted(missing_in_signals))[:10]
        raise ValueError(
            f"{signals_name}: missing {len(missing_in_signals)} cohort person_ids. Sample: {sample}"
        )
    if extra_in_signals:
        sample = list(sorted(extra_in_signals))[:10]
        raise ValueError(
            f"{signals_name}: has {len(extra_in_signals)} extra person_ids not in cohort. Sample: {sample}"
        )
    
    # Feature columns
    feat_cols = [c for c in signals.columns if c != "person_id"]
    if not feat_cols:
        raise ValueError(f"{signals_name}: no feature columns found (only person_id present)")
    
    # Naming convention
    bad_prefix = [c for c in feat_cols if not c.startswith("f_")]
    if bad_prefix:
        sample = bad_prefix[:10]
        raise ValueError(
            f"{signals_name}: {len(bad_prefix)} feature columns do not start with 'f_'. Sample: {sample}"
        )
    
    # Binary values and NaNs
    for c in feat_cols:
        if signals[c].isna().any():
            n_na = int(signals[c].isna().sum())
            raise ValueError(f"{signals_name}: column {c} contains {n_na} NaN values (should be 0/1)")
        
        vals = set(pd.unique(signals[c]))
        if not vals.issubset({0, 1}):
            sample = list(sorted(vals))[:10]
            raise ValueError(f"{signals_name}: column {c} has non-binary values {sample}")
    
    # Dtypes
    for c in feat_cols:
        if not pd.api.types.is_integer_dtype(signals[c]):
            raise ValueError(f"{signals_name}: column {c} is not integer dtype (found {signals[c].dtype})")
    
    print(f"[OK] {signals_name}:")
    print(f"  rows: {len(signals):,} (matches cohort)")
    print(f"  features: {len(feat_cols):,}")
    print(f"  all features binary, no NaNs, no duplicate person_id")


def compare_signals_structure(a: pd.DataFrame, b: pd.DataFrame, a_name="A", b_name="B") -> None:
    """
    Compare structure of two signals DataFrames.
    
    Validates both have:
    - person_id column
    - Matching person_id sets
    - Integer feature dtypes
    """
    # Check person_id
    for name, df in [(a_name, a), (b_name, b)]:
        if "person_id" not in df.columns:
            raise ValueError(f"{name} missing person_id")
    
    # Row-level alignment
    if set(a["person_id"]) != set(b["person_id"]):
        raise ValueError(f"{a_name} and {b_name} have different person_id sets")
    
    # Feature dtype check
    for name, df in [(a_name, a), (b_name, b)]:
        feat_cols = [c for c in df.columns if c != "person_id"]
        non_int = [c for c in feat_cols if not pd.api.types.is_integer_dtype(df[c])]
        if non_int:
            raise ValueError(f"{name} has non-integer feature columns. Sample: {non_int[:10]}")
    
    print(f"[OK] {a_name} and {b_name} match on person_id set and integer feature dtypes.")


def display_prevalence(signals: pd.DataFrame, top_n: int = 15) -> None:
    """Display prevalence summary for features."""
    feat_cols = [c for c in signals.columns if c != "person_id"]
    prevalence_pct = (signals[feat_cols].mean(axis=0) * 100).sort_values(ascending=False)
    
    print(f"\nTop {top_n} Most Prevalent Features:")
    print(prevalence_pct.head(top_n))
    
    print(f"\nBottom {top_n} Least Prevalent Features:")
    print(prevalence_pct.tail(top_n))
    
    return prevalence_pct

"""
Enhanced mechanistic signal extraction for Long COVID / PASC prediction.

Extends the existing binary indicator framework with:
  1. Multi-window temporal extraction (0-30d, 31-90d, 91-180d post-index)
  2. Numeric/trajectory lab features (peak, median, last, abnormal count, slope)
  3. CBC-derived inflammatory indices (NLR, PLR, SII)
  4. Composite features (complement+coag, organ-injury, GI-persistence)
  5. Enhanced indicator enums (upgraded viral, immuno, endo + exploratory)

Does NOT modify any existing modules.  Imports from mech_signals_common and
indicator_definitions for reuse.

Usage (from run_enhanced_mechsig.py):
    from enhanced_mech_signals import (
        make_window_cohort,
        extract_binary_signals_for_window,
        extract_numeric_labs,
        compute_cross_window_trajectories,
        extract_cbc_indices,
        build_composite_features,
        build_enhanced_model_configs,
        NUMERIC_LAB_SPECS,
        ENH_VIRAL_INDICATOR_LIST,
        ENH_IMMUNO_INDICATOR_LIST,
        ENH_ENDO_INDICATOR_LIST,
        ENH_EXPLORATORY_INDICATOR_LIST,
        MECH_WINDOWS,
    )
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Dict, List, Tuple

from mech_signals_common import (
    ConceptSourceKind,
    ConceptSpec,
    IndicatorKind,
    IndicatorSpec,
    prepare_temp_cohort,
    build_signals,
    run_query,
    resolve_concept_ids,
    _ints_to_sql_in,
    _strings_to_sql_in,
)
from indicator_definitions import (
    VIRAL_INDICATOR_LIST,
    IMMUNO_INDICATOR_LIST,
    ENDO_INDICATOR_LIST,
    COVID_PCR_POSITIVE_VALUES,
)

# =============================================================================
# CONSTANTS
# =============================================================================

from omop_config import CDM_SCHEMA as CDM  # OMOP schema name; set OMOP_CDM_SCHEMA to override

# Default mechanistic windows (days post covid_index_date)
MECH_WINDOWS = [(0, 30), (31, 90), (91, 180)]

# PASC-directed temporal shortlist from the exploratory prevalence / persistence analysis.
DEFAULT_TEMPORAL_BINARY_FEATURES = (
    "f_ind_enh_viral_resp_dx_any",
    # v2 purification moved CRP/ESR/D-dimer elevation into the numeric-lab stream;
    # point at the f_lab_*_abnormal_any base names (was: elev_crp / elev_esr /
    # coag_activation_any / coag_ddimer_elev_proxy binary indicators, now removed).
    "f_lab_crp_abnormal_any",
    "f_lab_esr_abnormal_any",
    "f_lab_ddimer_abnormal_any",
    "f_ind_endo_repeated_ddimer_testing",
    "f_ind_enh_viral_gi_dx_any",
    # complement now via C3/C4 ordered (was: f_ind_endo_complement_activity, removed)
    "f_ind_enh_immuno_complement_c3_ordered",
    "f_ind_enh_immuno_complement_c4_ordered",
    "f_ind_immuno_any_il6",
    "f_ind_enh_endo_pots_orthostatic",
)

DEFAULT_TEMPORAL_LAB_DIRECTIONS = {
    "crp": "up",
    "esr": "up",
    "ddimer": "up",
    "ferritin": "up",
    "ldh": "up",
    "troponin": "up",
    "nt_probnp": "up",
    "platelets": "up",
    "albumin": "down",
}


def window_label(start: int, end: int) -> str:
    """Canonical window label, e.g. 'w0_30'."""
    return f"w{start}_{end}"


def generate_sequential_windows(
    start_days: int = 0,
    end_days: int = 90,
    step_days: int = 10,
) -> List[Tuple[int, int]]:
    """Generate sequential windows such as 0-10, 10-20, ... up to ``end_days``."""
    if step_days <= 0:
        raise ValueError("step_days must be positive")
    if end_days <= start_days:
        raise ValueError("end_days must be greater than start_days")

    windows: List[Tuple[int, int]] = []
    cur = start_days
    while cur < end_days:
        nxt = min(cur + step_days, end_days)
        windows.append((cur, nxt))
        cur = nxt
    return windows


def _window_sort_key(win_label: str) -> Tuple[int, int]:
    m = re.match(r"^w(\d+)_(\d+)$", win_label)
    if not m:
        return (10**9, 10**9)
    return (int(m.group(1)), int(m.group(2)))


def _ordered_window_labels(window_map: Dict[str, pd.DataFrame]) -> List[str]:
    return sorted(window_map.keys(), key=_window_sort_key)


def _base_feature_name(col: str) -> str:
    """Convert window-specific feature names back to a common base feature name."""
    if col.startswith("f_ind_"):
        return re.sub(r"_(w\d+_\d+)_", "_", col, count=1)

    patterns = (
        (r"^f_lab_(w\d+_\d+)_(.+)$", "f_lab_{}"),
        (r"^f_cbc_(w\d+_\d+)_(.+)$", "f_cbc_{}"),
        (r"^f_comp_(w\d+_\d+)_(.+)$", "f_comp_{}"),
    )
    for pattern, template in patterns:
        m = re.match(pattern, col)
        if m:
            return template.format(m.group(2))
    return col


def _short_feature_name(base_feature: str) -> str:
    return re.sub(r"^f_(ind|lab|cbc|comp)_", "", base_feature)


def _collect_person_ids(window_maps: Sequence[Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    person_ids = pd.DataFrame(columns=["person_id"])
    for window_map in window_maps:
        for df in window_map.values():
            if "person_id" not in df.columns:
                continue
            person_ids = pd.concat([person_ids, df[["person_id"]]], ignore_index=True)
    if person_ids.empty:
        return person_ids
    person_ids = person_ids.drop_duplicates().copy()
    person_ids["person_id"] = person_ids["person_id"].astype("int64")
    return person_ids


def build_temporal_trend_features(
    windowed_binary: Dict[str, pd.DataFrame],
    windowed_labs: Dict[str, pd.DataFrame],
    binary_features: Sequence[str] = DEFAULT_TEMPORAL_BINARY_FEATURES,
    lab_directions: Dict[str, str] = DEFAULT_TEMPORAL_LAB_DIRECTIONS,
    window_order: Optional[Sequence[str]] = None,
    lab_stat: str = "median",
    min_delta: float = 0.0,
) -> pd.DataFrame:
    """
    Build patient-level temporal trend binaries across an ordered set of windows.

    Binary signals produce 0→1 (rise) and 1→0 (fall) flags for each adjacent
    interval plus first→last summary flags. Numeric lab signals produce
    direction-of-change flags using per-window lab summaries.
    """
    ordered = list(window_order) if window_order else _ordered_window_labels({**windowed_binary, **windowed_labs})
    if len(ordered) < 2:
        return _collect_person_ids([windowed_binary, windowed_labs])

    result = _collect_person_ids([windowed_binary, windowed_labs])
    if result.empty:
        return result

    # Binary trend features for selected PASC-associated indicators.
    for base_feature in binary_features:
        short_name = _short_feature_name(base_feature)
        window_cols: Dict[str, str] = {}
        for wl in ordered:
            bdf = windowed_binary.get(wl)
            if bdf is None or bdf.empty:
                continue
            for col in bdf.columns:
                if col == "person_id":
                    continue
                if _base_feature_name(col) == base_feature:
                    window_cols[wl] = col
                    break

        # Guard (defect 5): zero matches across ALL windows means the base
        # feature name is a ghost (its source column is no longer generated),
        # not merely too sparse for a trend.  Warn loudly, naming the feature.
        if len(window_cols) == 0:
            print(
                f"  [temporal-trend] *** GHOST WARNING: base feature "
                f"'{base_feature}' matched ZERO columns in any window "
                f"{list(ordered)} -> no trend features emitted. Repoint or remove it."
            )
        if len(window_cols) < 2:
            continue

        merged = result[["person_id"]].copy()
        usable_windows = [wl for wl in ordered if wl in window_cols]
        for wl in usable_windows:
            col = window_cols[wl]
            merged = merged.merge(
                windowed_binary[wl][["person_id", col]].rename(columns={col: wl}),
                on="person_id",
                how="left",
            )
            merged[wl] = merged[wl].fillna(0).astype("int8")

        if len(usable_windows) < 2:
            continue

        rise_flags = []
        fall_flags = []
        for early_wl, late_wl in zip(usable_windows[:-1], usable_windows[1:]):
            rise_col = f"f_temp_bin_{short_name}_rise_{early_wl}_to_{late_wl}"
            fall_col = f"f_temp_bin_{short_name}_fall_{early_wl}_to_{late_wl}"
            merged[rise_col] = ((merged[early_wl] == 0) & (merged[late_wl] == 1)).astype("int8")
            merged[fall_col] = ((merged[early_wl] == 1) & (merged[late_wl] == 0)).astype("int8")
            rise_flags.append(rise_col)
            fall_flags.append(fall_col)

        first_wl = usable_windows[0]
        last_wl = usable_windows[-1]
        merged[f"f_temp_bin_{short_name}_late_gt_early"] = (
            (merged[first_wl] == 0) & (merged[last_wl] == 1)
        ).astype("int8")
        merged[f"f_temp_bin_{short_name}_late_lt_early"] = (
            (merged[first_wl] == 1) & (merged[last_wl] == 0)
        ).astype("int8")
        merged[f"f_temp_bin_{short_name}_any_rise"] = merged[rise_flags].max(axis=1).astype("int8")
        merged[f"f_temp_bin_{short_name}_any_fall"] = merged[fall_flags].max(axis=1).astype("int8")

        keep_cols = [c for c in merged.columns if c.startswith(f"f_temp_bin_{short_name}_")]
        result = result.merge(merged[["person_id"] + keep_cols], on="person_id", how="left")

    # Numeric lab trend features for selected PASC-associated labs.
    for lab_name, direction in lab_directions.items():
        direction = direction.lower().strip()
        if direction not in {"up", "down", "both"}:
            raise ValueError(f"Unsupported lab direction '{direction}' for {lab_name}")

        merged = result[["person_id"]].copy()
        usable_windows: List[str] = []
        for wl in ordered:
            ldf = windowed_labs.get(wl)
            col = f"f_lab_{wl}_{lab_name}_{lab_stat}"
            if ldf is None or ldf.empty or col not in ldf.columns:
                continue
            merged = merged.merge(ldf[["person_id", col]].rename(columns={col: wl}), on="person_id", how="left")
            usable_windows.append(wl)

        if len(usable_windows) < 2:
            continue

        created_cols: List[str] = []
        up_flags: List[str] = []
        down_flags: List[str] = []
        for early_wl, late_wl in zip(usable_windows[:-1], usable_windows[1:]):
            delta = merged[late_wl] - merged[early_wl]
            if direction in {"up", "both"}:
                col = f"f_temp_lab_{lab_name}_up_{early_wl}_to_{late_wl}"
                merged[col] = ((merged[early_wl].notna()) & (merged[late_wl].notna()) & (delta > min_delta)).astype("int8")
                created_cols.append(col)
                up_flags.append(col)
            if direction in {"down", "both"}:
                col = f"f_temp_lab_{lab_name}_down_{early_wl}_to_{late_wl}"
                merged[col] = ((merged[early_wl].notna()) & (merged[late_wl].notna()) & (delta < -min_delta)).astype("int8")
                created_cols.append(col)
                down_flags.append(col)

            # Continuous: actual delta value (late - early) for this pair
            delta_col = f"f_temp_lab_{lab_name}_delta_{early_wl}_to_{late_wl}"
            both_present = merged[early_wl].notna() & merged[late_wl].notna()
            merged[delta_col] = np.where(both_present, delta, np.nan)
            created_cols.append(delta_col)

        first_wl = usable_windows[0]
        last_wl = usable_windows[-1]
        overall_delta = merged[last_wl] - merged[first_wl]
        if direction in {"up", "both"}:
            col = f"f_temp_lab_{lab_name}_late_gt_early"
            merged[col] = ((merged[first_wl].notna()) & (merged[last_wl].notna()) & (overall_delta > min_delta)).astype("int8")
            created_cols.append(col)
            if up_flags:
                col_any = f"f_temp_lab_{lab_name}_any_up"
                merged[col_any] = merged[up_flags].max(axis=1).astype("int8")
                created_cols.append(col_any)
        if direction in {"down", "both"}:
            col = f"f_temp_lab_{lab_name}_late_lt_early"
            merged[col] = ((merged[first_wl].notna()) & (merged[last_wl].notna()) & (overall_delta < -min_delta)).astype("int8")
            created_cols.append(col)
            if down_flags:
                col_any = f"f_temp_lab_{lab_name}_any_down"
                merged[col_any] = merged[down_flags].max(axis=1).astype("int8")
                created_cols.append(col_any)

        # Continuous: overall delta (last window - first window)
        overall_delta_col = f"f_temp_lab_{lab_name}_delta_overall"
        both_present = merged[first_wl].notna() & merged[last_wl].notna()
        merged[overall_delta_col] = np.where(both_present, overall_delta, np.nan)
        created_cols.append(overall_delta_col)

        result = result.merge(merged[["person_id"] + created_cols], on="person_id", how="left")

    trend_cols = [c for c in result.columns if c.startswith("f_temp_")]
    # Fill binary trend cols with 0; leave continuous delta cols as NaN (handled downstream)
    for col in trend_cols:
        if "_delta_" in col:
            continue  # continuous — leave NaN for median imputation downstream
        result[col] = result[col].fillna(0).astype("int8")

    print(f"  Temporal trend features built: {len(trend_cols)} columns across {len(ordered)} windows")
    return result


# =============================================================================
# 1.  MULTI-WINDOW HELPERS
# =============================================================================

def make_window_cohort(
    cohort_df: pd.DataFrame,
    start_days: int,
    end_days: int,
) -> pd.DataFrame:
    """
    Create a cohort copy with postcovid_window_start/end set to
    covid_index_date + offset (in days).

    Parameters
    ----------
    cohort_df : DataFrame
        Must contain ``person_id`` and ``covid_index_date``.
    start_days, end_days : int
        Window boundaries in days relative to index date.

    Returns
    -------
    DataFrame with columns person_id, postcovid_window_start, postcovid_window_end
    """
    df = cohort_df[["person_id", "covid_index_date"]].drop_duplicates(subset=["person_id"]).copy()
    idx = pd.to_datetime(df["covid_index_date"])
    df["postcovid_window_start"] = (idx + pd.Timedelta(days=start_days)).dt.date
    df["postcovid_window_end"] = (idx + pd.Timedelta(days=end_days)).dt.date
    return df[["person_id", "postcovid_window_start", "postcovid_window_end"]]


def _parse_window_label(win_label: str) -> Optional[Tuple[int, int]]:
    """Parse labels like 'w0_30' -> (0, 30)."""
    m = re.match(r"^w(\d+)_(\d+)$", win_label)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def extract_binary_signals_for_window(
    cur,
    cohort_df: pd.DataFrame,
    indicators: Sequence,
    mechanism_prefix: str,
    win_label: str,
) -> pd.DataFrame:
    """
    Extract binary indicator signals for a specific temporal window.

    Uploads the window-specific cohort to #cohort_temp, calls build_signals,
    then renames feature columns with a window prefix.

    Parameters
    ----------
    cur : DB cursor
    cohort_df : DataFrame with person_id, postcovid_window_start, postcovid_window_end
    indicators : indicator enum list
    mechanism_prefix : e.g. "viral", "immuno", "endo"
    win_label : e.g. "w0_30"

    Returns
    -------
    DataFrame with person_id and prefixed binary columns
    """
    # For episodic-persistence indicators (e.g., elevated episodes with gaps),
    # short windows are unstable and mostly encode sparsity.
    filtered_indicators = list(indicators)
    bounds = _parse_window_label(win_label)
    if bounds is not None:
        start_d, end_d = bounds
        window_days = max(0, end_d - start_d)
        if window_days < 60:
            kept = []
            dropped = []
            for ind in filtered_indicators:
                spec = getattr(ind, "value", None)
                if spec is not None and getattr(spec, "kind", None) == IndicatorKind.MEAS_ELEV_EPISODES_GAP:
                    dropped.append(ind.name.lower())
                    continue
                kept.append(ind)
            filtered_indicators = kept
            if dropped:
                print(f"  [{win_label}] Skipping episodic persistence indicators (<60d window): {dropped}")

    prepare_temp_cohort(cur, cohort_df)
    signals = build_signals(cur, cohort_df, filtered_indicators, mechanism_prefix)

    # Rename feature columns with window prefix
    rename = {}
    for c in signals.columns:
        if c.startswith("f_ind_"):
            # f_ind_viral_xxx  ->  f_ind_viral_w0_30_xxx
            parts = c.split("_", 3)  # ['f', 'ind', 'viral', 'xxx']
            if len(parts) >= 4:
                rename[c] = f"{parts[0]}_{parts[1]}_{parts[2]}_{win_label}_{parts[3]}"
            else:
                rename[c] = f"{c}_{win_label}"
    signals = signals.rename(columns=rename)
    return signals


def compute_cross_window_summaries(
    window_dfs: Dict[str, pd.DataFrame],
    mechanism_prefix: str,
    original_indicators: Sequence,
) -> pd.DataFrame:
    """
    Compute cross-window summary features from per-window binary DataFrames.

    For each original indicator, produces:
      - ``persistent``: hit in >=2 windows
      - ``early_only``: hit in first window only
      - ``late_only``:  hit in last window only

    Parameters
    ----------
    window_dfs : dict  win_label -> DataFrame (person_id + binary cols)
    mechanism_prefix : e.g. "viral"
    original_indicators : indicator enum list (for name extraction)

    Returns
    -------
    DataFrame with person_id and summary columns
    """
    sorted_wins = sorted(window_dfs.keys())
    if len(sorted_wins) < 2:
        return window_dfs[sorted_wins[0]][["person_id"]].copy()

    first_win = sorted_wins[0]
    last_win = sorted_wins[-1]
    pids = window_dfs[first_win][["person_id"]].copy()

    for ind in original_indicators:
        base_name = ind.name.lower()
        col_per_win = {}
        for wl in sorted_wins:
            wdf = window_dfs[wl]
            col = f"f_ind_{mechanism_prefix}_{wl}_{base_name}"
            if col in wdf.columns:
                col_per_win[wl] = wdf[["person_id", col]].rename(columns={col: wl})

        if not col_per_win:
            continue

        merged = pids.copy()
        for wl, part in col_per_win.items():
            merged = merged.merge(part, on="person_id", how="left")

        win_cols = [wl for wl in sorted_wins if wl in merged.columns]
        if not win_cols:
            continue

        for wl in win_cols:
            merged[wl] = merged[wl].fillna(0).astype(int)

        n_hits = merged[win_cols].sum(axis=1)
        pids[f"f_ind_{mechanism_prefix}_xw_persistent_{base_name}"] = (n_hits >= 2).astype("int8")
        pids[f"f_ind_{mechanism_prefix}_xw_early_only_{base_name}"] = (
            (merged[first_win] == 1) & (n_hits == 1)
        ).astype("int8")
        pids[f"f_ind_{mechanism_prefix}_xw_late_only_{base_name}"] = (
            (merged[last_win] == 1) & (n_hits == 1)
        ).astype("int8")

    return pids


# =============================================================================
# 2.  NUMERIC LAB EXTRACTION
# =============================================================================

@dataclass(frozen=True)
class NumericLabSpec:
    """Specification for a numeric lab feature."""
    lab_name: str
    loinc_codes: Tuple[str, ...]
    direct_concept_ids: Tuple[int, ...] = ()
    normal_high: Optional[float] = None   # value > this is abnormal-high
    normal_low: Optional[float] = None    # value < this is abnormal-low
    unit_hint: str = ""


# Priority numeric labs with literature-based normal ranges
NUMERIC_LAB_SPECS = {
    "crp": NumericLabSpec("crp", ("1988-5", "30522-7", "71426-1"), normal_high=10.0, unit_hint="mg/L"),
    "esr": NumericLabSpec("esr", ("30341-2", "18184-2"), normal_high=20.0, unit_hint="mm/hr"),
    "ferritin": NumericLabSpec("ferritin", ("2276-4",), normal_high=300.0, unit_hint="ng/mL"),
    "ddimer": NumericLabSpec("ddimer", ("48065-7", "48066-5", "48067-3", "71427-9"), normal_high=0.5, unit_hint="ug/mL FEU"),
    "fibrinogen": NumericLabSpec("fibrinogen", ("3255-7",), normal_high=400.0, unit_hint="mg/dL"),
    "albumin": NumericLabSpec("albumin", ("1751-7",), normal_low=3.5, unit_hint="g/dL"),
    "ldh": NumericLabSpec("ldh", ("2532-0", "14805-6"), normal_high=250.0, unit_hint="U/L"),
    "creatinine": NumericLabSpec("creatinine", ("2160-0",), normal_high=1.2, unit_hint="mg/dL"),
    "ast": NumericLabSpec("ast", ("1920-8",), direct_concept_ids=(4189605,), normal_high=40.0, unit_hint="U/L"),
    "alt": NumericLabSpec("alt", ("1742-6",), direct_concept_ids=(4189605,), normal_high=40.0, unit_hint="U/L"),
    "troponin": NumericLabSpec("troponin", ("6598-7", "10839-9", "49563-0", "89579-7"), normal_high=0.04, unit_hint="ng/mL"),
    "nt_probnp": NumericLabSpec("nt_probnp", ("33762-6", "83107-3"), normal_high=125.0, unit_hint="pg/mL"),
    "lactate": NumericLabSpec("lactate", ("2524-7",), normal_high=2.0, unit_hint="mmol/L"),
    "platelets": NumericLabSpec("platelets", ("777-3", "26515-7"), normal_high=400.0, normal_low=150.0, unit_hint="10^3/uL"),
    "pt_inr": NumericLabSpec("pt_inr", ("34714-6", "6301-6"), normal_high=1.1, unit_hint="INR"),
    # REMOVED: 0% numeric extraction — LOINC 3173-2 doesn't map to value_as_number rows
    # Binary indicator (aptt_ordered) works via concept-based query; numeric values absent.
    # "aptt": NumericLabSpec("aptt", ("3173-2",), normal_high=35.0, unit_hint="sec"),
}

# CBC components needed for derived indices
CBC_SPECS = {
    "neutrophils_abs": NumericLabSpec("neutrophils_abs", ("751-8", "26499-4"), unit_hint="10^3/uL"),
    "lymphocytes_abs": NumericLabSpec("lymphocytes_abs", ("731-0", "26474-7"), unit_hint="10^3/uL"),
    "monocytes_abs": NumericLabSpec("monocytes_abs", ("742-7", "26484-6"), unit_hint="10^3/uL"),
    "eosinophils_abs": NumericLabSpec("eosinophils_abs", ("711-2", "26449-9"), unit_hint="10^3/uL"),
    "rdw": NumericLabSpec("rdw", ("788-0", "30385-9"), normal_high=14.5, unit_hint="%"),
}


def _resolve_lab_concept_ids(cur, spec: NumericLabSpec) -> List[int]:
    """Resolve LOINC codes + direct IDs into a list of concept_ids."""
    ids = list(spec.direct_concept_ids)
    if spec.loinc_codes:
        cs = ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=spec.loinc_codes)
        ids.extend(resolve_concept_ids(cur, cs))
    return list(set(ids))


def extract_numeric_labs(
    cur,
    cohort_df: pd.DataFrame,
    lab_specs: Dict[str, NumericLabSpec],
    win_label: str,
) -> pd.DataFrame:
    """
    Extract numeric lab values for a specific temporal window.

    Fetches raw measurement values, then aggregates per person:
      - peak, median, last, abnormal_count, abnormal_any, measured (binary)

    Parameters
    ----------
    cur : DB cursor
    cohort_df : DataFrame with person_id, postcovid_window_start, postcovid_window_end
    lab_specs : dict of lab_name -> NumericLabSpec
    win_label : e.g. "w0_30"

    Returns
    -------
    DataFrame with person_id and aggregated feature columns
    """
    # Upload window cohort to temp table
    prepare_temp_cohort(cur, cohort_df)

    # Resolve all concept IDs up front
    lab_concept_map = {}  # concept_id -> lab_name
    for lab_name, spec in lab_specs.items():
        cids = _resolve_lab_concept_ids(cur, spec)
        for cid in cids:
            lab_concept_map[cid] = lab_name

    if not lab_concept_map:
        print(f"  [{win_label}] No lab concepts resolved — returning empty")
        result = cohort_df[["person_id"]].drop_duplicates().copy()
        return result

    all_cids = list(lab_concept_map.keys())

    # Build CASE expression mapping concept_id -> lab_name
    case_parts = []
    for cid, lab_name in lab_concept_map.items():
        case_parts.append(f"WHEN m.measurement_concept_id = {cid} THEN '{lab_name}'")
    case_expr = "CASE " + " ".join(case_parts) + " END"

    sql = f"""
    SELECT
        m.person_id,
        {case_expr} AS lab_name,
        CAST(m.value_as_number AS DOUBLE) AS value_num,
        m.measurement_date,
        m.range_high
    FROM {CDM}.measurement m
    JOIN #cohort_temp c ON c.person_id = m.person_id
    WHERE m.measurement_date BETWEEN c.postcovid_window_start AND c.postcovid_window_end
      AND m.measurement_concept_id IN ({_ints_to_sql_in(all_cids)})
      AND m.value_as_number IS NOT NULL
    """
    print(f"  [{win_label}] Fetching numeric labs ({len(lab_specs)} types, {len(all_cids)} concepts)...")
    raw = run_query(cur, sql)
    raw.columns = [c.lower() for c in raw.columns]

    # Initialise result
    pids = cohort_df[["person_id"]].drop_duplicates().copy()
    pids["person_id"] = pids["person_id"].astype("int64")

    if raw.empty:
        print(f"  [{win_label}] No lab rows found — features will be empty")
        for lab_name in lab_specs:
            pids[f"f_lab_{win_label}_{lab_name}_measured"] = 0
        return pids

    raw["person_id"] = raw["person_id"].astype("int64")
    raw["value_num"] = pd.to_numeric(raw["value_num"], errors="coerce")
    raw = raw.dropna(subset=["value_num", "lab_name"])

    # Aggregate per (person, lab)
    for lab_name, spec in lab_specs.items():
        sub = raw[raw["lab_name"] == lab_name].copy()

        prefix = f"f_lab_{win_label}_{lab_name}"

        # Measured flag (binary)
        measured = sub.groupby("person_id").size().reset_index(name="cnt")
        measured[f"{prefix}_measured"] = 1
        pids = pids.merge(measured[["person_id", f"{prefix}_measured"]], on="person_id", how="left")
        pids[f"{prefix}_measured"] = pids[f"{prefix}_measured"].fillna(0).astype("int8")

        if sub.empty:
            pids[f"{prefix}_peak"] = np.nan
            pids[f"{prefix}_median"] = np.nan
            pids[f"{prefix}_last"] = np.nan
            pids[f"{prefix}_abnormal_count"] = 0
            pids[f"{prefix}_abnormal_any"] = 0
            continue

        # Core aggregations
        agg = sub.groupby("person_id")["value_num"].agg(
            peak="max", median="median"
        ).reset_index()
        agg = agg.rename(columns={"peak": f"{prefix}_peak", "median": f"{prefix}_median"})

        # Last value (most recent measurement)
        sub_sorted = sub.sort_values(["person_id", "measurement_date"])
        last_val = sub_sorted.groupby("person_id")["value_num"].last().reset_index()
        last_val = last_val.rename(columns={"value_num": f"{prefix}_last"})

        # Abnormal counts
        if spec.normal_high is not None:
            sub["_abnormal"] = (sub["value_num"] > spec.normal_high).astype(int)
        elif spec.normal_low is not None:
            sub["_abnormal"] = (sub["value_num"] < spec.normal_low).astype(int)
        else:
            sub["_abnormal"] = 0

        abn = sub.groupby("person_id")["_abnormal"].agg(
            total="sum", any_flag="max"
        ).reset_index()
        abn = abn.rename(columns={
            "total": f"{prefix}_abnormal_count",
            "any_flag": f"{prefix}_abnormal_any",
        })

        # Merge all aggs
        pids = pids.merge(agg, on="person_id", how="left")
        pids = pids.merge(last_val, on="person_id", how="left")
        pids = pids.merge(abn, on="person_id", how="left")

        pids[f"{prefix}_abnormal_count"] = pids[f"{prefix}_abnormal_count"].fillna(0).astype(int)
        pids[f"{prefix}_abnormal_any"] = pids[f"{prefix}_abnormal_any"].fillna(0).astype("int8")

    print(f"  [{win_label}] Numeric labs extracted: {len([c for c in pids.columns if c.startswith('f_lab_')])} columns")
    return pids


def compute_cross_window_trajectories(
    early_df: pd.DataFrame,
    late_df: pd.DataFrame,
    lab_specs: Dict[str, NumericLabSpec],
    early_label: str,
    late_label: str,
) -> pd.DataFrame:
    """
    Compute delta (late - early) for median lab values across two windows.

    Returns DataFrame with person_id and trajectory columns.
    """
    merged = early_df[["person_id"]].merge(late_df[["person_id"]], on="person_id", how="inner")

    for lab_name in lab_specs:
        early_col = f"f_lab_{early_label}_{lab_name}_median"
        late_col = f"f_lab_{late_label}_{lab_name}_median"
        delta_col = f"f_lab_traj_{lab_name}_delta_{early_label}_to_{late_label}"

        if early_col in early_df.columns and late_col in late_df.columns:
            tmp = early_df[["person_id", early_col]].merge(
                late_df[["person_id", late_col]], on="person_id", how="inner"
            )
            tmp[delta_col] = tmp[late_col] - tmp[early_col]
            merged = merged.merge(tmp[["person_id", delta_col]], on="person_id", how="left")

    # Persistent elevation flags
    for lab_name, spec in lab_specs.items():
        if spec.normal_high is None:
            continue
        early_abn = f"f_lab_{early_label}_{lab_name}_abnormal_any"
        late_abn = f"f_lab_{late_label}_{lab_name}_abnormal_any"
        persist_col = f"f_lab_traj_{lab_name}_persistent_elev"
        if early_abn in early_df.columns and late_abn in late_df.columns:
            tmp = early_df[["person_id", early_abn]].merge(
                late_df[["person_id", late_abn]], on="person_id", how="inner"
            )
            tmp[persist_col] = ((tmp[early_abn] == 1) & (tmp[late_abn] == 1)).astype("int8")
            merged = merged.merge(tmp[["person_id", persist_col]], on="person_id", how="left")

    return merged


# =============================================================================
# 3.  CBC-DERIVED INDICES
# =============================================================================

def extract_cbc_indices(
    cur,
    cohort_df: pd.DataFrame,
    win_label: str,
) -> pd.DataFrame:
    """
    Compute NLR, PLR, SII from CBC components for a specific window.

    Returns per-person: peak, median for each index, plus persistent-high flags.
    """
    # Resolve CBC concept IDs
    neut_ids = _resolve_lab_concept_ids(cur, CBC_SPECS["neutrophils_abs"])
    lymph_ids = _resolve_lab_concept_ids(cur, CBC_SPECS["lymphocytes_abs"])
    mono_ids = _resolve_lab_concept_ids(cur, CBC_SPECS["monocytes_abs"])
    eos_ids = _resolve_lab_concept_ids(cur, CBC_SPECS["eosinophils_abs"])
    plt_ids = _resolve_lab_concept_ids(cur, NUMERIC_LAB_SPECS["platelets"])
    rdw_ids = _resolve_lab_concept_ids(cur, CBC_SPECS["rdw"])

    all_ids = neut_ids + lymph_ids + mono_ids + eos_ids + plt_ids + rdw_ids
    if not all_ids:
        print(f"  [{win_label}] No CBC concepts resolved")
        return cohort_df[["person_id"]].drop_duplicates().copy()

    # Build lab mapping
    id_to_lab = {}
    for cid in neut_ids: id_to_lab[cid] = "neut"
    for cid in lymph_ids: id_to_lab[cid] = "lymph"
    for cid in mono_ids: id_to_lab[cid] = "mono"
    for cid in eos_ids: id_to_lab[cid] = "eos"
    for cid in plt_ids: id_to_lab[cid] = "plt"
    for cid in rdw_ids: id_to_lab[cid] = "rdw"

    case_parts = [f"WHEN m.measurement_concept_id = {cid} THEN '{lab}'" for cid, lab in id_to_lab.items()]
    case_expr = "CASE " + " ".join(case_parts) + " END"

    prepare_temp_cohort(cur, cohort_df)

    sql = f"""
    SELECT
        m.person_id,
        {case_expr} AS component,
        CAST(m.value_as_number AS DOUBLE) AS value_num,
        m.measurement_date
    FROM {CDM}.measurement m
    JOIN #cohort_temp c ON c.person_id = m.person_id
    WHERE m.measurement_date BETWEEN c.postcovid_window_start AND c.postcovid_window_end
      AND m.measurement_concept_id IN ({_ints_to_sql_in(all_ids)})
      AND m.value_as_number IS NOT NULL
    """
    print(f"  [{win_label}] Fetching CBC components...")
    raw = run_query(cur, sql)
    raw.columns = [c.lower() for c in raw.columns]

    pids = cohort_df[["person_id"]].drop_duplicates().copy()
    pids["person_id"] = pids["person_id"].astype("int64")

    if raw.empty:
        print(f"  [{win_label}] No CBC data found")
        return pids

    raw["person_id"] = raw["person_id"].astype("int64")
    raw["value_num"] = pd.to_numeric(raw["value_num"], errors="coerce")
    raw = raw.dropna(subset=["value_num", "component"])
    raw["measurement_date"] = pd.to_datetime(raw["measurement_date"])

    # Pivot to wide per (person_id, date)
    pivoted = raw.pivot_table(
        index=["person_id", "measurement_date"],
        columns="component",
        values="value_num",
        aggfunc="mean",
    ).reset_index()

    prefix = f"f_cbc_{win_label}"

    # Compute indices where components are available on same date
    if "neut" in pivoted.columns and "lymph" in pivoted.columns:
        mask = (pivoted["lymph"] > 0) & pivoted["neut"].notna() & pivoted["lymph"].notna()
        pivoted.loc[mask, "nlr"] = pivoted.loc[mask, "neut"] / pivoted.loc[mask, "lymph"]

    if "plt" in pivoted.columns and "lymph" in pivoted.columns:
        mask = (pivoted["lymph"] > 0) & pivoted["plt"].notna() & pivoted["lymph"].notna()
        pivoted.loc[mask, "plr"] = pivoted.loc[mask, "plt"] / pivoted.loc[mask, "lymph"]

    if all(c in pivoted.columns for c in ["plt", "neut", "lymph"]):
        mask = (pivoted["lymph"] > 0) & pivoted["plt"].notna() & pivoted["neut"].notna()
        pivoted.loc[mask, "sii"] = (
            pivoted.loc[mask, "plt"] * pivoted.loc[mask, "neut"] / pivoted.loc[mask, "lymph"]
        )

    # Aggregate per person
    index_cols = ["nlr", "plr", "sii"]
    for idx_name in index_cols:
        if idx_name not in pivoted.columns:
            continue
        grp = pivoted.dropna(subset=[idx_name]).groupby("person_id")[idx_name]
        agg = grp.agg(peak="max", median="median").reset_index()
        agg = agg.rename(columns={
            "peak": f"{prefix}_{idx_name}_peak",
            "median": f"{prefix}_{idx_name}_median",
        })
        pids = pids.merge(agg, on="person_id", how="left")

    # NLR slope (continuous trajectory signal: rising inflammation in acute window)
    if "nlr" in pivoted.columns:
        nlr_rows = pivoted.dropna(subset=["nlr"])[
            ["person_id", "measurement_date", "nlr"]
        ].sort_values(["person_id", "measurement_date"])
        nlr_slope_records = []
        for pid, g in nlr_rows.groupby("person_id"):
            if len(g) < 2:
                continue
            days = (g["measurement_date"] - g["measurement_date"].iloc[0]).dt.days.astype(float).values
            if len(np.unique(days)) < 2:
                continue
            try:
                slope = float(np.polyfit(days, g["nlr"].values.astype(float), 1)[0])
            except (np.linalg.LinAlgError, ValueError):
                slope = 0.0
            nlr_slope_records.append({"person_id": pid, f"{prefix}_nlr_slope": slope})
        if nlr_slope_records:
            pids = pids.merge(pd.DataFrame(nlr_slope_records), on="person_id", how="left")

    # Persistent lymphopenia flag (lymph < 1.0 x10^3/uL)
    if "lymph" in pivoted.columns:
        low_lymph = pivoted[pivoted["lymph"] < 1.0].groupby("person_id").size().reset_index(name="cnt")
        low_lymph[f"{prefix}_persistent_lymphopenia"] = (low_lymph["cnt"] >= 2).astype("int8")
        low_lymph[f"{prefix}_low_lymph_count"] = low_lymph["cnt"]
        pids = pids.merge(
            low_lymph[["person_id", f"{prefix}_persistent_lymphopenia", f"{prefix}_low_lymph_count"]],
            on="person_id", how="left"
        )
        pids[f"{prefix}_persistent_lymphopenia"] = pids[f"{prefix}_persistent_lymphopenia"].fillna(0).astype("int8")
        pids[f"{prefix}_low_lymph_count"] = pids[f"{prefix}_low_lymph_count"].fillna(0).astype(int)

        # Minimum lymphocyte value per person
        lymph_min = pivoted.groupby("person_id")["lymph"].min().reset_index()
        lymph_min = lymph_min.rename(columns={"lymph": f"{prefix}_lymph_min"})
        pids = pids.merge(lymph_min, on="person_id", how="left")

        # Continuous lymphocyte dynamics: delta (last-first) and OLS slope.
        # These replace reliance on the binary persistent_lymphopenia flag by
        # giving the model a continuous trajectory signal. Requires ≥2 measurements
        # on distinct days; otherwise 0 (with missing indicator added at assembly).
        lymph_rows = pivoted.dropna(subset=["lymph"])[
            ["person_id", "measurement_date", "lymph"]
        ].sort_values(["person_id", "measurement_date"])
        lymph_dyn_records = []
        for pid, g in lymph_rows.groupby("person_id"):
            if len(g) < 2:
                continue
            first_val = float(g["lymph"].iloc[0])
            last_val = float(g["lymph"].iloc[-1])
            delta = last_val - first_val
            days = (g["measurement_date"] - g["measurement_date"].iloc[0]).dt.days.astype(float).values
            if len(np.unique(days)) >= 2:
                try:
                    slope = float(np.polyfit(days, g["lymph"].values.astype(float), 1)[0])
                except (np.linalg.LinAlgError, ValueError):
                    slope = 0.0
            else:
                slope = 0.0
            lymph_dyn_records.append({
                "person_id": pid,
                f"{prefix}_lymph_delta": delta,
                f"{prefix}_lymph_slope": slope,
            })
        if lymph_dyn_records:
            lymph_dyn_df = pd.DataFrame(lymph_dyn_records)
            pids = pids.merge(lymph_dyn_df, on="person_id", how="left")

    # Monocytosis flag (mono > 1.0)
    if "mono" in pivoted.columns:
        hi_mono = pivoted[pivoted["mono"] > 1.0].groupby("person_id").size().reset_index(name="cnt")
        hi_mono[f"{prefix}_monocytosis"] = (hi_mono["cnt"] >= 1).astype("int8")
        hi_mono[f"{prefix}_high_mono_count"] = hi_mono["cnt"]
        pids = pids.merge(
            hi_mono[["person_id", f"{prefix}_monocytosis", f"{prefix}_high_mono_count"]],
            on="person_id", how="left"
        )
        pids[f"{prefix}_monocytosis"] = pids[f"{prefix}_monocytosis"].fillna(0).astype("int8")
        pids[f"{prefix}_high_mono_count"] = pids[f"{prefix}_high_mono_count"].fillna(0).astype(int)

        # Maximum monocyte value per person
        mono_max = pivoted.groupby("person_id")["mono"].max().reset_index()
        mono_max = mono_max.rename(columns={"mono": f"{prefix}_mono_max"})
        pids = pids.merge(mono_max, on="person_id", how="left")

    # Eosinophilia flag (eos > 0.5)
    if "eos" in pivoted.columns:
        hi_eos = pivoted[pivoted["eos"] > 0.5].groupby("person_id").size().reset_index(name="cnt")
        hi_eos[f"{prefix}_eosinophilia"] = (hi_eos["cnt"] >= 1).astype("int8")
        hi_eos[f"{prefix}_high_eos_count"] = hi_eos["cnt"]
        pids = pids.merge(
            hi_eos[["person_id", f"{prefix}_eosinophilia", f"{prefix}_high_eos_count"]],
            on="person_id", how="left"
        )
        pids[f"{prefix}_eosinophilia"] = pids[f"{prefix}_eosinophilia"].fillna(0).astype("int8")
        pids[f"{prefix}_high_eos_count"] = pids[f"{prefix}_high_eos_count"].fillna(0).astype(int)

        # Maximum eosinophil value per person
        eos_max = pivoted.groupby("person_id")["eos"].max().reset_index()
        eos_max = eos_max.rename(columns={"eos": f"{prefix}_eos_max"})
        pids = pids.merge(eos_max, on="person_id", how="left")

    # RDW peak/median
    if "rdw" in pivoted.columns:
        grp = pivoted.dropna(subset=["rdw"]).groupby("person_id")["rdw"]
        agg = grp.agg(peak="max", median="median").reset_index()
        agg = agg.rename(columns={
            "peak": f"{prefix}_rdw_peak",
            "median": f"{prefix}_rdw_median",
        })
        pids = pids.merge(agg, on="person_id", how="left")

    n_cols = len([c for c in pids.columns if c.startswith("f_cbc_")])
    print(f"  [{win_label}] CBC indices extracted: {n_cols} columns")
    return pids


# =============================================================================
# 3B. DERIVED IMMUNOLOGICAL SIGNALS: Impaired Seroconversion
# =============================================================================

def extract_impaired_seroconversion(
    cur,
    cohort_df: pd.DataFrame,
    win_label: str,
) -> pd.DataFrame:
    """
    Extract impaired seroconversion indicator (mechsig_immuno).

    Logic: Patient has documented COVID diagnosis OR any COVID vaccine exposure,
    AND has any serology measurement showing negative result AFTER the exposure,
    AND has NO serology measurement showing positive result ever (within window).

    This flags failure to mount a detectable antibody response despite documented
    exposure — a direct immunodeficiency signal.

    Returns:
        DataFrame with person_id + f_ind_enh_immuno_<win_label>_impaired_seroconversion
    """
    # LOINC codes for SARS-CoV-2 serology (same as baseline_ext extraction)
    loinc_codes = ",".join(f"'{lc}'" for lc in [
        "94661-6",   # SARS-CoV-2 IgG+IgM primary
        "94563-4",   # SARS-CoV-2 IgG
        "94769-7",   # quantitative total
        "94505-5",   # quantitative IgG
        "94762-2",   # secondary
        "94564-2",   # IgM
    ])
    
    # Positive/negative interpretation concept IDs (from OMOP value_as_concept_id)
    pos_ids = ",".join(str(x) for x in [45884084])  # "Positive"
    neg_ids = ",".join(str(x) for x in [45877985])  # "Negative"
    
    sql = f"""
    WITH window_cohort AS (
        SELECT
            person_id,
            postcovid_window_start,
            postcovid_window_end
        FROM #cohort_temp
    ),
    
    covid_exposure AS (
        SELECT DISTINCT
            wc.person_id,
            MIN(co.condition_start_date) AS covid_date
        FROM window_cohort wc
        JOIN {CDM}.condition_occurrence co
          ON co.person_id = wc.person_id
         AND co.condition_start_date <= wc.postcovid_window_end
         AND co.condition_concept_id = 37311061  -- COVID-19 diagnosis
        GROUP BY wc.person_id
        
        UNION ALL
        
        SELECT DISTINCT
            wc.person_id,
            MIN(de.drug_exposure_start_date) AS covid_date
        FROM window_cohort wc
        JOIN {CDM}.drug_exposure de
          ON de.person_id = wc.person_id
         AND de.drug_exposure_start_date <= wc.postcovid_window_end
        JOIN {CDM}.concept c
          ON c.concept_id = de.drug_concept_id
         AND UPPER(c.concept_name) LIKE '%COVID%VACCIN%'
        GROUP BY wc.person_id
    ),
    
    covid_exposed AS (
        SELECT DISTINCT person_id, MIN(covid_date) AS first_exposure_date
        FROM covid_exposure
        GROUP BY person_id
    ),
    
    serology_tests AS (
        SELECT
            wc.person_id,
            m.measurement_date,
            CASE
                WHEN m.value_as_concept_id IN ({pos_ids}) THEN 1
                WHEN m.value_as_concept_id IN ({neg_ids}) THEN 0
                ELSE -1
            END AS interp_positive
        FROM window_cohort wc
        JOIN {CDM}.measurement m
          ON m.person_id = wc.person_id
         AND m.measurement_date BETWEEN wc.postcovid_window_start AND wc.postcovid_window_end
        JOIN {CDM}.concept cm
          ON cm.concept_id = m.measurement_concept_id
         AND UPPER(cm.concept_code) IN ({loinc_codes})
        WHERE m.value_as_concept_id IS NOT NULL OR m.value_as_number IS NOT NULL
    ),
    
    ever_positive AS (
        SELECT DISTINCT person_id, 1 AS has_positive
        FROM serology_tests
        WHERE interp_positive = 1
    ),
    
    ever_negative_after_exposure AS (
        SELECT DISTINCT
            ce.person_id,
            1 AS has_negative_after
        FROM covid_exposed ce
        JOIN serology_tests st
          ON st.person_id = ce.person_id
         AND st.measurement_date >= ce.first_exposure_date
         AND st.interp_positive = 0
    ),
    
    agg AS (
        SELECT
            ce.person_id,
            CASE
                WHEN ep.has_positive IS NOT NULL THEN 0
                WHEN ena.has_negative_after IS NOT NULL AND ep.has_positive IS NULL THEN 1
                ELSE 0
            END AS f_ind_enh_{win_label}_immuno_impaired_seroconversion
        FROM covid_exposed ce
        LEFT JOIN ever_positive ep ON ep.person_id = ce.person_id
        LEFT JOIN ever_negative_after_exposure ena ON ena.person_id = ce.person_id
    )
    
    SELECT * FROM agg
    """
    
    print(f"  [{win_label}] Extracting impaired seroconversion...")
    result_df = run_query(cur, sql)
    result_df.columns = [c.lower() for c in result_df.columns]
    
    # Ensure all cohort patients are present with default 0
    pids = cohort_df[["person_id"]].drop_duplicates().copy()
    pids["person_id"] = pids["person_id"].astype("int64")
    
    if not result_df.empty:
        result_df["person_id"] = result_df["person_id"].astype("int64")
        result = pids.merge(result_df, on="person_id", how="left")
    else:
        result = pids.copy()
    
    col_name = f"f_ind_enh_{win_label}_immuno_impaired_seroconversion"
    if col_name not in result.columns:
        result[col_name] = 0
    result[col_name] = result[col_name].fillna(0).astype("int8")
    
    print(f"  [{win_label}] Impaired seroconversion: {result[col_name].sum():,} cases")
    return result[["person_id", col_name]]


# =============================================================================
# 4.  COMPOSITE FEATURES
# =============================================================================

def build_composite_features(
    cur,
    cohort_df: pd.DataFrame,
    lab_dfs: Dict[str, pd.DataFrame],
    binary_dfs: Dict[str, pd.DataFrame],
    gi_window: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """
    Build composite features from lab and binary signal DataFrames.

    Composites:
      - Complement + coagulation clusters
      - Organ-injury composites (cardiac, renal, neurologic, peripheral vascular)
      - GI-persistence cluster

    Parameters
    ----------
    cur : DB cursor (for GI-persistence SQL)
    cohort_df : cohort DataFrame (needs person_id, covid_index_date)
    lab_dfs : dict win_label -> numeric lab DataFrame
    binary_dfs : dict win_label -> binary signal DataFrame (combined viral+immuno+endo)
    gi_window : optional (start_days, end_days) for GI-persistence extraction.
               If None, defaults to (30, 180).
    """
    pids = cohort_df[["person_id"]].drop_duplicates().copy()
    pids["person_id"] = pids["person_id"].astype("int64")

    # --- A. Complement + coagulation clusters (per window) ---
    for wl, lab_df in lab_dfs.items():
        ddimer_abn = f"f_lab_{wl}_ddimer_abnormal_any"
        crp_abn = f"f_lab_{wl}_crp_abnormal_any"
        plt_peak = f"f_lab_{wl}_platelets_peak"

        # Complement abnormal from binary signals (if present)
        bdf = binary_dfs.get(wl, pd.DataFrame(columns=["person_id"]))
        comp_col = [c for c in bdf.columns if "complement" in c.lower()]
        has_comp = bdf[["person_id"]].copy()
        if comp_col:
            has_comp["_comp"] = bdf[comp_col].max(axis=1)
        else:
            has_comp["_comp"] = 0

        tmp = pids.merge(lab_df[["person_id"] + [c for c in [ddimer_abn, crp_abn] if c in lab_df.columns]],
                         on="person_id", how="left")
        tmp = tmp.merge(has_comp, on="person_id", how="left")

        # Cluster: complement + D-dimer
        if ddimer_abn in tmp.columns:
            col = f"f_comp_{wl}_complement_ddimer"
            tmp[col] = ((tmp.get("_comp", 0) == 1) & (tmp[ddimer_abn] == 1)).astype("int8")
            pids = pids.merge(tmp[["person_id", col]], on="person_id", how="left")
            pids[col] = pids[col].fillna(0).astype("int8")

        # Cluster: complement + CRP + D-dimer
        if ddimer_abn in tmp.columns and crp_abn in tmp.columns:
            col = f"f_comp_{wl}_complement_crp_ddimer"
            tmp[col] = (
                (tmp.get("_comp", 0) == 1) & (tmp[ddimer_abn] == 1) & (tmp[crp_abn] == 1)
            ).astype("int8")
            pids = pids.merge(tmp[["person_id", col]], on="person_id", how="left")
            pids[col] = pids[col].fillna(0).astype("int8")

            # Continuous: count of abnormal components (0–3)
            count_col = f"f_comp_{wl}_complement_crp_ddimer_count"
            tmp[count_col] = (
                tmp.get("_comp", pd.Series(0, index=tmp.index)).fillna(0).astype(int)
                + tmp[ddimer_abn].fillna(0).astype(int)
                + tmp[crp_abn].fillna(0).astype(int)
            )
            pids = pids.merge(tmp[["person_id", count_col]], on="person_id", how="left")
            pids[count_col] = pids[count_col].fillna(0).astype(int)

    # --- B. Organ-injury composites (per window from labs) ---
    for wl, lab_df in lab_dfs.items():
        # Cardiac: troponin_abnormal OR nt_probnp_abnormal
        trop_abn = f"f_lab_{wl}_troponin_abnormal_any"
        bnp_abn = f"f_lab_{wl}_nt_probnp_abnormal_any"
        trop_peak = f"f_lab_{wl}_troponin_peak"
        bnp_peak = f"f_lab_{wl}_nt_probnp_peak"
        if trop_abn in lab_df.columns or bnp_abn in lab_df.columns:
            merge_cols = [c for c in [trop_abn, bnp_abn, trop_peak, bnp_peak] if c in lab_df.columns]
            tmp = pids.merge(
                lab_df[["person_id"] + merge_cols],
                on="person_id", how="left"
            )
            col = f"f_comp_{wl}_cardiac_injury"
            vals = pd.DataFrame(index=tmp.index)
            if trop_abn in tmp.columns:
                vals["a"] = tmp[trop_abn].fillna(0)
            if bnp_abn in tmp.columns:
                vals["b"] = tmp[bnp_abn].fillna(0)
            tmp[col] = (vals.max(axis=1) >= 1).astype("int8")
            pids = pids.merge(tmp[["person_id", col]], on="person_id", how="left")
            pids[col] = pids[col].fillna(0).astype("int8")

            # Continuous: max severity ratio (peak / normal_high) across troponin & BNP
            severity_parts = []
            if trop_peak in tmp.columns:
                severity_parts.append(tmp[trop_peak] / 0.04)  # troponin normal_high
            if bnp_peak in tmp.columns:
                severity_parts.append(tmp[bnp_peak] / 125.0)  # nt_probnp normal_high
            if severity_parts:
                score_col = f"f_comp_{wl}_cardiac_injury_score"
                tmp[score_col] = pd.concat(severity_parts, axis=1).max(axis=1)
                pids = pids.merge(tmp[["person_id", score_col]], on="person_id", how="left")

        # Renal: creatinine_abnormal
        creat_abn = f"f_lab_{wl}_creatinine_abnormal_any"
        creat_peak = f"f_lab_{wl}_creatinine_peak"
        if creat_abn in lab_df.columns:
            col = f"f_comp_{wl}_renal_injury"
            merge_cols_r = ["person_id", creat_abn] + ([creat_peak] if creat_peak in lab_df.columns else [])
            tmp = pids.merge(lab_df[merge_cols_r], on="person_id", how="left")
            pids = pids.merge(
                tmp[["person_id"]].assign(**{col: tmp[creat_abn].fillna(0).astype("int8")}),
                on="person_id", how="left"
            )
            pids[col] = pids[col].fillna(0).astype("int8")

            # Continuous: creatinine severity ratio (peak / normal_high)
            if creat_peak in tmp.columns:
                ratio_col = f"f_comp_{wl}_renal_injury_ratio"
                tmp[ratio_col] = tmp[creat_peak] / 1.2  # creatinine normal_high
                pids = pids.merge(tmp[["person_id", ratio_col]], on="person_id", how="left")

    # --- C. GI-persistence cluster ---
    gi_start, gi_end = gi_window if gi_window else (30, 180)
    gi_df = _extract_gi_persistence(cur, cohort_df, start_days=gi_start, end_days=gi_end)
    pids = pids.merge(gi_df, on="person_id", how="left")
    for c in gi_df.columns:
        if c != "person_id":
            pids[c] = pids[c].fillna(0).astype("int8")

    n_cols = len([c for c in pids.columns if c.startswith("f_comp_")])
    print(f"  Composite features: {n_cols} columns")
    return pids


def _extract_gi_persistence(cur, cohort_df: pd.DataFrame, start_days: int = 30, end_days: int = 180) -> pd.DataFrame:
    """
    Extract GI-persistence cluster features within a specified window.

    - Chronic GI symptoms (diarrhea, abdominal pain, nausea)
    - GI procedures (endoscopy, colonoscopy, biopsy)
    - GI medication escalation (PPI, antidiarrheal)

    Parameters
    ----------
    start_days, end_days : window bounds in days post covid_index_date (default 30-180)
    """
    wdf = cohort_df[["person_id", "covid_index_date"]].drop_duplicates(subset=["person_id"]).copy()
    wdf["postcovid_window_start"] = (pd.to_datetime(wdf["covid_index_date"]) + pd.Timedelta(days=start_days)).dt.date
    wdf["postcovid_window_end"] = (pd.to_datetime(wdf["covid_index_date"]) + pd.Timedelta(days=end_days)).dt.date

    prepare_temp_cohort(cur, wdf)

    # GI symptoms via condition_occurrence with ancestor concepts
    sql = f"""
    SELECT
        c.person_id,
        MAX(CASE WHEN ca.ancestor_concept_id IN (196523, 4091513, 196152)    THEN 1 ELSE 0 END) AS f_comp_gi_chronic_diarrhea,
        MAX(CASE WHEN ca.ancestor_concept_id IN (200219, 4103703)             THEN 1 ELSE 0 END) AS f_comp_gi_abdominal_pain,
        MAX(CASE WHEN ca.ancestor_concept_id IN (27674, 4101344)              THEN 1 ELSE 0 END) AS f_comp_gi_nausea_vomiting
    FROM #cohort_temp c
    LEFT JOIN {CDM}.condition_occurrence co
      ON co.person_id = c.person_id
     AND co.condition_start_date BETWEEN c.postcovid_window_start AND c.postcovid_window_end
    LEFT JOIN {CDM}.concept_ancestor ca
      ON ca.descendant_concept_id = co.condition_concept_id
     AND ca.ancestor_concept_id IN (196523, 4091513, 196152, 200219, 4103703, 27674, 4101344)
    GROUP BY c.person_id
    """
    gi_symptoms = run_query(cur, sql)
    gi_symptoms.columns = [c.lower() for c in gi_symptoms.columns]

    # GI procedures (endoscopy, colonoscopy, biopsy)
    sql2 = f"""
    SELECT
        c.person_id,
        MAX(CASE WHEN UPPER(cn.concept_name) LIKE '%ENDOSCOP%'
                   OR UPPER(cn.concept_name) LIKE '%COLONOSCOP%'
                   OR UPPER(cn.concept_name) LIKE '%SIGMOIDOSCOP%' THEN 1 ELSE 0 END) AS f_comp_gi_endoscopy,
        MAX(CASE WHEN UPPER(cn.concept_name) LIKE '%BIOPSY%GI%'
                   OR UPPER(cn.concept_name) LIKE '%BIOPSY%INTESTIN%'
                   OR UPPER(cn.concept_name) LIKE '%BIOPSY%COLON%'
                   OR UPPER(cn.concept_name) LIKE '%BIOPSY%GASTRIC%' THEN 1 ELSE 0 END) AS f_comp_gi_biopsy
    FROM #cohort_temp c
    LEFT JOIN {CDM}.procedure_occurrence po
      ON po.person_id = c.person_id
     AND po.procedure_date BETWEEN c.postcovid_window_start AND c.postcovid_window_end
    LEFT JOIN {CDM}.concept cn
      ON cn.concept_id = po.procedure_concept_id
    GROUP BY c.person_id
    """
    gi_procs = run_query(cur, sql2)
    gi_procs.columns = [c.lower() for c in gi_procs.columns]

    # GI medication escalation
    sql3 = f"""
    SELECT
        c.person_id,
        MAX(CASE WHEN LOWER(cn.concept_name) LIKE '%omeprazole%'
                   OR LOWER(cn.concept_name) LIKE '%pantoprazole%'
                   OR LOWER(cn.concept_name) LIKE '%lansoprazole%'
                   OR LOWER(cn.concept_name) LIKE '%esomeprazole%'
                   OR LOWER(cn.concept_name) LIKE '%rabeprazole%' THEN 1 ELSE 0 END) AS f_comp_gi_ppi,
        MAX(CASE WHEN LOWER(cn.concept_name) LIKE '%loperamide%'
                   OR LOWER(cn.concept_name) LIKE '%bismuth%'
                   OR LOWER(cn.concept_name) LIKE '%cholestyramine%'
                   OR LOWER(cn.concept_name) LIKE '%diphenoxylate%' THEN 1 ELSE 0 END) AS f_comp_gi_antidiarrheal
    FROM #cohort_temp c
    LEFT JOIN {CDM}.drug_exposure de
      ON de.person_id = c.person_id
     AND de.drug_exposure_start_date BETWEEN c.postcovid_window_start AND c.postcovid_window_end
    LEFT JOIN {CDM}.concept cn
      ON cn.concept_id = de.drug_concept_id
    GROUP BY c.person_id
    """
    gi_drugs = run_query(cur, sql3)
    gi_drugs.columns = [c.lower() for c in gi_drugs.columns]

    # Merge
    result = gi_symptoms.merge(gi_procs, on="person_id", how="outer")
    result = result.merge(gi_drugs, on="person_id", how="outer")

    # Composite flag: any GI feature
    gi_cols = [c for c in result.columns if c.startswith("f_comp_gi_")]
    for c in gi_cols:
        result[c] = result[c].fillna(0).astype("int8")
    result["f_comp_gi_any"] = (result[gi_cols].max(axis=1) >= 1).astype("int8")

    # Continuous: burden score = count of GI categories present (0–7)
    result["f_comp_gi_burden_score"] = result[gi_cols].sum(axis=1).astype(int)

    return result


# =============================================================================
# 4b. TEMPORAL-DIVERGENCE COMPOSITES
# =============================================================================
#
# Features motivated by the exploratory analysis: capture post-acute (≥30 d) temporal
# patterns where PASC diverges most from controls — prolonged elevation,
# delayed resolution, late-emerging signals, and multi-system chronicity.
#
# These operate on *multi-window* binary DataFrames and produce features
# prefixed ``f_comp_td_*``.

# Mapping: indicator column substrings → composite feature
# Renamed from _TD_PROLONGED_INFLAMMATION and redefined (2026-07-17): the old
# binary elev_crp/elev_esr indicators were removed in v2 purification; this now
# fires on the numeric-lab abnormal flags (CRP > 10 mg/L, ESR > 20 mm/hr).  This
# is a NEW threshold feature, not a restoration of the availability-biased one.
_TD_CRP_ESR_ELEVATED = (
    "crp_abnormal_any",
    "esr_abnormal_any",
)

_TD_POST_ACUTE_RESPIRATORY = (
    "enh_viral_resp_dx_any",
)

# Restored (2026-07-17): coag_activation_any / coag_ddimer_elev_proxy were removed
# in v2; d-dimer elevation now lives in the numeric-lab stream.  No fibrinogen
# (that would widen beyond the original d-dimer focus).
_TD_SUSTAINED_COAG = (
    "ddimer_abnormal_any",
    "endo_repeated_ddimer_testing",
)

_TD_EMERGING_DYSAUTONOMIA = (
    "enh_endo_pots_orthostatic",
)

_TD_LATE_AUTOIMMUNE = (
    "enh_immuno_ana_ordered",
    "enh_immuno_antiphospholipid_ordered",
    "enh_immuno_rheumatology_visit",
    "enh_immuno_new_rheum_dx",
)

_TD_PERSISTENT_CYTOKINE = (
    "immuno_any_il6",
    "immuno_any_tnfa",
)

# endo_thrombotic_events (removed THROMBOTIC_EVENTS) replaced by the v2 arterial
# indicator (now MI + stroke, TIA stripped) -> restores venous + arterial cover.
_TD_UNRESOLVED_THROMBOSIS = (
    "enh_endo_dvt_pe",
    "enh_endo_arterial_thrombosis",
)

# All composite definitions keyed by output column name
_TD_DEFINITIONS: Dict[str, Tuple[str, ...]] = {
    "f_comp_td_crp_esr_elevated":         _TD_CRP_ESR_ELEVATED,
    "f_comp_td_post_acute_respiratory":   _TD_POST_ACUTE_RESPIRATORY,
    "f_comp_td_sustained_coag":           _TD_SUSTAINED_COAG,
    "f_comp_td_emerging_dysautonomia":    _TD_EMERGING_DYSAUTONOMIA,
    "f_comp_td_late_autoimmune":          _TD_LATE_AUTOIMMUNE,
    "f_comp_td_persistent_cytokine":      _TD_PERSISTENT_CYTOKINE,
    "f_comp_td_unresolved_thrombosis":    _TD_UNRESOLVED_THROMBOSIS,
}

# Multi-system score: which mechanism families to check
_TD_MULTISYSTEM_GROUPS: Dict[str, Tuple[str, ...]] = {
    "viral":  ("viral_", "enh_viral_"),
    "immuno": ("immuno_", "enh_immuno_"),
    "endo":   ("endo_", "enh_endo_"),
}


def build_temporal_divergence_composites(
    windowed_binary: Dict[str, pd.DataFrame],
    cohort_df: pd.DataFrame,
    post_acute_days: int = 30,
    max_window_end_day: Optional[int] = None,
) -> pd.DataFrame:
    """Build temporal-divergence composite features from multi-window binaries.

    For each composite, any matching indicator column that is positive in a
    window whose start day is >= *post_acute_days* triggers the flag.

    Additionally computes ``f_comp_td_multisystem_late`` — the count (0-3) of
    mechanism groups (viral, immuno, endo) with ≥ 1 positive indicator in any
    post-acute window.

    **Leakage gate** (Step 3 of ascertainment/leakage fix):
    If *max_window_end_day* is given, only windows whose end day
    ``<= max_window_end_day`` are eligible.  This prevents td_* features
    from drawing on windows that overlap the PASC label window (day ≥60).
    For a w0-21 experiment, no windows qualify → td_* all = 0.
    For a w0-60 experiment with max_window_end_day=60, only w30_60 qualifies.

    Parameters
    ----------
    windowed_binary : dict  win_label -> DataFrame with person_id + binary cols
        Keys like ``"w0_21"``, ``"w30_60"``, ``"w60_90"`` etc.
    cohort_df : DataFrame with at least ``person_id``
    post_acute_days : int, default 30
        Minimum window-start (in days) to consider "post-acute".
    max_window_end_day : int or None
        If given, only include windows whose end day <= this value.
        Prevents temporal leakage into the PASC label window.

    Returns
    -------
    DataFrame with person_id + composite columns (int8, except multisystem which is int8 0-3).
    """
    pids = cohort_df[["person_id"]].drop_duplicates().copy()
    pids["person_id"] = pids["person_id"].astype("int64")

    # Identify post-acute windows (start_day >= post_acute_days)
    # Leakage gate: also exclude windows whose end_day > max_window_end_day
    late_windows: Dict[str, pd.DataFrame] = {}
    for wl, bdf in windowed_binary.items():
        # Parse start/end days from label like "w30_60" → start=30, end=60
        try:
            parts = wl.split("_")
            start_d = int(parts[0].replace("w", ""))
            end_d = int(parts[1])
        except (ValueError, IndexError):
            continue
        if start_d < post_acute_days:
            continue
        if max_window_end_day is not None and end_d > max_window_end_day:
            continue
        late_windows[wl] = bdf

    if not late_windows:
        print("  [temporal-divergence] No post-acute windows found; skipping.")
        for col in list(_TD_DEFINITIONS.keys()) + ["f_comp_td_multisystem_late"]:
            pids[col] = np.int8(0)
        return pids

    print(f"  [temporal-divergence] Post-acute windows (>={post_acute_days}d): "
          f"{sorted(late_windows.keys())}")

    # Helper: strip window label from column name for substring matching.
    # e.g. "f_ind_immuno_w30_60_elev_crp" → "f_ind_immuno_elev_crp"
    import re
    _WIN_RE = re.compile(r"_w\d+_\d+")

    def _strip_window(col: str) -> str:
        return _WIN_RE.sub("", col)

    # Single source of truth for which windowed indicator columns feed a
    # temporal-divergence composite.  The zero-match guard below and both
    # aggregation helpers use this, so guard and matcher can never disagree.
    # Accept both f_ind_* binary indicators and f_lab_*_abnormal_any numeric-lab
    # flags (defect-6: v2 purification moved crp/esr/ddimer elevation into f_lab_*).
    def _col_matches(col: str, substrings: Tuple[str, ...]) -> bool:
        if not (col.startswith("f_ind_") or col.startswith("f_lab_")):
            return False
        return any(s in _strip_window(col) for s in substrings)

    # Helper: check if *any* post-acute window has a 1 for columns matching
    # any of the given substrings (after stripping window labels)
    def _any_late_hit(substrings: Tuple[str, ...]) -> pd.Series:
        """Return Series (index = pids.index) with 1 if any match found."""
        hits = pd.Series(0, index=pids.index, dtype="int8")
        for _wl, bdf in late_windows.items():
            matched_cols = [c for c in bdf.columns if _col_matches(c, substrings)]
            if not matched_cols:
                continue
            tmp = pids[["person_id"]].merge(bdf[["person_id"] + matched_cols],
                                            on="person_id", how="left")
            for mc in matched_cols:
                tmp[mc] = tmp[mc].fillna(0)
            row_max = tmp[matched_cols].max(axis=1)
            hits = hits | (row_max.values >= 1).astype("int8")
        return hits

    def _count_late_windows(substrings: Tuple[str, ...]) -> pd.Series:
        """Return Series with count of post-acute windows with any match (0–N)."""
        window_count = pd.Series(0, index=pids.index, dtype=int)
        for _wl, bdf in late_windows.items():
            matched_cols = [c for c in bdf.columns if _col_matches(c, substrings)]
            if not matched_cols:
                continue
            tmp = pids[["person_id"]].merge(bdf[["person_id"] + matched_cols],
                                            on="person_id", how="left")
            for mc in matched_cols:
                tmp[mc] = tmp[mc].fillna(0)
            row_max = tmp[matched_cols].max(axis=1)
            window_count = window_count + (row_max.values >= 1).astype(int)
        return window_count

    # --- Binary composites (features 1-7) + continuous window-count variants ---
    for comp_col, substrings in _TD_DEFINITIONS.items():
        # Guard (defect 5): a composite whose substrings match NO indicator
        # column in ANY late window is silently constant-zero -- that is how
        # prolonged_inflammation / sustained_coag decayed unnoticed.  Warn
        # loudly, naming the composite and its unmatched substrings.
        if not any(_col_matches(c, substrings)
                   for bdf in late_windows.values() for c in bdf.columns):
            raise ValueError(
                f"[temporal-divergence] composite '{comp_col}' matched ZERO columns "
                f"for substrings {substrings} across late windows {sorted(late_windows)} "
                f"-> would be constant-zero. Repoint or remove it before extraction."
            )
        pids[comp_col] = _any_late_hit(substrings)
        # Continuous: number of post-acute windows with this signal
        pids[f"{comp_col}_n_windows"] = _count_late_windows(substrings)
    prev_summary = []
    for comp_col in _TD_DEFINITIONS:
        n = int(pids[comp_col].sum())
        pct = n / len(pids) * 100
        prev_summary.append(f"    {comp_col:45s}: {n:6,} ({pct:5.2f}%)")

    # --- Multi-system late activity score (feature 8) ---
    mech_hits = {}
    for mech, prefixes in _TD_MULTISYSTEM_GROUPS.items():
        mech_hits[mech] = _any_late_hit(prefixes)

    pids["f_comp_td_multisystem_late"] = (
        mech_hits["viral"].astype("int8")
        + mech_hits["immuno"].astype("int8")
        + mech_hits["endo"].astype("int8")
    )

    n_cols = len([c for c in pids.columns if c.startswith("f_comp_td_")])
    print(f"  [temporal-divergence] {n_cols} composite features built")
    for line in prev_summary:
        print(line)
    n_multi = int((pids["f_comp_td_multisystem_late"] >= 2).sum())
    print(f"    {'f_comp_td_multisystem_late (>=2)':45s}: {n_multi:6,} ({n_multi/len(pids)*100:5.2f}%)")

    return pids


# =============================================================================
# 5.  ENHANCED INDICATOR ENUMS
# =============================================================================

class EnhViralIndicator(Enum):
    """Enhanced viral persistence / reactivation indicators."""

    # REMOVED 2026-05-13 (v3): this indicator was a renamed duplicate of
    # ViralIndicator.ANY_POS_SARS_COV2_TEST (in indicator_definitions.py):
    # identical concept IDs, identical value_concept_ids, identical
    # MEAS_ANY_RECORDED kind. The "late" naming was aspirational -- no
    # temporal lateness was enforced at the spec level. Empirically at
    # w0_21: prevalence 47.5%, correlation with PASC -0.058 (slight
    # negative). The feature was behaving as a cohort-modality /
    # ascertainment-completeness indicator rather than a viral persistence
    # indicator. Removed to clean up the SHAP interpretation of the viral
    # mechanism arm. The literature-faithful viral persistence indicators
    # live in ViralIndicator.PERSISTENT_POS_SARS_COV2_{STRICT,14D,ANY}.
    #
    # LATE_POS_SARS_COV2 = IndicatorSpec(
    #     label="Any positive SARS-CoV-2 test in window (for late-window use)",
    #     kind=IndicatorKind.MEAS_ANY_RECORDED,
    #     table="CDMPHI.measurement",
    #     concepts_any_of=(
    #         ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS,
    #                     direct_ids=(706169, 586526, 706170, 706163, 723476)),
    #     ),
    #     value_concept_ids=tuple(COVID_PCR_POSITIVE_VALUES),
    # )

    # EBV: separate ordered vs positive, PCR vs serology
    EBV_PCR_ANY = IndicatorSpec(
        label="EBV PCR test (any result/value recorded)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES,
                        loinc_codes=("32585-2",)),
        ),
        require_any_value=True,
    )

    EBV_EARLY_ANTIGEN = IndicatorSpec(
        label="EBV early antigen (EA) IgG — reactivation marker",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES,
                        loinc_codes=("24007-2", "30339-6")),
        ),
        source_value_patterns=("%ebv%early antigen%", "%ebv%ea%igg%", "%ebv%ea-d%"),
        require_any_value=True,
    )

    EBV_VCA_IGM = IndicatorSpec(
        label="EBV VCA IgM — acute/reactivation marker",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES,
                        loinc_codes=("7885-7",)),
        ),
        source_value_patterns=("%ebv%vca%igm%", "%ebv%capsid%igm%"),
        require_any_value=True,
    )

    # CMV: any PCR row with a result value (upgraded from ordering-only)
    CMV_PCR_ANY = IndicatorSpec(
        label="CMV PCR test (any result/value recorded)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS, ancestor_ids=(37172169,)),
        ),
        require_any_value=True,
    )

    # GI symptoms (condition-based, any occurrence in window)
    GI_DX_ANY = IndicatorSpec(
        label="GI diagnosis (diarrhea, abdominal pain, nausea) in window",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS,
                        ancestor_ids=(196523, 200219, 27674)),
        ),
    )

    # Respiratory symptoms (condition-based, any occurrence in window)
    RESP_DX_ANY = IndicatorSpec(
        label="Respiratory symptoms (cough, dyspnea) in window",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS,
                        ancestor_ids=(254761, 312437)),
        ),
    )

    # --- SARS-CoV-2 antibody subtypes (decomposition of aggregate) ---
    SARS_COV2_NUCLEOCAPSID_IGG = IndicatorSpec(
        label="SARS-CoV-2 nucleocapsid IgG (prior infection marker)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(
                1988397,  # SARS-CoV-2 nucleocapsid IgG Ab (audit-validated)
            )),
            # was: (723478, 40771922)
            #   723478 = ORF1ab NAA (NOT nucleocapsid; PCR target gene)
            #   40771922 = eGFR (kidney function lab) — fired on 33,396 patients
        ),
        source_value_patterns=("%nucleocapsid%igg%", "%sars%cov%nucleocapsid%", "%covid%nucleocapsid%"),
        require_any_value=True,
    )

    SARS_COV2_SPIKE_IGG = IndicatorSpec(
        label="SARS-CoV-2 spike IgG (vaccination or infection response)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(
                1988202,  # SARS-CoV-2 spike IgG (audit-validated)
                4132298,  # SARS-CoV-2 IgG
                4196936,  # SARS-CoV-2 IgG Ab [quantitative]
                4211116,  # SARS-CoV-2 IgG Ab [qualitative]
            )),
            # was: 40763481 included — maps to original-SARS IgG Ab, not CoV-2
        ),
        source_value_patterns=("%spike%igg%", "%sars%cov%igg%", "%covid%igg%", "%sars%igg%"),
        require_any_value=True,
    )

    SARS_COV2_IGM = IndicatorSpec(
        label="SARS-CoV-2 IgM (early/acute humoral response)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.NAME_PATTERN,
                        name_patterns=("%SARS%COV%2%IGM%",),
                        domain_id="Measurement"),
        ),
        source_value_patterns=("%sars%cov%igm%", "%covid%igm%"),
        require_any_value=True,
    )

    # REMOVED: 0% prevalence in Mount Sinai OMOP — no data captured
    # SARS_COV2_NEUTRALIZING = IndicatorSpec(
    #     label="SARS-CoV-2 neutralizing antibody",
    #     kind=IndicatorKind.MEAS_ANY_RECORDED,
    #     table="CDMPHI.measurement",
    #     concepts_any_of=(
    #         ConceptSpec(kind=ConceptSourceKind.NAME_PATTERN,
    #                     name_patterns=("%SARS%COV%2%NEUTRALI%",),
    #                     domain_id="Measurement"),
    #     ),
    #     source_value_patterns=("%sars%neutraliz%", "%covid%neutraliz%"),
    #     require_any_value=True,
    # )


ENH_VIRAL_INDICATOR_LIST = list(EnhViralIndicator)


class EnhImmunoIndicator(Enum):
    """Enhanced immunoinflammatory indicators."""

    # Complement panel — value-gated (require result, not just ordering)
    COMPLEMENT_C3_ORDERED = IndicatorSpec(
        label="Complement C3 test with result",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("4485-9",)),
        ),
        source_value_patterns=("%complement c3%", "%complement 3%"),
        require_any_value=True,
    )

    COMPLEMENT_C4_ORDERED = IndicatorSpec(
        label="Complement C4 test with result",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("4498-2",)),
        ),
        source_value_patterns=("%complement c4%", "%complement 4%"),
        require_any_value=True,
    )

    COMPLEMENT_CH50_ORDERED = IndicatorSpec(
        label="Complement CH50/AH50 test with result",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("4532-8",)),
            ConceptSpec(kind=ConceptSourceKind.NAME_PATTERN, name_patterns=("%COMPLEMENT%CH50%", "%COMPLEMENT%AH50%", "%COMPLEMENT%TOTAL%HEMOLYTIC%"), domain_id="Measurement"),
        ),
        source_value_patterns=("%ch50%", "%total complement%", "%total hemolytic%"),
        require_any_value=True,
    )

    # Autoimmunity panel — value-gated (require result, not just ordering)
    ANA_ORDERED = IndicatorSpec(
        label="ANA test with result",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("8061-4",)),
        ),
        source_value_patterns=("%antinuclear%antibod%", "%ana %screen%", "%ana %titer%"),
        require_any_value=True,
    )

    RF_ORDERED = IndicatorSpec(
        label="Rheumatoid factor with result",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("11572-5",)),
        ),
        source_value_patterns=("%rheumatoid factor%",),
        require_any_value=True,
    )

    ANTI_CCP_ORDERED = IndicatorSpec(
        label="Anti-CCP antibody with result",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("53027-9",)),
            ConceptSpec(kind=ConceptSourceKind.NAME_PATTERN, name_patterns=("%ANTI%CCP%", "%CYCLIC CITRULLINATED PEPTIDE%"), domain_id="Measurement"),
        ),
        source_value_patterns=("%anti%ccp%", "%cyclic citrullinated%"),
        require_any_value=True,
    )

    ANTIPHOSPHOLIPID_ORDERED = IndicatorSpec(
        label="Any antiphospholipid antibody test with result (aCL, B2GP1, LAC)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.NAME_PATTERN, name_patterns=(
                "%ANTIPHOSPHOLIPID%",
                "%ANTICARDIOLIPIN%",
                "%BETA-2 GLYCOPROTEIN%",
                "%B2 GLYCOPROTEIN%",
                "%LUPUS ANTICOAGULANT%",
            ), domain_id="Measurement"),
        ),
        source_value_patterns=("%antiphospholipid%", "%anticardiolipin%", "%lupus anticoag%", "%beta-2 glycoprotein%"),
        require_any_value=True,
    )

    # Post-COVID rheumatologic diagnosis
    # Concept audit (2026-05-01): expanded value-set after audit found prior tuple
    # captured only ~30% of true rheumatic patients in cohort. Vasculitis
    # ancestor 81893 was too broad (449 descendants dominated by phlebitis/
    # varicose); replaced with curated 8-id vasculitis subset.
    NEW_RHEUM_DX = IndicatorSpec(
        label="New autoimmune/rheumatologic diagnosis in window",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS,
                        ancestor_ids=(
                            # Core rheumatology
                            80809,    # RA
                            257628,   # Rheumatic disease (broad)
                            134442,   # Dermatomyositis
                            80182,    # SLE
                            80800,    # Systemic sclerosis / scleroderma
                            254443,   # Sjogren syndrome
                            255348,   # Polymyositis
                            # Vasculitis (curated, was: 81893 — too broad)
                            313219,   # GPA (Wegener)
                            4101602,  # IgA vasculitis (HSP)
                            436642,   # Behcet disease
                            314963,   # Giant cell arteritis
                            4290976,  # Temporal arteritis
                            314381,   # Kawasaki disease
                            42535714, # ANCA-positive vasculitis
                            196431,   # Hypersensitivity vasculitis
                        )),
            # was: (257628, 80809, 134442, 4058824, 81893)
            #   4058824 — not Sjogren (correct = 254443)
            #   81893 — vasculitis ancestor too broad (449 descendants)
        ),
    )

    # JAK inhibitor / DMARD initiation post-acute
    JAK_INHIBITOR = IndicatorSpec(
        label="JAK inhibitor exposure in window",
        kind=IndicatorKind.DRUG_INGREDIENT_DESC,
        table="CDMPHI.drug_exposure",
        rxnorm_ingredient_names=(
            "tofacitinib", "baricitinib", "ruxolitinib", "upadacitinib",
        ),
        source_value_patterns=("%tofacitinib%", "%xeljanz%", "%baricitinib%", "%olumiant%", "%ruxolitinib%"),
    )

    DMARD_INITIATION = IndicatorSpec(
        label="DMARD exposure in window",
        kind=IndicatorKind.DRUG_INGREDIENT_DESC,
        table="CDMPHI.drug_exposure",
        rxnorm_ingredient_names=(
            "methotrexate", "hydroxychloroquine", "sulfasalazine",
            "leflunomide", "azathioprine", "mycophenolate",
        ),
        source_value_patterns=("%methotrexate%", "%hydroxychloroquine%", "%plaquenil%", "%sulfasalazine%", "%leflunomide%"),
    )

    IVIG_EXPOSURE = IndicatorSpec(
        label="IVIG exposure in window",
        kind=IndicatorKind.DRUG_INGREDIENT_DESC,
        table="CDMPHI.drug_exposure",
        rxnorm_ingredient_names=("immune globulin",),
        source_value_patterns=("%ivig%", "%gamunex%", "%gammagard%", "%privigen%", "%octagam%", "%immune globulin%intravenous%"),
    )

    # Specialty visits
    RHEUMATOLOGY_VISIT = IndicatorSpec(
        label="Rheumatology visit in window",
        kind=IndicatorKind.VISIT_SPECIALTY_NAME,
        table="CDMPHI.visit_occurrence",
        specialty_name_patterns=("%rheumatol%",),
    )

    IMMUNOLOGY_VISIT = IndicatorSpec(
        label="Immunology / allergy visit in window",
        kind=IndicatorKind.VISIT_SPECIALTY_NAME,
        table="CDMPHI.visit_occurrence",
        specialty_name_patterns=("%immunol%", "%allerg%"),
    )

    # --- Literature-driven: cytokines ---
    ANY_IL1B = IndicatorSpec(
        label="IL-1β measured (any result recorded)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("47032-8",)),
            ConceptSpec(kind=ConceptSourceKind.NAME_PATTERN,
                        name_patterns=("%INTERLEUKIN%1%BETA%",),
                        domain_id="Measurement"),
        ),
        source_value_patterns=("%il-1%beta%", "%il1b%", "%interleukin 1b%"),
        require_any_value=True,
    )

    # REMOVED: 0.02% prevalence — too sparse for modeling
    # ANY_IFNG = IndicatorSpec(
    #     label="IFN-γ measured (any result recorded)",
    #     kind=IndicatorKind.MEAS_ANY_RECORDED,
    #     table="CDMPHI.measurement",
    #     concepts_any_of=(
    #         ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("33264-3",)),
    #         ConceptSpec(kind=ConceptSourceKind.NAME_PATTERN,
    #                     name_patterns=("%INTERFERON%GAMMA%",),
    #                     domain_id="Measurement"),
    #     ),
    #     source_value_patterns=("%ifn%gamma%", "%interferon gamma%"),
    #     require_any_value=True,
    # )

    # --- Literature-driven: adaptive immune subsets ---
    CD4_COUNT = IndicatorSpec(
        label="CD4+ T-cell count ordered",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("24467-3",)),
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(37396514,)),
        ),
        source_value_patterns=("%cd4%count%", "%cd4%abs%", "%t-helper%"),
        require_any_value=True,
    )

    # REMOVED: 0% prevalence — LOINC 8113-2 returns no data in this DB
    # CD8_COUNT = IndicatorSpec(
    #     label="CD8+ T-cell count ordered",
    #     kind=IndicatorKind.MEAS_ANY_RECORDED,
    #     table="CDMPHI.measurement",
    #     concepts_any_of=(
    #         ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("8113-2",)),
    #     ),
    #     source_value_patterns=("%cd8%count%", "%cd8%abs%", "%t-cytotoxic%", "%t-suppressor%"),
    #     require_any_value=True,
    # )

    # --- Literature-driven: autoantibodies — value-gated ---
    DSDNA_ANTIBODY = IndicatorSpec(
        label="Anti-dsDNA antibody with result",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("5130-0",)),
        ),
        source_value_patterns=("%dsdna%", "%double stranded dna%", "%anti-dna%"),
        require_any_value=True,
    )

    THYROID_AUTOANTIBODY = IndicatorSpec(
        label="Thyroid autoantibody with result (TPO or thyroglobulin Ab)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES,
                        loinc_codes=("5382-7", "8098-6")),  # Anti-TPO, Anti-TG
        ),
        source_value_patterns=("%thyroid peroxidase%ab%", "%anti%tpo%", "%thyroglobulin%ab%"),
        require_any_value=True,
    )


ENH_IMMUNO_INDICATOR_LIST = [
    # Complement panel (value-gated)
    EnhImmunoIndicator.COMPLEMENT_C3_ORDERED,
    EnhImmunoIndicator.COMPLEMENT_C4_ORDERED,
    EnhImmunoIndicator.COMPLEMENT_CH50_ORDERED,
    # Autoimmunity panel (value-gated)
    EnhImmunoIndicator.ANA_ORDERED,
    EnhImmunoIndicator.RF_ORDERED,
    EnhImmunoIndicator.ANTI_CCP_ORDERED,
    EnhImmunoIndicator.ANTIPHOSPHOLIPID_ORDERED,
    # Diagnoses & treatment exposures
    EnhImmunoIndicator.NEW_RHEUM_DX,
    EnhImmunoIndicator.JAK_INHIBITOR,
    EnhImmunoIndicator.DMARD_INITIATION,
    EnhImmunoIndicator.IVIG_EXPOSURE,
    # Dropped: RHEUMATOLOGY_VISIT, IMMUNOLOGY_VISIT (utilisation proxies)
    # Cytokines & adaptive immune
    EnhImmunoIndicator.ANY_IL1B,
    EnhImmunoIndicator.CD4_COUNT,
    # Autoantibodies (value-gated)
    EnhImmunoIndicator.DSDNA_ANTIBODY,
    EnhImmunoIndicator.THYROID_AUTOANTIBODY,
]


class EnhEndoIndicator(Enum):
    """Enhanced endothelial dysfunction / thromboinflammation indicators."""

    # Specialised endothelial labs — value-gated (require result, not just ordering)
    VWF_ORDERED = IndicatorSpec(
        label="von Willebrand factor test with result",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES,
                        loinc_codes=("6014-5", "27816-8")),
        ),
        source_value_patterns=("%von willebrand%", "%vwf%"),
        require_any_value=True,
    )

    FACTOR_VIII_ORDERED = IndicatorSpec(
        label="Factor VIII activity with result",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("3209-4",)),
        ),
        source_value_patterns=("%factor viii%", "%factor 8%activity%"),
        require_any_value=True,
    )

    # REMOVED: 0.02% prevalence — too sparse for modeling
    # ADAMTS13_ORDERED = IndicatorSpec(
    #     label="ADAMTS13 activity ordered",
    #     kind=IndicatorKind.MEAS_ANY_RECORDED,
    #     table="CDMPHI.measurement",
    #     concepts_any_of=(
    #         ConceptSpec(kind=ConceptSourceKind.NAME_PATTERN, name_patterns=("%ADAMTS13%", "%ADAMTS 13%"), domain_id="Measurement"),
    #     ),
    #     source_value_patterns=("%adamts%13%", "%adamts13%"),
    # )

    # Thrombotic sub-types (arterial vs venous)
    DVT_PE = IndicatorSpec(
        label="DVT or PE diagnosis",
        kind=IndicatorKind.COND_NAME_PATTERN,
        table="CDMPHI.condition_occurrence",
        concept_name_patterns=(
            "%DEEP VEIN THROMBOSIS%",
            "%PULMONARY EMBOLISM%",
            "%PULMONARY THROMBOEMBOLISM%",
        ),
    )

    ARTERIAL_THROMBOSIS = IndicatorSpec(
        label="Arterial thrombotic event (MI, stroke)",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS,
                        ancestor_ids=(
                            312327,   # MI
                            4310996,  # Cerebral infarction / ischemic stroke
                        )),
            # 373503 TIA removed: transient ischemia (no infarction), never in
            # the original THROMBOTIC_EVENTS; keeps this a completed-event flag.
        ),
    )

    # Myocarditis / pericarditis
    MYOCARDITIS_PERICARDITIS = IndicatorSpec(
        label="Myocarditis or pericarditis diagnosis",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS,
                        ancestor_ids=(314383, 320116)),  # Myocarditis, Pericarditis
        ),
        source_value_patterns=("%myocardit%", "%pericardit%"),
    )

    # AKI / proteinuria as microvascular injury
    AKI_DIAGNOSIS = IndicatorSpec(
        label="Acute kidney injury diagnosis",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS, ancestor_ids=(197320,)),
        ),
    )

    PROTEINURIA_DX = IndicatorSpec(
        label="Proteinuria / albuminuria diagnosis or observation",
        kind=IndicatorKind.COND_NAME_PATTERN,
        table="CDMPHI.condition_occurrence",
        concept_name_patterns=(
            "%PROTEINURIA%",
            "%ALBUMINURIA%",
        ),
    )

    # POTS / orthostatic intolerance (exploratory)
    POTS_ORTHOSTATIC = IndicatorSpec(
        label="POTS or orthostatic intolerance diagnosis",
        kind=IndicatorKind.COND_NAME_PATTERN,
        table="CDMPHI.condition_occurrence",
        concept_name_patterns=(
            "%POSTURAL ORTHOSTATIC TACHYCARDIA%",
            "%ORTHOSTATIC HYPOTENSION%",
            "%ORTHOSTATIC INTOLERANCE%",
        ),
        source_value_patterns=("%pots%", "%orthostatic%", "%dysautonomia%", "%postural tachycardia%"),
    )

    # Antiplatelet (separate from anticoagulant)
    ANTIPLATELET_THERAPY = IndicatorSpec(
        label="Antiplatelet therapy in window",
        kind=IndicatorKind.DRUG_INGREDIENT_DESC,
        table="CDMPHI.drug_exposure",
        rxnorm_ingredient_names=(
            "clopidogrel", "prasugrel", "ticagrelor", "dipyridamole",
        ),
        source_value_patterns=("%clopidogrel%", "%plavix%", "%ticagrelor%", "%brilinta%"),
    )

    # Vascular imaging
    VASCULAR_IMAGING = IndicatorSpec(
        label="Vascular imaging (echo, vascular US, CTPA)",
        kind=IndicatorKind.PROC_NAME_PATTERN,
        table="CDMPHI.procedure_occurrence",
        concept_name_patterns=(
            "%ECHOCARDIOGRA%",
            "%CT PULMONARY ANGIO%",
            "%VASCULAR ULTRASOUND%",
            "%DOPPLER%VENOUS%",
        ),
    )

    # --- Literature-driven: coagulation labs ---
    PT_INR_ORDERED = IndicatorSpec(
        label="PT/INR test ordered",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES,
                        loinc_codes=("34714-6", "5902-2", "6301-6")),  # INR, PT
        ),
        source_value_patterns=("%inr%", "%prothrombin time%"),
    )

    APTT_ORDERED = IndicatorSpec(
        label="aPTT test ordered",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("3173-2",)),
        ),
        source_value_patterns=("%aptt%", "%activated partial thromboplastin%"),
    )

    # --- Literature-driven: cardiac ---
    ARRHYTHMIA_DX = IndicatorSpec(
        label="Arrhythmia diagnosis (any type)",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS,
                        ancestor_ids=(
                            313217,   # Atrial fibrillation
                            314665,   # Atrial flutter
                            316999,   # Heart disease — cardiac arrhythmia
                            4068155,  # Tachycardia (supraventricular)
                            4103295,  # Ventricular tachycardia
                        )),
        ),
        source_value_patterns=("%arrhythm%", "%atrial fib%", "%atrial flutter%"),
    )

    CARDIAC_IMAGING_ORDERED = IndicatorSpec(
        label="Cardiac imaging ordered (echo, cardiac MRI, CT coronary)",
        kind=IndicatorKind.PROC_NAME_PATTERN,
        table="CDMPHI.procedure_occurrence",
        concept_name_patterns=(
            "%ECHOCARDIOGRA%",
            "%CARDIAC MRI%",
            "%CARDIAC MAGNETIC%",
            "%CT CORONARY%",
            "%MYOCARDIAL PERFUSION%",
        ),
    )

    # REMOVED: 0% prevalence — concept name patterns match nothing
    # EXERCISE_INTOLERANCE_DX = IndicatorSpec(
    #     label="Exercise intolerance / deconditioning diagnosis",
    #     kind=IndicatorKind.COND_NAME_PATTERN,
    #     table="CDMPHI.condition_occurrence",
    #     concept_name_patterns=(
    #         "%EXERCISE INTOLERANCE%",
    #         "%DECONDITIONING%",
    #         "%REDUCED EXERCISE TOLERANCE%",
    #     ),
    #     source_value_patterns=("%exercise intolerance%", "%deconditioning%"),
    # )


ENH_ENDO_INDICATOR_LIST = [
    # Endothelial labs (value-gated)
    EnhEndoIndicator.VWF_ORDERED,
    EnhEndoIndicator.FACTOR_VIII_ORDERED,
    # Thrombotic sub-types
    EnhEndoIndicator.DVT_PE,
    EnhEndoIndicator.ARTERIAL_THROMBOSIS,
    # Cardiac / microvascular
    EnhEndoIndicator.MYOCARDITIS_PERICARDITIS,
    EnhEndoIndicator.AKI_DIAGNOSIS,
    EnhEndoIndicator.PROTEINURIA_DX,
    EnhEndoIndicator.POTS_ORTHOSTATIC,
    # Treatment exposure (kept — not workup)
    EnhEndoIndicator.ANTIPLATELET_THERAPY,
    # Dropped: PT_INR_ORDERED, APTT_ORDERED (numeric quant stream covers these)
    # Dropped: VASCULAR_IMAGING, CARDIAC_IMAGING_ORDERED (workup proxies)
    # Cardiac diagnosis (kept — directional)
    EnhEndoIndicator.ARRHYTHMIA_DX,
]


class EnhExploratoryIndicator(Enum):
    """Exploratory mechanistic indicators (mast-cell, allergic)."""

    TRYPTASE_ORDERED = IndicatorSpec(
        label="Tryptase test ordered",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(4007807,)),
        ),
        source_value_patterns=("%tryptase%",),
    )

    H1_H2_BLOCKER = IndicatorSpec(
        label="H1 or H2 blocker exposure in window",
        kind=IndicatorKind.DRUG_INGREDIENT_DESC,
        table="CDMPHI.drug_exposure",
        rxnorm_ingredient_names=(
            "cetirizine", "loratadine", "fexofenadine", "diphenhydramine",
            "hydroxyzine", "famotidine", "ranitidine",
        ),
        source_value_patterns=(
            "%cetirizine%", "%zyrtec%", "%loratadine%", "%claritin%",
            "%fexofenadine%", "%allegra%", "%diphenhydramine%", "%benadryl%",
            "%famotidine%", "%pepcid%",
        ),
    )

    LEUKOTRIENE_ANTAGONIST = IndicatorSpec(
        label="Leukotriene antagonist exposure in window",
        kind=IndicatorKind.DRUG_INGREDIENT_DESC,
        table="CDMPHI.drug_exposure",
        rxnorm_ingredient_names=("montelukast", "zafirlukast"),
        source_value_patterns=("%montelukast%", "%singulair%", "%zafirlukast%"),
    )

    URTICARIA_DX = IndicatorSpec(
        label="Urticaria or flushing diagnosis",
        kind=IndicatorKind.COND_NAME_PATTERN,
        table="CDMPHI.condition_occurrence",
        concept_name_patterns=(
            "%URTICARIA%",
            "%FLUSHING%",
            "%ANGIOEDEMA%",
        ),
    )


ENH_EXPLORATORY_INDICATOR_LIST = list(EnhExploratoryIndicator)


# =============================================================================
# 6.  CONCEPT AVAILABILITY AUDIT
# =============================================================================

def audit_concept_availability(
    cur,
    cohort_df: pd.DataFrame,
    lab_specs: Dict[str, NumericLabSpec] = None,
    indicator_lists: Dict[str, Sequence] = None,
) -> pd.DataFrame:
    """
    Audit concept availability for all proposed feature families.

    Returns a summary DataFrame with family, concept_name, patient_count, pct.
    """
    if lab_specs is None:
        lab_specs = {**NUMERIC_LAB_SPECS, **CBC_SPECS}
    if indicator_lists is None:
        indicator_lists = {
            "enh_viral": ENH_VIRAL_INDICATOR_LIST,
            "enh_immuno": ENH_IMMUNO_INDICATOR_LIST,
            "enh_endo": ENH_ENDO_INDICATOR_LIST,
            "enh_exploratory": ENH_EXPLORATORY_INDICATOR_LIST,
        }

    n_total = cohort_df["person_id"].nunique()
    rows = []

    # Audit numeric labs
    for lab_name, spec in lab_specs.items():
        cids = _resolve_lab_concept_ids(cur, spec)
        if not cids:
            rows.append({"family": "numeric_lab", "name": lab_name, "n_patients": 0, "pct": 0.0})
            continue
        sql = f"""
        SELECT COUNT(DISTINCT m.person_id) AS cnt
        FROM {CDM}.measurement m
        WHERE m.measurement_concept_id IN ({_ints_to_sql_in(cids)})
          AND m.value_as_number IS NOT NULL
          AND m.person_id IN (SELECT person_id FROM #cohort_temp)
        """
        # Need cohort temp table
        try:
            df = run_query(cur, sql)
            cnt = int(df.iloc[0, 0]) if not df.empty else 0
        except Exception:
            cnt = 0
        rows.append({
            "family": "numeric_lab",
            "name": lab_name,
            "n_patients": cnt,
            "pct": round(100 * cnt / n_total, 2) if n_total > 0 else 0,
        })

    # Audit binary indicators (check if concepts resolve)
    for family_name, indicators in indicator_lists.items():
        for ind in indicators:
            spec = ind.value
            if spec.constant_zero:
                rows.append({"family": family_name, "name": ind.name, "n_patients": 0, "pct": 0.0})
                continue
            rows.append({
                "family": family_name,
                "name": ind.name,
                "n_patients": -1,  # -1 = not audited (would need per-indicator SQL)
                "pct": -1,
            })

    return pd.DataFrame(rows)


# =============================================================================
# 7.  MODEL CONFIG BUILDER
# =============================================================================

def _route_labs_to_mechanism(cols: List[str]) -> Dict[str, List[str]]:
    """Route numeric-lab columns into mechanistic groups (exclusive).

    Mapping rationale:
      Immuno labs – inflammatory / immune markers:
        CRP, ESR, ferritin, albumin (acute-phase), LDH, lactate
      Endo labs – coagulation / cardiac / renal markers:
        D-dimer, fibrinogen, PT/INR, aPTT, troponin, NT-proBNP,
        creatinine, platelets
    """
    IMMUNO_LABS = {"crp", "esr", "ferritin", "albumin", "ldh", "lactate"}
    ENDO_LABS = {"ddimer", "fibrinogen", "pt_inr", "aptt", "troponin",
                 "nt_probnp", "creatinine", "platelets"}
    # AST/ALT are general organ injury – route to endo (liver / multi-organ)
    ENDO_LABS.update({"ast", "alt"})

    immuno, endo = [], []
    for c in cols:
        # Extract lab name: f_lab_{window}_{labname}_{suffix}
        parts = c.replace("f_lab_", "").split("_")
        # Skip the window token (e.g. w0_90 → two parts)
        lab_name = "_".join(parts[2:]).rsplit("_", 1)[0] if len(parts) > 3 else parts[-1]
        # Normalize: check all known prefixes
        matched = False
        for name in IMMUNO_LABS:
            if name in c:
                immuno.append(c)
                matched = True
                break
        if not matched:
            for name in ENDO_LABS:
                if name in c:
                    endo.append(c)
                    matched = True
                    break
        if not matched:
            # Default unrecognised labs to endo
            endo.append(c)
    return {"immuno": immuno, "endo": endo}


def _route_cbc_to_mechanism(cols: List[str]) -> Dict[str, List[str]]:
    """Route CBC-index columns into mechanistic groups (exclusive).

    Mapping rationale:
      Immuno CBC – immune-cell ratios & counts:
        NLR, SII, persistent_lymphopenia, monocytosis, eosinophilia
      Endo CBC – platelet / RBC markers:
        PLR, RDW
    """
    ENDO_CBC = {"plr", "rdw"}
    immuno, endo = [], []
    for c in cols:
        if any(k in c for k in ENDO_CBC):
            endo.append(c)
        else:
            immuno.append(c)
    return {"immuno": immuno, "endo": endo}


def _route_composites_to_mechanism(cols: List[str]) -> Dict[str, List[str]]:
    """Route composite columns into mechanistic groups (exclusive).

    Mapping rationale:
      Viral composites – GI-persistence cluster (viral persistence → gut):
        f_comp_gi_*
      Immuno composites – complement + inflammation clusters:
        f_comp_*_complement_*, f_comp_*_complement_crp_ddimer
      Endo composites – organ-injury composites:
        f_comp_*_cardiac_injury, f_comp_*_renal_injury

    Temporal-divergence composites (f_comp_td_*):
      Viral:  td_post_acute_respiratory
      Immuno: td_crp_esr_elevated, td_late_autoimmune, td_persistent_cytokine
      Endo:   td_sustained_coag, td_emerging_dysautonomia, td_unresolved_thrombosis
      All-only: td_multisystem_late (not routed to a single mechanism)
    """
    viral, immuno, endo = [], [], []
    for c in cols:
        # --- Temporal-divergence composites ---
        if "_td_post_acute_respiratory" in c:
            viral.append(c)
        elif "_td_crp_esr_elevated" in c or "_td_late_autoimmune" in c or "_td_persistent_cytokine" in c:
            immuno.append(c)
        elif "_td_sustained_coag" in c or "_td_emerging_dysautonomia" in c or "_td_unresolved_thrombosis" in c:
            endo.append(c)
        elif "_td_multisystem_late" in c:
            # Cross-mechanism score (count of active mechanism families).
            # Not assigned to any single mechanism — only appears in
            # mechsig_all / mechsig_full via the "all" bucket.
            pass
        # --- Original composites ---
        elif "_gi_" in c:
            viral.append(c)
        elif "complement" in c:
            immuno.append(c)
        elif "cardiac" in c or "renal" in c:
            endo.append(c)
        else:
            # Unrecognised → endo
            endo.append(c)
    return {"viral": viral, "immuno": immuno, "endo": endo}


def build_enhanced_model_configs(feature_families: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Build model configs preserving the existing hierarchy plus enhanced variants.

    Numeric labs, CBC indices, and composites are distributed into viral /
    immuno / endo groups so that every feature belongs to exactly one
    mechanistic config (exclusive assignment).

    Expected families in feature_families:
        antony_comorbidities, antony_symptoms, antony_drugs,
        antony_demographics, antony_treatment, ext_features,
        mechsig_viral, mechsig_immuno, mechsig_endo, mechsig_exploratory,
        numeric_labs, cbc_indices, lab_trajectories, composites, temporal_trends

    Note: legacy `mechsig_*_orig` / `mechsig_*_enh` family keys are still
    accepted (merged into the corresponding plain `mechsig_<m>` family) for
    backwards compatibility with previously cached feature-family maps.

    Returns dict config_name -> list of feature column names.
    """
    # Antony baseline
    antony_families = [
        "antony_comorbidities", "antony_symptoms", "antony_drugs",
        "antony_demographics", "antony_treatment",
    ]
    antony_cols = []
    for fam in antony_families:
        antony_cols.extend(feature_families.get(fam, []))

    ext_cols = feature_families.get("ext_features", [])

    # Step 5: engagement controls are forced into ALL configs (never Boruta-droppable).
    # These soak up the ascertainment pathway so that mechanism features are not
    # rewarded for encoding "patient had MSHS contact".
    eng_cols = feature_families.get("engagement_controls", [])

    baseline_ext = antony_cols + ext_cols + eng_cols

    # Mechanistic binary indicators (combined v1 + v2; previously split across
    # `mechsig_<m>_orig` and `mechsig_<m>_enh` families).  We accept both the
    # new combined family key and the legacy split keys, taking the union and
    # de-duplicating while preserving order.
    def _combined_mech(mech: str) -> List[str]:
        seen: set = set()
        out: List[str] = []
        for key in (f"mechsig_{mech}", f"mechsig_{mech}_orig", f"mechsig_{mech}_enh"):
            for col in feature_families.get(key, []):
                if col not in seen:
                    seen.add(col)
                    out.append(col)
        return out

    viral = _combined_mech("viral")
    immuno = _combined_mech("immuno")
    endo = _combined_mech("endo")
    exploratory = feature_families.get("mechsig_exploratory", [])
    all_mech = viral + immuno + endo + exploratory

    # --- Distribute numeric / derived features into mechanistic groups ---
    numeric_labs = feature_families.get("numeric_labs", [])
    cbc_indices = feature_families.get("cbc_indices", [])
    lab_traj = feature_families.get("lab_trajectories", [])
    composites = feature_families.get("composites", [])
    temporal_trends = feature_families.get("temporal_trends", [])

    lab_routed = _route_labs_to_mechanism(numeric_labs + lab_traj)
    cbc_routed = _route_cbc_to_mechanism(cbc_indices)
    comp_routed = _route_composites_to_mechanism(composites)

    # Aggregate numeric additions per mechanism
    quant_viral = comp_routed["viral"]
    quant_immuno = lab_routed["immuno"] + cbc_routed["immuno"] + comp_routed["immuno"]
    quant_endo = lab_routed["endo"] + cbc_routed["endo"] + comp_routed["endo"]
    all_quant = quant_viral + quant_immuno + quant_endo

    # Cross-mechanism composites (not in any single-mechanism config)
    cross_mech = [c for c in composites if "_td_multisystem_late" in c]

    # Cross-window summaries
    xw_summaries = feature_families.get("cross_window_summaries", [])

    configs = {
        # Preserved hierarchy
        "baseline_antony": antony_cols,
        "baseline_ext": baseline_ext,

        # Mechanistic configs: binary indicators (v1 + v2 union) + quantitative
        # features routed to that mechanism.  These supersede the previous
        # `mechsig_<m>_orig` / `mechsig_<m>_enh` pair (every column from either
        # legacy config is included here).
        "mechsig_viral": baseline_ext + viral + quant_viral,
        "mechsig_immuno": baseline_ext + immuno + quant_immuno,
        "mechsig_endo": baseline_ext + endo + quant_endo,
        "mechsig_all": baseline_ext + all_mech + all_quant + cross_mech,

        # Temporal-trend additions derived from sequential windows.
        "mechsig_temporal": baseline_ext + temporal_trends,

        # Everything
        "mechsig_full": baseline_ext + all_mech + all_quant + cross_mech + xw_summaries + temporal_trends,
    }

    # --- Step 4: td-excluded sensitivity configs ---
    # For each mechsig config, add a *_no_td variant that strips all f_comp_td_*
    # columns.  This quantifies how much of the mech lift is post-acute leakage
    # (td_* features) vs genuine acute-window mechanism signal.
    _td_prefix = "f_comp_td_"
    _strip_td = lambda cols: [c for c in cols if not c.startswith(_td_prefix)]
    configs["mechsig_viral_no_td"]  = _strip_td(configs["mechsig_viral"])
    configs["mechsig_immuno_no_td"] = _strip_td(configs["mechsig_immuno"])
    configs["mechsig_endo_no_td"]   = _strip_td(configs["mechsig_endo"])
    configs["mechsig_all_no_td"]    = _strip_td(configs["mechsig_all"])
    configs["baseline_ext_no_td"]   = _strip_td(configs["baseline_ext"])

    # --- Step 5: engagement-control decomposition configs ---
    # These variants enable a clean three-way attribution of predictive lift
    # between (a) healthcare-engagement controls, (b) extended demographics /
    # severity / serology, and (c) mechanistic features.
    #
    #   f_eng_* columns (6): n_encounters_pre_index_1y, has_pcp_pre_index,
    #       insurance_{commercial, medicaid, medicare, self_pay}
    #   See combined_features.py::extract_engagement_features for definitions
    #   and ascertainment-bias rationale.
    #
    # Naming convention:
    #   baseline_antony_eng  = Antony A-E + f_eng_*   (engagement-only lift)
    #   <name>_no_eng        = <name> with f_eng_* stripped  (sensitivity)
    #
    # Decomposition path (engagement-adjusted interpretation):
    #   baseline_antony -> baseline_antony_eng -> baseline_ext -> mechsig_all
    #     Delta_1 = engagement-control lift
    #     Delta_2 = extended-demographics lift (net of engagement)
    #     Delta_3 = mechanism lift (net of engagement + demographics)
    #
    # Sensitivity: compare each <name> vs <name>_no_eng to check whether
    # mechanism lift survives without ascertainment adjustment.
    _eng_prefix = "f_eng_"
    _strip_eng = lambda cols: [c for c in cols if not c.startswith(_eng_prefix)]

    # Antony A-E + engagement only (no extended demographics, no mechanism).
    configs["baseline_antony_eng"]         = antony_cols + eng_cols

    # Engagement-stripped sensitivity variants.
    configs["baseline_ext_no_eng"]         = _strip_eng(configs["baseline_ext"])
    configs["mechsig_viral_no_eng"]    = _strip_eng(configs["mechsig_viral"])
    configs["mechsig_immuno_no_eng"]   = _strip_eng(configs["mechsig_immuno"])
    configs["mechsig_endo_no_eng"]     = _strip_eng(configs["mechsig_endo"])
    configs["mechsig_all_no_eng"]      = _strip_eng(configs["mechsig_all"])

    # Log routing summary
    print("\nQuantitative feature routing:")
    print(f"  Labs   → immuno: {len(lab_routed['immuno']):3d}  endo: {len(lab_routed['endo']):3d}")
    print(f"  CBC    → immuno: {len(cbc_routed['immuno']):3d}  endo: {len(cbc_routed['endo']):3d}")
    print(f"  Comp   → viral:  {len(comp_routed['viral']):3d}  immuno: {len(comp_routed['immuno']):3d}  endo: {len(comp_routed['endo']):3d}")

    print("\nEnhanced model configurations:")
    for name, cols in configs.items():
        print(f"  {name:30s}: {len(cols):4d} features")

    return configs


# =============================================================================
# 8.  FEATURE MATRIX ASSEMBLY
# =============================================================================

def assemble_enhanced_feature_matrix(
    baseline_df: pd.DataFrame,
    baseline_families: Dict[str, List[str]],
    windowed_binary: Dict[str, pd.DataFrame],
    windowed_labs: Dict[str, pd.DataFrame],
    windowed_cbc: Dict[str, pd.DataFrame],
    trajectory_df: Optional[pd.DataFrame] = None,
    temporal_trend_df: Optional[pd.DataFrame] = None,
    composite_df: Optional[pd.DataFrame] = None,
    temporal_divergence_df: Optional[pd.DataFrame] = None,
    xw_summary_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    original_indicators: Optional[Dict[str, Sequence]] = None,
    enhanced_indicators: Optional[Dict[str, Sequence]] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Assemble the complete enhanced feature matrix from all components.

    Parameters
    ----------
    baseline_df : DataFrame from build_combined_feature_matrix (include_mechanistic=False)
    baseline_families : feature_families dict from that call
    windowed_binary : dict win_label -> binary signal DF (all mechanisms combined)
    windowed_labs : dict win_label -> numeric lab DF
    windowed_cbc : dict win_label -> CBC index DF
    trajectory_df : cross-window trajectory features (optional, can be None)
    temporal_trend_df : temporal rise/fall binaries from sequential windows (optional)
    composite_df : composite features (optional, can be None)
    temporal_divergence_df : temporal-divergence composites from
        build_temporal_divergence_composites (optional, can be None)
    xw_summary_dfs : dict mechanism -> cross-window summary DF (optional, can be None)
    original_indicators : dict mechanism -> list of original indicator enums (optional)
    enhanced_indicators : dict mechanism -> list of enhanced indicator enums (optional)

    Returns
    -------
    (feature_df, feature_families) ready for modeling
    """
    result = baseline_df.copy()
    ff = dict(baseline_families)

    # Merge windowed binary signals
    all_mech_cols: List[str] = []
    for wl, bdf in windowed_binary.items():
        result = result.merge(bdf, on="person_id", how="left")
        new_cols = [c for c in bdf.columns if c.startswith("f_ind_")]
        for c in new_cols:
            result[c] = result[c].fillna(0).astype("int8")
        all_mech_cols.extend(new_cols)

    # Partition binary indicator columns by mechanism token.  Both the legacy
    # original-set token (e.g. `_viral_`) and the v2-enhanced token
    # (e.g. `_enh_viral_`) are matched so every indicator column is preserved.
    # `original_indicators` / `enhanced_indicators` parameters are accepted
    # for backwards compatibility but no longer used (the v1/v2 distinction
    # is collapsed into a single combined family per mechanism).
    ff["mechsig_viral"]  = [c for c in all_mech_cols if "_viral_"  in c or "_enh_viral_"  in c]
    ff["mechsig_immuno"] = [c for c in all_mech_cols if "_immuno_" in c or "_enh_immuno_" in c]
    ff["mechsig_endo"]   = [c for c in all_mech_cols if "_endo_"   in c or "_enh_endo_"   in c]
    ff["mechsig_exploratory"] = [c for c in all_mech_cols if "_exploratory_" in c or "_expl_" in c]

    # Sanity: every binary indicator column must be routed into exactly one
    # mechanism family (or the exploratory bucket).  Fail loudly if any are
    # unrouted -- that would silently lose features at training time.
    routed = set(ff["mechsig_viral"]) | set(ff["mechsig_immuno"]) \
             | set(ff["mechsig_endo"]) | set(ff["mechsig_exploratory"])
    unrouted = [c for c in all_mech_cols if c not in routed]
    if unrouted:
        raise AssertionError(
            f"assemble_enhanced_feature_matrix: {len(unrouted)} binary indicator "
            f"columns are not routed into any mechanism family: {unrouted[:10]}"
        )

    # Merge numeric labs
    all_lab_cols = []
    for wl, ldf in windowed_labs.items():
        result = result.merge(ldf, on="person_id", how="left")
        new_cols = [c for c in ldf.columns if c.startswith("f_lab_")]
        all_lab_cols.extend(new_cols)
    ff["numeric_labs"] = all_lab_cols

    # Merge CBC indices
    all_cbc_cols = []
    for wl, cdf in windowed_cbc.items():
        result = result.merge(cdf, on="person_id", how="left")
        new_cols = [c for c in cdf.columns if c.startswith("f_cbc_")]
        all_cbc_cols.extend(new_cols)
    ff["cbc_indices"] = all_cbc_cols

    # Merge trajectories
    if trajectory_df is not None and not trajectory_df.empty:
        result = result.merge(trajectory_df, on="person_id", how="left")
        traj_cols = [c for c in trajectory_df.columns if c.startswith("f_lab_traj_")]
        ff["lab_trajectories"] = traj_cols
    else:
        ff["lab_trajectories"] = []

    # Merge temporal trend binaries
    if temporal_trend_df is not None and not temporal_trend_df.empty:
        result = result.merge(temporal_trend_df, on="person_id", how="left")
        trend_cols = [c for c in temporal_trend_df.columns if c.startswith("f_temp_")]
        ff["temporal_trends"] = trend_cols
        for c in trend_cols:
            if "_delta_" in c:
                continue  # continuous delta — leave as float (NaN handled downstream)
            result[c] = result[c].fillna(0).astype("int8")
    else:
        ff["temporal_trends"] = []

    # Merge composites
    if composite_df is not None and not composite_df.empty:
        result = result.merge(composite_df, on="person_id", how="left")
        comp_cols = [c for c in composite_df.columns if c.startswith("f_comp_")]
        ff["composites"] = comp_cols
    else:
        ff["composites"] = []

    # Merge temporal-divergence composites (f_comp_td_*)
    if temporal_divergence_df is not None and not temporal_divergence_df.empty:
        result = result.merge(temporal_divergence_df, on="person_id", how="left")
        td_cols = [c for c in temporal_divergence_df.columns if c.startswith("f_comp_td_")]
        for c in td_cols:
            result[c] = result[c].fillna(0).astype("int8")
        # Append to composites family so they get routed by _route_composites_to_mechanism
        ff["composites"] = ff.get("composites", []) + td_cols

    # Merge cross-window summaries
    all_xw_cols = []
    for mech, xw_df in (xw_summary_dfs or {}).items():
        result = result.merge(xw_df, on="person_id", how="left")
        new_cols = [c for c in xw_df.columns if c.startswith("f_ind_") and "_xw_" in c]
        all_xw_cols.extend(new_cols)
        for c in new_cols:
            result[c] = result[c].fillna(0).astype("int8")
    ff["cross_window_summaries"] = all_xw_cols

    # Fill NaN for numeric features (missing-indicator pattern)
    for col in all_lab_cols + all_cbc_cols:
        if col.endswith("_measured") or col.endswith("_abnormal_any") or col.endswith("_abnormal_count"):
            result[col] = result[col].fillna(0)
        elif col in result.columns:
            # Add missing indicator for numeric columns
            miss_col = f"{col}_miss"
            if miss_col not in result.columns:
                result[miss_col] = result[col].isna().astype("int8")
            # Impute with median
            median_val = result[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            result[col] = result[col].fillna(median_val)

    # Fill NaN for continuous composite features (severity scores, ratios)
    continuous_comp_suffixes = ("_score", "_ratio")
    for col in ff.get("composites", []):
        if col in result.columns and any(col.endswith(s) for s in continuous_comp_suffixes):
            miss_col = f"{col}_miss"
            if miss_col not in result.columns:
                result[miss_col] = result[col].isna().astype("int8")
            median_val = result[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            result[col] = result[col].fillna(median_val)

    # Fill NaN for continuous temporal trend delta features
    for col in ff.get("temporal_trends", []):
        if col in result.columns and "_delta_" in col:
            miss_col = f"{col}_miss"
            if miss_col not in result.columns:
                result[miss_col] = result[col].isna().astype("int8")
            median_val = result[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            result[col] = result[col].fillna(median_val)

    # Summary
    total = 0
    print("\n" + "=" * 80)
    print("ENHANCED FEATURE MATRIX SUMMARY")
    print("=" * 80)
    print(f"  Patients:       {len(result):,}")
    for family, cols in ff.items():
        if cols:
            print(f"    {family:35s}: {len(cols):4d} features")
            total += len(cols)
    print(f"    {'TOTAL':35s}: {total:4d} features")
    print("=" * 80)

    return result, ff

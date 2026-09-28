"""
OMOP2OBO mapping loader and HPO symptom integration.

Loads the N3C-formatted OMOP2OBO Condition Occurrence Mappings (Callahan et al.)
and provides:
  - concept_id → HPO term lookup for symptom feature extraction
  - HPO-based acute symptom feature family (Family B alternative)
  - Coverage diagnostics for mapping quality assessment
  - Mechanistic signal HPO cross-check (audit utility)

Data source:
  Callahan, T. J. & N3C OMOP to OBO Working Group (2022).
  N3C-Formatted OMOP2OBO Mappings (v2.0.0).
  Zenodo. https://doi.org/10.5281/zenodo.7255922

  Place the expression_items CSV in data/omop2obo/:
    OMOP2OBO_v2.0.0_N3C_Enclave_CSV_concept_set_expression_items.csv

Pin: v2.0.0 (Oct 2022) — aligns with Antony 2023 publication timeline.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Set

import pandas as pd


# =============================================================================
# Constants
# =============================================================================

OMOP2OBO_VERSION = "v2.0.0"

_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MAPPING_PATH = (
    _MODULE_DIR
    / "data"
    / "omop2obo"
    / "OMOP2OBO_v2.0.0_N3C_Enclave_CSV_concept_set_expression_items.csv"
)

# Mapping quality categories — ordered from highest to lowest confidence
# (names as they appear in the N3C-formatted CSV)
QUALITY_TIERS = (
    "Automatic One-to-One Concept",
    "Automatic One-to-Many Concept",
    "Automatic One-to-One Ancestor",
    "Automatic One-to-Many Ancestor",
    "Manual One-to-One Concept",
    "Manual One-to-Many Concept",
    "Cosine Similarity One-to-One Concept",
)

# Trusted categories for replication (default filter — excludes cosine similarity)
TRUSTED_CATEGORIES = {
    "Automatic One-to-One Concept",
    "Automatic One-to-Many Concept",
    "Automatic One-to-One Ancestor",
    "Automatic One-to-Many Ancestor",
    "Manual One-to-One Concept",
    "Manual One-to-Many Concept",
}

from pasc.config.omop import CDM_SCHEMA  # OMOP schema name; set OMOP_CDM_SCHEMA to override


# =============================================================================
# Antony's 14 core symptom HPO terms
# =============================================================================

# These map each Antony symptom feature to its HPO curie.
# The exact HPO terms correspond to the SNOMED ancestors in
# antony_features.SYMPTOM_ANCESTORS, but resolved through OMOP2OBO.
HPO_SYMPTOM_TERMS = {
    "f_sym_hpo_fever":         "HP:0001945",  # Fever
    "f_sym_hpo_cough":         "HP:0012735",  # Cough
    "f_sym_hpo_fatigue":       "HP:0012378",  # Fatigue
    "f_sym_hpo_dyspnea":       "HP:0002094",  # Dyspnea
    "f_sym_hpo_myalgia":       "HP:0003326",  # Myalgia
    "f_sym_hpo_headache":      "HP:0002315",  # Headache
    "f_sym_hpo_anosmia":       "HP:0000458",  # Anosmia
    "f_sym_hpo_ageusia":       "HP:0000224",  # Ageusia
    "f_sym_hpo_diarrhea":      "HP:0002014",  # Diarrhea
    "f_sym_hpo_nausea":        "HP:0002018",  # Nausea
    "f_sym_hpo_sore_throat":   "HP:0033050",  # Pharyngitis / sore throat
    "f_sym_hpo_chest_pain":    "HP:0100749",  # Chest pain
    "f_sym_hpo_arthralgia":    "HP:0002829",  # Arthralgia
    "f_sym_hpo_vomiting":      "HP:0002013",  # Vomiting
}

# Five extra symptoms beyond Antony's original 14
# (gated by strict_antony=True, same as SNOMED version)
HPO_EXTRA_SYMPTOM_TERMS = {
    "f_sym_hpo_abdominal_pain": "HP:0002027",  # Abdominal pain
    "f_sym_hpo_dizziness":      "HP:0002321",  # Dizziness
    "f_sym_hpo_malaise":        "HP:0033834",  # Malaise
    "f_sym_hpo_rhinorrhea":     "HP:0031417",  # Rhinorrhea
    "f_sym_hpo_congestion":     "HP:0001742",  # Nasal congestion
}

# Combined for non-strict mode
HPO_ALL_SYMPTOM_TERMS = {**HPO_SYMPTOM_TERMS, **HPO_EXTRA_SYMPTOM_TERMS}


# =============================================================================
# SNOMED concept_id → HPO ancestor concept_id mapping
# =============================================================================
#
# For each HPO term in our symptom set, these are the SNOMED ancestor concept
# IDs whose descendants should map to that HPO term.  This is the bridge
# between SNOMED concept_ancestor (what we query from the CDM) and HPO terms
# (what Antony's pipeline conceptually uses via OMOP2OBO).
#
# These come from the same SYMPTOM_ANCESTORS dict in antony_features.py.
SNOMED_ANCESTOR_TO_HPO = {
    437663:  "HP:0001945",   # Fever
    254761:  "HP:0012735",   # Cough
    4223659: "HP:0012378",   # Fatigue
    312437:  "HP:0002094",   # Dyspnea
    442752:  "HP:0003326",   # Myalgia
    378253:  "HP:0002315",   # Headache
    4185711: "HP:0000458",   # Anosmia
    4289517: "HP:0000224",   # Ageusia
    196523:  "HP:0002014",   # Diarrhea
    31967:   "HP:0002018",   # Nausea
    4147326: "HP:0033050",   # Sore throat
    77670:   "HP:0100749",   # Chest pain
    200219:  "HP:0002027",   # Abdominal pain
    4276172: "HP:0031417",   # Rhinorrhea (was: 4100065 = "Disease caused by Coronaviridae")
    77074:   "HP:0002829",   # Arthralgia
    441408:  "HP:0002013",   # Vomiting
    4272240: "HP:0033834",   # Malaise
    4223938: "HP:0002321",   # Dizziness
    4195085: "HP:0001742",   # Nasal congestion
}


# =============================================================================
# Mapping Loader
# =============================================================================

def load_omop2obo_condition_mappings(
    path: Optional[str] = None,
    quality_filter: Optional[set] = None,
) -> pd.DataFrame:
    """
    Load the N3C-formatted OMOP2OBO condition occurrence mappings.

    Args:
        path: Path to the expression_items CSV. Defaults to
            data/omop2obo/OMOP2OBO_v2.0.0_N3C_...expression_items.csv
        quality_filter: Set of mapping_category strings to keep.
            Defaults to TRUSTED_CATEGORIES (Automatic Exact + Manual).
            Pass None to keep all rows (including unmapped).

    Returns:
        DataFrame with columns:
            concept_id  (int)   — OMOP standard concept ID
            hpo_id      (str)   — HPO curie, e.g. "HP:0001945"
            hpo_label   (str)   — HPO term label
            mapping_category (str) — mapping quality tier
    """
    if path is None:
        path = str(DEFAULT_MAPPING_PATH)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"OMOP2OBO mapping file not found: {path}\n"
            f"Download from https://doi.org/10.5281/zenodo.7255922\n"
            f"Place in data/omop2obo/"
        )

    print(f"Loading OMOP2OBO {OMOP2OBO_VERSION} mappings from {path}...")
    raw = pd.read_csv(
        path,
        usecols=["concept_id", "ontology_id", "ontology_label", "mapping_category"],
        dtype={"concept_id": int, "ontology_id": str, "ontology_label": str,
               "mapping_category": str},
    )
    print(f"  Raw rows: {len(raw):,}")

    # Apply quality filter before exploding (faster)
    if quality_filter is not None:
        n_before = len(raw)
        raw = raw[raw["mapping_category"].isin(quality_filter)].copy()
        print(f"  Quality filter → {len(raw):,} rows (dropped {n_before - len(raw):,})")

    # Explode pipe-separated ontology_id into individual rows.
    # Format: "HP_0001903 | HP_0002664 | ..." with " | " separator.
    # Labels may have mismatched pipe counts, so we explode IDs only and
    # reconstruct labels via a secondary join.
    raw = raw.copy()
    raw["_row_idx"] = range(len(raw))
    raw["ontology_id"] = raw["ontology_id"].str.split(r"\s*\|\s*")
    exploded = raw.explode("ontology_id", ignore_index=True)

    # Normalise HP_NNNNNNN → HP:NNNNNNN and keep only HPO terms
    exploded["ontology_id"] = exploded["ontology_id"].str.strip()
    hpo_mask = exploded["ontology_id"].str.startswith("HP_", na=False)
    exploded = exploded[hpo_mask].copy()
    exploded["ontology_id"] = "HP:" + exploded["ontology_id"].str[3:]

    # Build label lookup from the original pipe-separated ontology_label.
    # Best-effort: if counts mismatch, leave label as "" for extra IDs.
    label_map: Dict[str, str] = {}
    for _, r in raw.iterrows():
        labels = [s.strip() for s in str(r["ontology_label"]).split("|")]
        ids = [s.strip() for s in str(r["ontology_id"]).split("|")]  # already list
        for i, oid_list_item in enumerate(ids if isinstance(ids, list) else [ids]):
            oid = oid_list_item.strip() if isinstance(oid_list_item, str) else str(oid_list_item)
            if oid.startswith("HP_"):
                hpo = "HP:" + oid[3:]
                label_map.setdefault(hpo, labels[i] if i < len(labels) else "")

    exploded["hpo_label"] = exploded["ontology_id"].map(
        lambda x: label_map.get(x, ""))

    df = exploded.rename(columns={
        "ontology_id": "hpo_id",
    })[["concept_id", "hpo_id", "hpo_label", "mapping_category"]].copy()

    print(f"  {len(df):,} HPO mappings (exploded from {len(raw):,} rows)")
    print(f"  Covering {df['concept_id'].nunique():,} unique OMOP concept IDs")
    print(f"  Mapping to {df['hpo_id'].nunique():,} unique HPO terms")
    return df


def build_concept_to_hpo_lookup(
    mapping_df: pd.DataFrame,
) -> Dict[int, Set[str]]:
    """
    Build a concept_id → set of HPO term IDs lookup.

    Args:
        mapping_df: Output of load_omop2obo_condition_mappings().

    Returns:
        Dict mapping OMOP concept_id (int) → set of HPO IDs (str).
    """
    lookup: Dict[int, Set[str]] = defaultdict(set)
    for cid, hpo in zip(mapping_df["concept_id"], mapping_df["hpo_id"]):
        lookup[int(cid)].add(hpo)
    return dict(lookup)


def build_hpo_to_concept_lookup(
    mapping_df: pd.DataFrame,
) -> Dict[str, Set[int]]:
    """
    Build HPO term ID → set of OMOP concept_ids (inverse lookup).

    Args:
        mapping_df: Output of load_omop2obo_condition_mappings().

    Returns:
        Dict mapping HPO ID (str) → set of OMOP concept_ids (int).
    """
    lookup: Dict[str, Set[int]] = defaultdict(set)
    for cid, hpo in zip(mapping_df["concept_id"], mapping_df["hpo_id"]):
        lookup[hpo].add(int(cid))
    return dict(lookup)


# =============================================================================
# Coverage Diagnostics
# =============================================================================

def get_coverage_stats(
    concept_ids: Sequence[int],
    concept_to_hpo: Dict[int, Set[str]],
) -> dict:
    """
    Compute coverage statistics for a set of OMOP concept IDs.

    Args:
        concept_ids: Iterable of OMOP condition concept IDs from cohort data.
        concept_to_hpo: Lookup from build_concept_to_hpo_lookup().

    Returns:
        Dict with keys: n_total, n_mapped, n_unmapped, pct_mapped,
        unmapped_ids (set).
    """
    unique_ids = set(int(c) for c in concept_ids)
    mapped = {c for c in unique_ids if c in concept_to_hpo}
    unmapped = unique_ids - mapped
    n = len(unique_ids)
    return {
        "n_total": n,
        "n_mapped": len(mapped),
        "n_unmapped": len(unmapped),
        "pct_mapped": len(mapped) / n * 100 if n > 0 else 0.0,
        "unmapped_ids": unmapped,
    }


def emit_hpo_coverage_report(
    cur,
    concept_to_hpo: Dict[int, Set[str]],
    output_dir: str = None,
) -> pd.DataFrame:
    """
    Generate a coverage report for the cohort's acute-window conditions.

    Queries all distinct condition_concept_id values from the cohort's acute
    window, checks mapping coverage, and ranks unmapped concepts by frequency.

    Args:
        cur: Database cursor (expects #antony_cohort temp table).
        concept_to_hpo: Lookup from build_concept_to_hpo_lookup().
        output_dir: Directory for the output CSV.

    Returns:
        DataFrame with unmapped concepts ranked by frequency.
    """
    print("\n[HPO Coverage] Generating mapping coverage report...")

    sql = f"""
    SELECT
        co.condition_concept_id,
        c2.concept_name,
        COUNT(DISTINCT co.person_id) AS n_patients,
        COUNT(*) AS n_records
    FROM "#antony_cohort" coh
    JOIN {CDM_SCHEMA}.condition_occurrence co
      ON co.person_id = coh.person_id
     AND co.condition_start_date BETWEEN coh.acute_start AND coh.acute_end
    LEFT JOIN {CDM_SCHEMA}.concept c2
      ON c2.concept_id = co.condition_concept_id
    GROUP BY co.condition_concept_id, c2.concept_name
    ORDER BY n_records DESC
    """

    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0].lower() for d in cur.description]
    df = pd.DataFrame(rows, columns=cols)

    if df.empty:
        print("  WARNING: No condition records found in acute window.")
        return df

    # Tag each concept as mapped or not
    df["is_mapped"] = df["condition_concept_id"].apply(
        lambda x: int(x) in concept_to_hpo
    )

    total_records = df["n_records"].sum()
    mapped_records = df.loc[df["is_mapped"], "n_records"].sum()
    total_concepts = len(df)
    mapped_concepts = df["is_mapped"].sum()

    total_patients = df["n_patients"].sum()
    mapped_patients = df.loc[df["is_mapped"], "n_patients"].sum()

    print(f"  Concepts:  {mapped_concepts}/{total_concepts} "
          f"({mapped_concepts/total_concepts*100:.1f}%) mapped")
    print(f"  Records:   {mapped_records:,}/{total_records:,} "
          f"({mapped_records/total_records*100:.1f}%) mapped")
    print(f"  Patients:  {mapped_patients:,}/{total_patients:,} "
          f"({mapped_patients/total_patients*100:.1f}%) with ≥1 mapped condition")

    # Unmapped concepts ranked by frequency
    unmapped = df[~df["is_mapped"]].sort_values("n_records", ascending=False)

    # Save
    if output_dir is None:
        from pasc.config.paths import MAIN_RESULTS_DIR
        output_dir = str(MAIN_RESULTS_DIR)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "hpo_coverage_report.csv")
    df.to_csv(out_path, index=False)
    print(f"  Full report saved to {out_path}")

    if len(unmapped) > 0:
        top_path = os.path.join(output_dir, "hpo_unmapped_top50.csv")
        unmapped.head(50).to_csv(top_path, index=False)
        print(f"  Top-50 unmapped concepts saved to {top_path}")

    return unmapped


# =============================================================================
# Mechsig HPO Cross-Check (Audit Utility)
# =============================================================================

# HPO subtrees for each mechanism cluster.
# These are high-level HPO ancestors; any HPO term that is a descendant of
# these terms is considered "in-subtree" for that mechanism.
MECHANISM_HPO_SUBTREES = {
    "viral": {
        "HP:0002721",  # Immunodeficiency (viral reactivation context)
        "HP:0031035",  # Chronic infection
        "HP:0002633",  # Vasculitis (viral-mediated)
    },
    "immuno": {
        "HP:0002960",  # Autoimmunity
        "HP:0012649",  # Increased inflammatory response
        "HP:0001875",  # Neutropenia
        "HP:0011893",  # Abnormal leukocyte count
    },
    "endo": {
        "HP:0001907",  # Thromboembolism
        "HP:0003256",  # Coagulopathy
        "HP:0001626",  # Abnormality of the cardiovascular system
    },
}


def cross_check_mechsig_hpo_routing(
    indicator_list: Sequence,
    concept_to_hpo: Dict[int, Set[str]],
    mechanism_name: str = "",
) -> pd.DataFrame:
    """
    Cross-check mechanistic signal indicators against HPO mappings.

    For each OMOP concept used in the given indicator list, retrieves its
    HPO mapping and logs whether it falls within the expected mechanism
    HPO subtree.  This is an audit tool — informational only.

    NOTE: Full HPO hierarchy traversal requires the HPO OBO file, which is
    out of scope.  This initial version checks for direct HPO term matches
    against the subtree roots, not full ancestor chains.

    Args:
        indicator_list: Sequence of IndicatorSpec-valued Enum members.
        concept_to_hpo: Lookup from build_concept_to_hpo_lookup().
        mechanism_name: One of "viral", "immuno", "endo".

    Returns:
        DataFrame with columns: indicator, concept_id, hpo_terms,
        expected_subtree, direct_match.
    """
    expected_hpo = MECHANISM_HPO_SUBTREES.get(mechanism_name, set())
    rows = []

    for member in indicator_list:
        spec = member.value if hasattr(member, "value") else member
        label = spec.label if hasattr(spec, "label") else str(member)

        # Collect all direct concept IDs from this indicator's ConceptSpecs
        concept_ids = set()
        for cspec in getattr(spec, "concepts_any_of", ()):
            concept_ids.update(cspec.direct_ids)

        for cid in concept_ids:
            hpo_terms = concept_to_hpo.get(cid, set())
            direct_match = bool(hpo_terms & expected_hpo) if expected_hpo else None
            rows.append({
                "indicator": label,
                "concept_id": cid,
                "mechanism": mechanism_name,
                "hpo_terms": "; ".join(sorted(hpo_terms)) if hpo_terms else "",
                "expected_subtree": "; ".join(sorted(expected_hpo)),
                "direct_match": direct_match,
            })

    return pd.DataFrame(rows)

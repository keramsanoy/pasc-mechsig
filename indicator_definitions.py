"""
Importable mechanistic signal indicator definitions.

Binary indicator definitions of the three mechanism clusters (thesis 3.4.2, Tables A.1-A.3),
consumed by enhanced_mech_signals.py and mech_signals_common.py.

Usage:
    from indicator_definitions import (
        ViralIndicator, VIRAL_INDICATOR_LIST,
        ImmunoIndicator, IMMUNO_INDICATOR_LIST,
        EndoIndicator, ENDO_INDICATOR_LIST,
    )
"""

from enum import Enum
from mech_signals_common import (
    ConceptSourceKind,
    ConceptSpec,
    IndicatorKind,
    IndicatorSpec,
)


# SARS-CoV-2 NAA / antigen concept set used by the viral-persistence
# indicators. Combines the original 5 OMOP IDs (which cover the dominant
# MSH PCR assays) with a LOINC-coded expansion that catches additional
# NAA / antigen panels and target-gene-specific assays seen in MSH's
# CDMPHI.measurement. Broader concept coverage + the value_source_value
# fallback (see _compile_meas_persistent_positivity) together address the
# near-zero raw prevalence of MEAS_PERSISTENT_POSITIVITY.
_SARS_COV2_TEST_CONCEPTS = (
    ConceptSpec(
        kind=ConceptSourceKind.DIRECT_IDS,
        direct_ids=(706169, 586526, 706170, 706163, 723476),
    ),
    ConceptSpec(
        kind=ConceptSourceKind.LOINC_CODES,
        loinc_codes=(
            # NAA (PCR) — pathogen-level
            "94500-6",  # SARS-CoV-2 RNA NAA Pres (Resp)
            "94309-2",  # SARS-CoV-2 RNA NAA Pres (XXX)
            "94759-8",  # SARS-CoV-2 RNA NAA Pres (Nph)
            "94660-8",  # SARS-CoV-2 RNA NAA (Resp, qual)
            "94306-8",  # SARS-CoV-2 RNA Pnl
            # NAA — target-gene specific
            "94531-1",  # SARS-CoV-2 N gene NAA
            "94534-5",  # SARS-CoV-2 N gene NAA
            "94565-9",  # SARS-CoV-2 RdRp gene NAA
            "94308-4",  # SARS-CoV-2 N gene NAA
            # Antigen
            "94558-4",  # SARS-CoV-2 Ag IA.rapid (Resp)
            "95406-5",  # SARS-CoV-2 Ag (Resp)
            "94640-0",  # SARS-CoV-2 Ag IA.rapid (Nph)
            "94769-7",  # SARS-CoV-2 Ag (Resp)
        ),
    ),
)


# ============================================================================
# Canonical SARS-CoV-2 PCR/antigen value_as_concept_id sets
# ----------------------------------------------------------------------------
# Source: concept-id audit (2026-05-01) cross-referenced against the
# OMOP standard vocabulary on Mount Sinai CDMPHI. Earlier code conflated
# 45877985 with "Negative"; OMOP defines it as "Detected" (a positive
# marker), and this caused ~100k positive rows / ~83k cohort patients to
# be mis-classified.
# ============================================================================
COVID_PCR_POSITIVE_VALUES = (45884084, 45877985, 36715206)   # Positive, Detected, Presumptive positive
# Confirmed-positive subset -- excludes 36715206 'Presumptive positive' (an
# UNCONFIRMED positive).  Used by the repeated-positive / persistence viral
# indicators, where a presumptive positive is not evidence the virus failed to
# clear.  COVID_PCR_POS_OR_NEG_VALUES intentionally keeps the full set.
COVID_PCR_POSITIVE_VALUES_STRICT = (45884084, 45877985)      # Positive, Detected
COVID_PCR_NEGATIVE_VALUES = (45880296, 45878583, 1261264)    # Not detected, Negative, Non-reactive
COVID_PCR_INCONCLUSIVE_VALUES = (45877990, 46237613, 45884199)
# Combined pos+neg, used by MEAS_PERSISTENT_POSITIVITY handlers that need
# to detect run-breaks (positive followed by negative => not persistent).
COVID_PCR_POS_OR_NEG_VALUES = COVID_PCR_POSITIVE_VALUES + COVID_PCR_NEGATIVE_VALUES


# ============================================================================
# Viral Persistence
# ============================================================================

class ViralIndicator(Enum):
    """Viral persistence and reactivation indicators."""

    REPEATED_COVID_DX = IndicatorSpec(
        label="Repeated COVID-19 diagnoses (>=2 episodes >=30d apart)",
        kind=IndicatorKind.COND_EPISODES_GAP,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(37311061,)),
        ),
        episode_gap_days=30,
        min_episodes=2,
    )

    ANY_SARS_COV2_TEST = IndicatorSpec(
        label="Any SARS-CoV-2 PCR/antigen test recorded (any result)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(706169, 586526, 706170, 706163, 723476)),
        ),
    )

    # PCR/antigen positivity: include all OMOP value_as_concept_ids that
    # signal a positive result. The concept-id audit (2026-05-01) found that
    # 45877985 = "Detected" (NOT "Negative" as previously commented) and
    # is a positive marker; restricting to 45884084 alone missed ~100k
    # "Detected" rows / ~83k cohort patients. Negative concepts are
    # 45880296 ("Not detected"), 45878583 ("Negative"), 1261264.
    # Inconclusive: 45877990, 46237613, 45884199.
    ANY_POS_SARS_COV2_TEST = IndicatorSpec(
        label="Any positive PCR/antigen test (>=1 positive)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(706169, 586526, 706170, 706163, 723476)),
        ),
        value_concept_ids=COVID_PCR_POSITIVE_VALUES,  # was: (45884084,) — missed 100k "Detected" rows
    )

    REPEATED_POS_SARS_COV2_TEST = IndicatorSpec(
        label="Repeated positive PCR/antigen tests (>=2 positives)",
        kind=IndicatorKind.MEAS_REPEAT_AT_LEAST_N,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(706169, 586526, 706170, 706163, 723476)),
        ),
        value_concept_ids=COVID_PCR_POSITIVE_VALUES_STRICT,  # confirmed positives only (excl. 36715206 presumptive); handler value filter now enforced (defect-2)
        min_count_per_person=2,
    )

    # Persistent positivity: positive -> positive with NO intervening negative.
    # Distinguishes unresolved persistence from reinfection (positive -> negative
    # -> positive). Strict variant requires >=1 positive at day >=30 from
    # postcovid_window_start, operationalizing the post-acute viral-reservoir
    # hypothesis (Proal et al. Nat Immunol 2023; Chen et al. eLife 2023).
    #
    # NOTE: For w0_21 (and any w0_N with N<30) the strict variant is
    # structurally zero by construction -- no positives can satisfy the
    # post-acute anchor. REPEATED_POS_SARS_COV2_TEST remains the acute-window
    # persistence proxy. value_concept_ids includes BOTH Positive (45884084)
    # AND Negative (45877985) -- handler reads both to detect run breaks.
    # The handler additionally falls back to value_source_value text
    # matching ("Positive"/"Detected"/"Negative"/"Not Detected") when
    # value_as_concept_id is NULL, since MSH (CDMPHI) PCR results are
    # frequently stored as text rather than concept-mapped. The
    # "no intervening negative" semantic is preserved.
    PERSISTENT_POS_SARS_COV2_STRICT = IndicatorSpec(
        label="Persistent SARS-CoV-2 positivity (index pos + >=1 in-window positive >=21d post-index, no intervening negative)",
        kind=IndicatorKind.MEAS_PERSISTENT_POSITIVITY,
        table="CDMPHI.measurement",
        concepts_any_of=_SARS_COV2_TEST_CONCEPTS,
        value_concept_ids=COVID_PCR_POS_OR_NEG_VALUES,  # was: (45884084, 45877985) — both pos; needs negatives too for run-break detection
        min_post_acute_day=21,
        index_is_anchor_positive=True,
    )

    # Sensitivity-analysis variant of strict persistence at a 14-day threshold.
    # Clinical motivation: CDC acute isolation guidance is 10 days for mild and
    # 20 days for severe; 14 days operationalises "delayed clearance past
    # typical mild-case shedding" while remaining inside the recommended
    # monitoring window for severe cases. Used as a threshold sensitivity
    # analysis against the 21-day strict variant, NOT as a replacement.
    PERSISTENT_POS_SARS_COV2_14D = IndicatorSpec(
        label=("Persistent SARS-CoV-2 positivity (index pos + >=1 in-window "
               "positive >=14d post-index, no intervening negative)"),
        kind=IndicatorKind.MEAS_PERSISTENT_POSITIVITY,
        table="CDMPHI.measurement",
        concepts_any_of=_SARS_COV2_TEST_CONCEPTS,
        value_concept_ids=COVID_PCR_POS_OR_NEG_VALUES,
        min_post_acute_day=14,
        index_is_anchor_positive=True,
    )

    # Less-strict variant: any >=2 positives with no intervening negative,
    # no post-acute anchor. Captures prolonged-shedder + reservoir combined.
    PERSISTENT_POS_SARS_COV2_ANY = IndicatorSpec(
        label="Persistent SARS-CoV-2 positivity (>=2 positives, no negative between; any timing)",
        kind=IndicatorKind.MEAS_PERSISTENT_POSITIVITY,
        table="CDMPHI.measurement",
        concepts_any_of=_SARS_COV2_TEST_CONCEPTS,
        value_concept_ids=COVID_PCR_POS_OR_NEG_VALUES,  # was: (45884084, 45877985)
    )

    EBV_PCR_ANY_VALUE = IndicatorSpec(
        label="EBV DNA PCR (any result/value recorded)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(
                kind=ConceptSourceKind.LOINC_CODES,
                loinc_codes=("32585-2", "43730-1", "47982-4", "100677-4", "100678-2"),
            ),
            ConceptSpec(
                kind=ConceptSourceKind.DIRECT_IDS,
                direct_ids=(3014258, 3050637, 3037329, 3043849),  # PCR qualitative, viral load variants
            ),
        ),
        source_value_patterns=("%ebv%", "%epstein%barr%"),
        require_any_value=True,
    )

    EBV_ANTIBODY = IndicatorSpec(
        label="EBV antibodies (IgM capsid or IgG early diffuse)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(
                kind=ConceptSourceKind.DIRECT_IDS,
                direct_ids=(3007844, 3029892),  # IgM capsid (acute), IgG early diffuse (reactivation)
            ),
        ),
        source_value_patterns=("%ebv%antibod%", "%ebv%igg%", "%ebv%igm%", "%epstein%barr%antibod%"),
        require_any_value=True,
    )

    CMV_PCR_ANY_VALUE = IndicatorSpec(
        label="CMV DNA PCR (any result/value recorded)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS, ancestor_ids=(37172169,)),
        ),
        require_any_value=True,
    )

    HHV6_PCR_ANY_VALUE = IndicatorSpec(
        label="HHV-6 DNA PCR (any result/value recorded)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(
                kind=ConceptSourceKind.DIRECT_IDS,
                direct_ids=(3031625, 3049401, 3052811, 1761324, 3029493, 3965815),
            ),
        ),
        source_value_patterns=("%hhv%6%", "%hhv-6%", "%human herpesvirus 6%"),
        require_any_value=True,
    )

    # Viral infection condition diagnoses
    INFLUENZA_A_DX = IndicatorSpec(
        label="Influenza A diagnosis (condition)",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(40483537,)),
        ),
    )

    INFLUENZA_B_DX = IndicatorSpec(
        label="Influenza B diagnosis (condition)",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(4266367,)),
        ),
    )

    HEPATITIS_B_DX = IndicatorSpec(
        label="Hepatitis B diagnosis (condition)",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(4281232,)),
        ),
    )

    HEPATITIS_C_DX = IndicatorSpec(
        label="Hepatitis C diagnosis (condition)",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(197494,)),
        ),
    )

    HEPATITIS_E_DX = IndicatorSpec(
        label="Hepatitis E diagnosis (condition)",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(45769824,)),
        ),
    )

    RSV_DX = IndicatorSpec(
        label="Respiratory Syncytial Virus (RSV) diagnosis (condition)",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(437222,)),
        ),
    )

    CMV_DX = IndicatorSpec(
        label="Cytomegalovirus (CMV) diagnosis (condition)",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(440032,)),
        ),
    )

    BRONCHOSCOPY = IndicatorSpec(
        label="Bronchoscopy (any; descendants)",
        kind=IndicatorKind.PROC_ANY_RECORDED,
        table="CDMPHI.procedure_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS, ancestor_ids=(4032404,)),
        ),
    )

    GI_BIOPSY = IndicatorSpec(
        label="GI biopsy (any; descendants)",
        kind=IndicatorKind.PROC_ANY_RECORDED,
        table="CDMPHI.procedure_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS, ancestor_ids=(4002733,)),
        ),
    )

    LIVER_BIOPSY = IndicatorSpec(
        label="Liver biopsy (any; descendants)",
        kind=IndicatorKind.PROC_ANY_RECORDED,
        table="CDMPHI.procedure_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS, ancestor_ids=(4314001,)),
        ),
    )

    REMDESIVIR = IndicatorSpec(
        label="Remdesivir exposure (any; descendants)",
        kind=IndicatorKind.DRUG_ANY_RECORDED,
        table="CDMPHI.drug_exposure",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS, ancestor_ids=(37499271,)),
        ),
    )

    NIRMATRELVIR = IndicatorSpec(
        label="Nirmatrelvir exposure (any; descendants)",
        kind=IndicatorKind.DRUG_ANY_RECORDED,
        table="CDMPHI.drug_exposure",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS, ancestor_ids=(702530,)),
        ),
    )

    RITONAVIR = IndicatorSpec(
        label="Ritonavir exposure (any; descendants)",
        kind=IndicatorKind.DRUG_ANY_RECORDED,
        table="CDMPHI.drug_exposure",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS, ancestor_ids=(1748921,)),
        ),
    )

    OTHER_COVID_DAAS = IndicatorSpec(
        label="Other COVID direct-acting antivirals (any; descendants)",
        kind=IndicatorKind.DRUG_ANY_RECORDED,
        table="CDMPHI.drug_exposure",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS, ancestor_ids=(21603127,)),
        ),
    )

    ANY_COVID_ANTIVIRAL = IndicatorSpec(
        label="Any COVID antiviral exposure (any; descendants of key antivirals)",
        kind=IndicatorKind.DRUG_ANY_RECORDED,
        table="CDMPHI.drug_exposure",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS, ancestor_ids=(37499271, 702530, 1748921, 21603127)),
        ),
    )

    REPEATED_COVID_ANTIVIRAL = IndicatorSpec(
        label="Repeated COVID antiviral treatment (>=2 courses, >=14d apart)",
        kind=IndicatorKind.DRUG_EPISODES_GAP,
        table="CDMPHI.drug_exposure",
        # DRUG_EPISODES_GAP resolves ingredient NAMES -> RxNorm Ingredient concepts
        # -> descendants; it ignores concepts_any_of, so the old concept-ID spec
        # silently errored out ("requires rxnorm_ingredient_names") and this
        # indicator was skipped.  Names cover the 4 main COVID antivirals (was:
        # DESCENDANTS(37499271 remdesivir, 702530 nirmatrelvir, 1748921 ritonavir,
        # 21603127 ATC direct-acting-antivirals class)).
        rxnorm_ingredient_names=(
            "REMDESIVIR", "NIRMATRELVIR", "RITONAVIR", "MOLNUPIRAVIR",
        ),
        episode_gap_days=14,
        min_episodes=2,
    )

    SARS_COV2_ANTIBODY = IndicatorSpec(
        label="SARS-CoV-2 antibody (IgG / related; direct measurement IDs)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(
                3033147,   # SARS coronavirus IgG Ab [measurement]
                40763481,  # SARS-CoV-2 IgG Ab
                4206313,   # SARS-CoV-2 antibody
                4132298,   # SARS-CoV-2 IgG
                3031372,   # SARS-CoV-2 Ab
                4196936,   # SARS-CoV-2 IgG Ab [quantitative]
                4211116,   # SARS-CoV-2 IgG Ab [qualitative]
            )),
        ),
        source_value_patterns=(
            "%covid%antibod%", "%sars%antibod%", "%sars%igg%",
            "%nucleocapsid%antibod%",
        ),
        require_any_value=True,
    )

    RECURRENT_RESPIRATORY_INFECTION = IndicatorSpec(
        label="Recurrent respiratory tract infections (>=2 episodes >=30d apart)",
        kind=IndicatorKind.COND_EPISODES_GAP,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS, ancestor_ids=(
                4103703,  # Upper respiratory tract infection
                255848,   # Pneumonia
                260139,   # Acute bronchitis
                258780,   # Lower respiratory tract infection
            )),
        ),
        episode_gap_days=30,
        min_episodes=2,
    )


VIRAL_INDICATOR_LIST = [
    # ── Cluster-purified: SARS-CoV-2 persistence + herpesvirus reactivation only ──
    # Removed: SARS_COV2_ANTIBODY (subsumed by v2 SPIKE_IGG / NUCLEOCAPSID_IGG / IGM)
    # Removed: EBV_ANTIBODY (subsumed by v2 EBV_VCA_IGM + EBV_EARLY_ANTIGEN)
    # Moved to immuno: INFLUENZA_A/B_DX, RSV_DX, HEPATITIS_E_DX,
    #   RECURRENT_RESPIRATORY_INFECTION (acquired-immune-dysfunction markers)
    # Moved to baseline_ext: HEPATITIS_B/C_DX (chronic comorbidity confounders)
    # Dropped: BRONCHOSCOPY, GI_BIOPSY, LIVER_BIOPSY (workup-intensity proxies)
    ViralIndicator.REPEATED_POS_SARS_COV2_TEST,
    # Persistent positivity (new) -- tighter reservoir-hypothesis operationalization
    ViralIndicator.PERSISTENT_POS_SARS_COV2_STRICT,
    ViralIndicator.PERSISTENT_POS_SARS_COV2_14D,    # v3 threshold sensitivity (14d)
    ViralIndicator.PERSISTENT_POS_SARS_COV2_ANY,
    ViralIndicator.REPEATED_COVID_DX,
    ViralIndicator.EBV_PCR_ANY_VALUE,
    ViralIndicator.CMV_PCR_ANY_VALUE,
    ViralIndicator.CMV_DX,
    ViralIndicator.HHV6_PCR_ANY_VALUE,
    # Antiviral treatment (kept — exposure, not workup)
    ViralIndicator.REMDESIVIR,
    ViralIndicator.NIRMATRELVIR,
    ViralIndicator.RITONAVIR,
    ViralIndicator.OTHER_COVID_DAAS,
    ViralIndicator.ANY_COVID_ANTIVIRAL,
    ViralIndicator.REPEATED_COVID_ANTIVIRAL,
]


# ============================================================================
# Immuno-Inflammatory
# ============================================================================

class ImmunoIndicator(Enum):
    """Immuno-inflammatory dysregulation indicators."""

    # ELEV_CRP, ELEV_ESR removed 2026-07-17: replaced by the numeric-lab
    # abnormal_any flags (CRP > 10 mg/L, ESR > 20 mm/hr).  The old
    # MEAS_ELEVATED_OR_ANY binaries were availability-biased (fired on any value
    # recorded without a reference range).  See f_lab_*_crp/esr_abnormal_any.
    ELEV_FERRITIN = IndicatorSpec(
        label="Systemic inflammation: Ferritin elevated (range_high if available; else any ferritin recorded)",
        kind=IndicatorKind.MEAS_ELEVATED_OR_ANY,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("2276-4",)),
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(4176561,)),
        ),
    )

    ANY_IL6 = IndicatorSpec(
        label="Cytokine activity: IL-6 measured (any result recorded) [measurement + observation]",
        kind=IndicatorKind.MEAS_OR_OBS_ANY,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("26881-3",)),
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(4332015, 42529420)),
        ),
        obs_concept_ids=(4150054,),
        require_any_value=True,
    )

    ANY_TNFA = IndicatorSpec(
        label="Cytokine activity: TNF-alpha measured (any result recorded) [measurement + observation]",
        kind=IndicatorKind.MEAS_OR_OBS_ANY,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.NAME_PATTERN, name_patterns=("%TNF%ALPHA%", "%TUMOR NECROSIS FACTOR%"), domain_id="Measurement"),
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(4225604,)),
        ),
        obs_concept_ids=(4216487,),
        require_any_value=True,
    )

    ANY_IL12 = IndicatorSpec(
        label="Cytokine activity: IL-12 measured (any result recorded)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS, ancestor_ids=(3034621,)),  # was: 37046229 (non-standard)
        ),
        require_any_value=True,
    )

    ANY_LYMPH_ABS = IndicatorSpec(
        label="Immune cell imbalance: Lymphocyte absolute count measured (any result)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("731-0",)),
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(37208689,)),
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS, ancestor_ids=(4254663,)),
        ),
        require_any_value=True,
    )

    ANY_NEUT_ABS = IndicatorSpec(
        label="Immune cell imbalance: Neutrophil absolute count measured (any result)",
        kind=IndicatorKind.MEAS_ANY_RECORDED,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("751-8",)),
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(4148615,)),
        ),
        require_any_value=True,
    )

    NLR_COMPUTABLE = IndicatorSpec(
        label="Immune cell imbalance: NLR computable (neutrophils & lymphocytes numeric same date)",
        kind=IndicatorKind.MEAS_PAIRED_SAME_DATE_NUMERIC,
        table="CDMPHI.measurement",
        left_concepts=ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("731-0", "26474-7")),
        right_concepts=ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("751-8", "26499-4")),
    )

    PERSISTENT_CRP = IndicatorSpec(
        label="Repeated inflammatory labs: CRP persistently elevated (>=2 episodes >=30d apart)",
        kind=IndicatorKind.MEAS_ELEV_EPISODES_GAP,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("1988-5", "30522-7")),
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(4208414,)),
        ),
        episode_gap_days=30,
        min_episodes=2,
    )

    ANY_IL6_OBS = IndicatorSpec(
        label="Cytokine activity: IL-6 observed (any result recorded) [OBSERVATION domain]",
        kind=IndicatorKind.OBS_ANY_RECORDED,
        table="CDMPHI.observation",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(4150054,)),
        ),
        require_any_value=True,
    )

    ANY_TNFA_OBS = IndicatorSpec(
        label="Cytokine activity: TNF-alpha observed (any result recorded) [OBSERVATION domain]",
        kind=IndicatorKind.OBS_ANY_RECORDED,
        table="CDMPHI.observation",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(4216487,)),
        ),
        require_any_value=True,
    )

    NEUTROPHIL_ANTIBODY_OBS = IndicatorSpec(
        label="Immune cell imbalance: Neutrophil antibody observed (any result recorded) [OBSERVATION domain]",
        kind=IndicatorKind.OBS_ANY_RECORDED,
        table="CDMPHI.observation",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(35624339,)),
        ),
        require_any_value=True,
    )

    IMMUNE_RELATED_DX = IndicatorSpec(
        label="Immune-related diagnoses: immune/inflammatory disorder codes (DESC of chosen ancestors)",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS, ancestor_ids=(4098244, 443454, 433590)),
        ),
    )

    IMMUNOMOD_DRUGS = IndicatorSpec(
        label="Immunomodulatory therapy: corticosteroids/DMARDs/biologics exposures (RxNorm ingredients)",
        kind=IndicatorKind.DRUG_INGREDIENT_DESC,
        table="CDMPHI.drug_exposure",
        rxnorm_ingredient_names=(
            "prednisone", "prednisolone", "methylprednisolone", "dexamethasone", "hydrocortisone",
            "methotrexate", "azathioprine", "mycophenolate mofetil", "leflunomide", "sulfasalazine", "hydroxychloroquine",
            "adalimumab", "infliximab", "etanercept", "rituximab", "tocilizumab", "abatacept", "ustekinumab",
            "tofacitinib", "baricitinib", "upadacitinib",
        ),
        source_value_patterns=(
            "%prednisone%", "%dexamethasone%", "%methylprednisolone%",
            "%methotrexate%", "%hydroxychloroquine%", "%rituximab%", "%tocilizumab%",
        ),
    )

    SPECIALIST_VISITS = IndicatorSpec(
        label="Specialist care: rheumatology/immunology visits (via PROVIDER.specialty)",
        kind=IndicatorKind.VISIT_SPECIALTY_NAME,
        table="CDMPHI.visit_occurrence",
        specialty_name_patterns=("%rheumatolog%", "%immunolog%"),
    )

    # Bacterial infection condition diagnoses
    PNEUMONIA_DX = IndicatorSpec(
        label="Pneumonia diagnosis (Chlamydophila/Mycoplasma) (condition)",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(255848,)),
        ),
    )

    LYME_DISEASE_DX = IndicatorSpec(
        label="Lyme Disease (Borrelia burgdorferi) diagnosis (condition)",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(440638,)),
        ),
    )

    OPPORTUNISTIC_INFECTION = IndicatorSpec(
        label="Opportunistic infection (fungal, PCP, atypical) suggesting immunosuppression",
        kind=IndicatorKind.COND_ANY_RECORDED,
        table="CDMPHI.condition_occurrence",
        concepts_any_of=(
            # NOTE: 437663 was previously listed here mislabeled as "Aspergillosis";
            # audit confirmed it is the OMOP "Fever" concept, which over-fired the
            # indicator (~5% prevalence). Removed. A real Aspergillosis ancestor
            # is not yet included -- track as follow-up if needed.
            ConceptSpec(kind=ConceptSourceKind.DESCENDANTS, ancestor_ids=(
                433701,   # Candidiasis
                4140111,  # Toxoplasmosis
                432584,   # Cryptococcosis
                440704,   # Histoplasmosis
            )),
        ),
    )


IMMUNO_INDICATOR_LIST = [
    # ── Cluster-purified: autoimmunity + cytokine/complement activation ──
    # Removed: ELEV_CRP, ELEV_ESR, ELEV_FERRITIN (numeric quant stream preferred;
    #   f_lab_*_crp/esr/ferritin_abnormal_any subsumes elevated-flag)
    # Removed: SPECIALIST_VISITS (subsumed by v2 RHEUMATOLOGY_VISIT + IMMUNOLOGY_VISIT)
    # Removed: IMMUNE_RELATED_DX (subsumed by v2 NEW_RHEUM_DX, more specific)
    # Removed: ANY_IL6_OBS, ANY_TNFA_OBS (merged into ANY_IL6 / ANY_TNFA
    #   via obs_concept_ids — single cross-domain indicator now)
    # Cytokines (now cross-domain: measurement + observation)
    ImmunoIndicator.ANY_IL6,
    ImmunoIndicator.ANY_TNFA,
    ImmunoIndicator.ANY_IL12,
    # Immune cell markers
    ImmunoIndicator.ANY_LYMPH_ABS,
    ImmunoIndicator.ANY_NEUT_ABS,
    ImmunoIndicator.NLR_COMPUTABLE,
    # Persistent inflammation
    ImmunoIndicator.PERSISTENT_CRP,
    # Immunomodulatory therapy (exposure, not workup)
    ImmunoIndicator.IMMUNOMOD_DRUGS,
    ImmunoIndicator.NEUTROPHIL_ANTIBODY_OBS,
    # Acquired immune dysfunction markers (reclassified from viral / reframed)
    ImmunoIndicator.PNEUMONIA_DX,
    ImmunoIndicator.LYME_DISEASE_DX,
    ImmunoIndicator.OPPORTUNISTIC_INFECTION,
    ViralIndicator.INFLUENZA_A_DX,           # moved from viral (Phetsouphanh 2022)
    ViralIndicator.INFLUENZA_B_DX,           # moved from viral
    ViralIndicator.RSV_DX,                   # moved from viral
    ViralIndicator.HEPATITIS_E_DX,           # moved from viral (acute infection)
    ViralIndicator.RECURRENT_RESPIRATORY_INFECTION,  # moved from viral
]


# ============================================================================
# Endothelial Dysfunction & Thrombosis
# ============================================================================

class EndoIndicator(Enum):
    """Endothelial dysfunction and immune thrombosis indicators."""

    # THROMBOTIC_EVENTS removed 2026-07-17: v2 replaced it with DVT_PE +
    # ARTERIAL_THROMBOSIS (MI + stroke).  Its post-acute role is now carried by
    # the f_comp_td_unresolved_thrombosis composite (dvt_pe + arterial_thrombosis).
    MICROVASCULAR_INJURY = IndicatorSpec(
        label="Endothelial dysfunction & immune thrombosis: Microvascular injury",
        kind=IndicatorKind.COND_NAME_PATTERN,
        table="CDMPHI.condition_occurrence",
        concept_name_patterns=(
            "%THROMBOPHLEBITIS%",
            "%MICROVASCULAR%",
            "%SMALL VESSEL%",
            "%ISCHEMIC%",
            "%ACRAL ISCHEM%",
        ),
        notes="Pattern matching is a broad proxy; under-coding likely",
    )

    # COAG_ACTIVATION_ANY, COAG_DDIMER_ELEV_PROXY removed 2026-07-17: d-dimer
    # elevation now lives in the numeric-lab stream (f_lab_*_ddimer_abnormal_any).
    # (Both also carried LOINC 71425-3 = NT-proBNP, a wrong-analyte code, 0 rows.)
    PLATELETS_ANY = IndicatorSpec(
        label="Platelet count available",
        kind=IndicatorKind.MEAS_NUMERIC_ANY,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("777-3", "26515-7")),
            ConceptSpec(kind=ConceptSourceKind.DIRECT_IDS, direct_ids=(4267147,)),
        ),
        source_value_patterns=("%platelet%count%", "%plt%"),
        notes="Availability only (numeric recorded); 777-3 is high-volume Platelet count LOINC",
    )

    # COMPLEMENT_ACTIVITY removed 2026-07-17: superseded by the EnhImmuno
    # complement panel (C3/C4/CH50 ordered).  Its LOINC 11572-5 was actually RF.
    ANTICOAG_THERAPY = IndicatorSpec(
        label="Anticoagulation therapy (heparin/DOACs/warfarin)",
        kind=IndicatorKind.DRUG_INGREDIENT_DESC,
        table="CDMPHI.drug_exposure",
        rxnorm_ingredient_names=(
            "HEPARIN", "WARFARIN", "APIXABAN", "RIVAROXABAN",
            "DABIGATRAN ETEXILATE", "EDOXABAN", "ENOXAPARIN",
            "DALTEPARIN", "FONDAPARINUX",
        ),
        source_value_patterns=(
            "%heparin%", "%warfarin%", "%apixaban%", "%eliquis%",
            "%rivaroxaban%", "%xarelto%", "%enoxaparin%", "%lovenox%",
        ),
        notes="Uses ingredient name matching + descendants (RxNorm); validate locally",
    )

    APHERESIS = IndicatorSpec(
        label="Apheresis (plasmapheresis / therapeutic plasma exchange)",
        kind=IndicatorKind.PROC_NAME_PATTERN,
        table="CDMPHI.procedure_occurrence",
        concept_name_patterns=(
            "%PLASMAPHERESIS%",
            "%THERAPEUTIC PLASMA EXCHANGE%",
            "%PLASMA EXCHANGE%",
        ),
        source_value_patterns=("%apheresis%", "%plasmapheresis%", "%plasma exchange%"),
        notes="Low frequency but specific; validate concept set locally",
    )

    REPEATED_DDIMER_TESTING = IndicatorSpec(
        label="Repeated D-dimer testing (>=2 rows)",
        kind=IndicatorKind.MEAS_REPEAT_AT_LEAST_N,
        table="CDMPHI.measurement",
        concepts_any_of=(
            ConceptSpec(kind=ConceptSourceKind.LOINC_CODES, loinc_codes=("48065-7",)),  # was: ("48065-7", "3246-6", "71425-3"); 3246-6 deprecated, 71425-3 = NT-proBNP (wrong analyte)
        ),
        min_count_per_person=2,
        notes="Row-count proxy for data richness (not episode-based)",
    )

    REPEATED_ANTICOAG_EXPOSURE = IndicatorSpec(
        label="Repeated anticoagulant exposure (>=2 rows)",
        kind=IndicatorKind.DRUG_REPEAT_AT_LEAST_N,
        table="CDMPHI.drug_exposure",
        rxnorm_ingredient_names=(
            "HEPARIN", "WARFARIN", "APIXABAN", "RIVAROXABAN",
            "DABIGATRAN ETEXILATE", "EDOXABAN", "ENOXAPARIN",
            "DALTEPARIN", "FONDAPARINUX",
        ),
        min_count_per_person=2,
        notes="Row-count proxy for data richness; ingredient descendants via RxNorm",
    )


ENDO_INDICATOR_LIST = [
    # ── Cluster-purified: thrombo-inflammation + vascular/cardiac/renal injury ──
    # Removed: THROMBOTIC_EVENTS (subsumed by v2 DVT_PE + ARTERIAL_THROMBOSIS)
    # Removed: MICROVASCULAR_INJURY (subsumed by v2 AKI_DIAGNOSIS + PROTEINURIA_DX
    #   + MYOCARDITIS_PERICARDITIS)
    # Removed: COAG_ACTIVATION_ANY, PLATELETS_ANY (availability-only flags;
    #   numeric quant f_lab_*_ddimer/platelets_* preferred)
    # Removed: COAG_DDIMER_ELEV_PROXY (numeric f_lab_*_ddimer_abnormal_any
    #   subsumes elevated-threshold flag)
    # Removed: COMPLEMENT_ACTIVITY (retired — wrong LOINC 11572-5 = RF;
    #   C3/C4/CH50 fully handled by EnhImmuno complement indicators)
    EndoIndicator.ANTICOAG_THERAPY,
    EndoIndicator.APHERESIS,
    EndoIndicator.REPEATED_DDIMER_TESTING,
    EndoIndicator.REPEATED_ANTICOAG_EXPOSURE,
]

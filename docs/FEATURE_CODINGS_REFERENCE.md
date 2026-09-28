# Feature Codings & Concept Reference

Exhaustive catalog of every OMOP concept ID, LOINC code, RxNorm ingredient, SNOMED
ancestor, `source_value` LIKE pattern, and OMOP2OBO HPO mapping used by the
post-COVID (PASC) prediction pipeline.

> Written during development and kept as a lookup aid for other OMOP sites. The
> configuration names `mechsig_*_enh` below are the development names of what the
> thesis calls `mechsig_viral`, `mechsig_immuno`, `mechsig_endo` and `mechsig_all`
> (Table 3.2); the module code is authoritative where the two disagree.

The document is organized **hierarchically** by model config:

1. [`baseline_antony`](#1-baseline_antony) — Antony replication features (Families A–E)
2. [`baseline_ext`](#2-baseline_ext) — Antony + extended baseline (demographics, severity, vaccination …)
3. [`mechsig_viral_enh`](#3-mechsig_viral_enh) — v1 + v2 viral persistence indicators
4. [`mechsig_immuno_enh`](#4-mechsig_immuno_enh) — v1 + v2 immuno-inflammatory indicators
5. [`mechsig_endo_enh`](#5-mechsig_endo_enh) — v1 + v2 endothelial dysfunction indicators
6. [`mechsig_all_enh`](#6-mechsig_all_enh) — everything above + quantitative labs, CBC indices, composites, temporal-divergence features
7. [OMOP2OBO / HPO symptom mapping](#7-omop2obo--hpo-symptom-mapping)
8. [Query-kind glossary](#8-indicator-kind--concept-source-glossary)

Every feature entry lists:

| Column | Meaning |
|---|---|
| Feature | Column name produced in the feature matrix |
| Source | OMOP CDM table (`condition_occurrence`, `measurement`, `observation`, `drug_exposure`, `procedure_occurrence`, `visit_occurrence`) |
| Concept resolution | How concept IDs are obtained (`DIRECT_IDS`, `LOINC_CODES`, `DESCENDANTS` via `concept_ancestor`, `NAME_PATTERN`, `MIXED`) |
| Codes / IDs | The actual concept IDs / LOINC codes / RxNorm ingredient names |
| `source_value` LIKE | Fallback case-insensitive pattern applied to `*_source_value` |
| Threshold / logic | Episode gaps, min counts, numeric cut-offs, etc. |

Source files: [antony_features.py](../antony_features.py), [combined_features.py](../combined_features.py), [indicator_definitions.py](../indicator_definitions.py), [enhanced_mech_signals.py](../enhanced_mech_signals.py), [mech_signals_common.py](../mech_signals_common.py), [omop2obo_mapping.py](../omop2obo_mapping.py).

---

## 1. `baseline_antony`

Antony et al. 2023 replication. Built in [antony_features.py](../antony_features.py) via `build_antony_feature_matrix()`. Five families:

### 1A. Comorbidities — Family A

- **Source:** `CDMPHI.condition_occurrence`
- **Window:** on or before `covid_index_date`
- **Resolution:** `DESCENDANTS` via `concept_ancestor` (ancestor concept IDs)

| Feature | Ancestor concept_id | SNOMED meaning |
|---|---|---|
| f_comor_mi | 4329847 | Myocardial infarction |
| f_comor_chf | 316139 | Heart failure |
| f_comor_pvd | 321052 | Peripheral vascular disease |
| f_comor_cerebrovascular | 381591 | Cerebrovascular disease |
| f_comor_dementia | 4182210 | Dementia |
| f_comor_chronic_pulm | 4063381 | Chronic lower respiratory disease |
| f_comor_rheumatic | 257628 | Rheumatic disease |
| f_comor_peptic_ulcer | 4247120 | Peptic ulcer disease |
| f_comor_liver_mild | 4212540 | Chronic liver disease |
| f_comor_diabetes_complicated | 443767 | Diabetes w/ complication |
| f_comor_diabetes_uncomplicated | 201820 | Diabetes mellitus (broad, minus 443767) |
| f_comor_hemiplegia | 374022 | Hemiplegia / paraplegia |
| f_comor_renal | 198124 | Chronic kidney disease |
| f_comor_cancer | 443392 | Malignant neoplasm |
| f_comor_hiv | 439727 | HIV / AIDS |
| f_comor_cardiovascular | 134057 | CDC severe-COVID CV disorder |
| f_comor_cad | 4185932 | Ischemic heart disease (was: 321318 Angina pectoris) |
| f_comor_obesity | 433736 | Obesity |
| f_comor_depression | 440383 | Depressive disorder |
| f_comor_anxiety | 442077 | Anxiety disorder |
| f_comor_autoimmune | 434621 | Autoimmune disease |
| f_comor_hypertension | 316866 | Hypertensive disorder |

SQL shape (see [antony_features.py](../antony_features.py)):

```sql
SELECT c.person_id,
       MAX(CASE WHEN ca.ancestor_concept_id = <ID> THEN 1 ELSE 0 END) AS <feature>
FROM "#antony_cohort" c
LEFT JOIN condition_occurrence co ON co.person_id = c.person_id
     AND co.condition_start_date <= c.covid_index_date
LEFT JOIN concept_ancestor ca ON ca.descendant_concept_id = co.condition_concept_id
GROUP BY c.person_id;
```

### 1B. Acute Symptoms — Family B

- **Source:** `condition_occurrence`
- **Window:** `acute_start … acute_end`
- **Resolution:** `DESCENDANTS` (SNOMED ancestor) *or* HPO mapping (see [§7](#7-omop2obo--hpo-symptom-mapping) when `use_hpo_symptoms=True`)

Core 14 Antony symptoms:

| Feature | SNOMED ancestor | HPO curie |
|---|---|---|
| f_sym_fever | 437663 | HP:0001945 |
| f_sym_cough | 254761 | HP:0012735 |
| f_sym_fatigue | 4223659 | HP:0012378 |
| f_sym_dyspnea | 312437 | HP:0002094 |
| f_sym_myalgia | 442752 | HP:0003326 |
| f_sym_headache | 378253 | HP:0002315 |
| f_sym_anosmia | 4185711 | HP:0000458 |
| f_sym_ageusia | 4289517 | HP:0000224 |
| f_sym_diarrhea | 196523 | HP:0002014 |
| f_sym_nausea | 31967 | HP:0002018 |
| f_sym_sore_throat | 4147326 | HP:0033050 |
| f_sym_chest_pain | 77670 | HP:0100749 |
| f_sym_arthralgia | 77074 | HP:0002829 |
| f_sym_vomiting | 441408 | HP:0002013 |

Extra symptoms (`strict_antony=False`):

| Feature | SNOMED ancestor | HPO curie |
|---|---|---|
| f_sym_abdominal_pain | 200219 | HP:0002027 |
| f_sym_dizziness | 4223938 | HP:0002321 |
| f_sym_malaise | 4272240 | HP:0033834 |
| f_sym_rhinorrhea | 4276172 | HP:0031417 |
| f_sym_congestion | 4195085 | HP:0001742 |

### 1C. Acute Drugs — Family C

- **Source:** `drug_exposure` pivoted to RxNorm ingredients via `concept_ancestor`
- **Window:** `acute_start … acute_end`
- **Prevalence gate:** drop ingredient columns with <1% cohort prevalence
- **Column naming:** `f_drug_<sanitized_ingredient_name>`

Ingredient resolution:

```sql
SELECT DISTINCT ca.ancestor_concept_id AS ingredient_concept_id
FROM concept_ancestor ca
JOIN concept con ON con.concept_id = ca.ancestor_concept_id
WHERE con.concept_class_id = 'Ingredient'
  AND con.vocabulary_id  = 'RxNorm'
  AND ca.descendant_concept_id IN (<drug_concept_ids_in_window>);
```

Composite: `f_drug_corticosteroid_any` = OR of ingredients in
`{prednisone, dexamethasone, methylprednisolone, hydrocortisone, prednisolone, betamethasone, triamcinolone}`.

COVID-regimen corticosteroids (inpatient only, direct ingredient ancestors):
`1518254` prednisone, `1518978` dexamethasone, `1506270` methylprednisolone,
`1507705` hydrocortisone, `1550557` prednisolone, `920458` betamethasone,
`903963` triamcinolone.

### 1D. Demographics — Family D

- `f_age` — `DAYS_BETWEEN(index, birth)/365.25`
- `f_sex_female` — `gender_concept_id = 8532`
- `f_sex_male`   — `gender_concept_id = 8507`
- `f_sex_unknown` — otherwise

### 1E. Treatment — Family E (inpatient subset)

| Feature | Source | Codes | Resolution |
|---|---|---|---|
| f_tx_los_days | `visit_occurrence` | inpatient visit types `(9201, 262)` | `DAYS_BETWEEN(end, start)` |
| f_tx_imv | `procedure_occurrence` | ancestor `4230167` | `DESCENDANTS` |
| f_tx_imv_alt | `observation` | `3004921` (vent-mode) | `DIRECT_IDS` |
| f_tx_ecmo | `procedure_occurrence` | ancestor `4052536` | `DESCENDANTS` |
| f_tx_remdesivir | `drug_exposure` | ancestor `37499271` | `DESCENDANTS` |

WHO severity one-hot (derived, not a new DB query):

| Feature | Logic |
|---|---|
| f_tx_severity_who_severe_icu_vent | inpatient AND (IMV OR vent-mode observation) |
| f_tx_severity_who_moderate_hosp | inpatient, no IMV |
| f_tx_severity_who_mild_ed | outpatient AND ED visit in acute window (ED visit types `9203, 262`) |
| f_tx_severity_who_mild_no_ed | outpatient, no ED visit |

---

## 2. `baseline_ext`

`baseline_antony` **plus** the extended features below. Built by
`build_combined_feature_matrix(..., include_extended=True)` in
[combined_features.py](../combined_features.py).

### 2A. Vitals / BMI / lifestyle (pre-index, last value)

| Feature | Source | Concept IDs / LIKE | Notes |
|---|---|---|---|
| f_ext_bmi_last | `measurement` | `3038553` | last numeric value ≤ index |
| f_ext_sbp_last | `measurement` | `3004249` | last numeric SBP |
| f_ext_dbp_last | `measurement` | `3012888` | last numeric DBP |
| f_ext_smoking_any | `observation` | `1585856` or `observation_source_value LIKE '%smok%'` / `'%tobacco%'` | binary |
| f_ext_alcohol_any | `observation` | `1586197` or `LIKE '%alcohol%'` | binary |

### 2B. Acute severity

| Feature | Source | Codes / LIKE | Notes |
|---|---|---|---|
| f_ext_hosp_acute | `visit_occurrence` | inpatient types `(9201, 262)` | any inpatient overlap with acute window |
| f_ext_los_max | `visit_occurrence` | — | max LOS across acute admissions |
| f_ext_n_admissions | `visit_occurrence` | — | count of inpatient visits in window |
| f_ext_icu_obs | `observation` | `1259883` or `LIKE '%icu%'` | |
| f_ext_icu_care_site | `measurement` | `706367` | |
| f_ext_icu_site_kw | `care_site.care_site_name` | `LIKE '%icu%' / '%intensive%' / '%critical care%' / '%micu%' / '%sicu%' / '%ccu%'` | |
| f_ext_icu_any | derived | OR of the three above | |
| f_ext_spo2_any | `measurement` | `3013502` | |
| f_ext_spo2_min | `measurement` | `3013502` | min numeric value |
| f_ext_vent_any | `observation` | `3004921` | |

### 2C. Variant-wave assignment (pure pandas on `covid_index_date`)

| Feature | Window |
|---|---|
| f_ext_wave_ancestral_1 | 2020-02-09 … 2020-08-30 |
| f_ext_wave_iota_alpha_2 | 2020-08-31 … 2021-06-20 |
| f_ext_wave_delta_3 | 2021-06-21 … 2021-10-31 |
| f_ext_wave_delta_omicron_4 | 2021-11-01 … 2022-03-06 |
| f_ext_wave_omicron_ba2_ba5_5 | 2022-03-07 … 2022-07-18 |
| f_ext_wave_unclassified | otherwise |

### 2D. Vaccination (pre-index `drug_exposure`)

Concept ID sets:

- **Any COVID vaccine:** `724907, 724906, 702866, 702678, 724904, 702676, 724905, 702664, 702672, 702679, 702677, 702666, 905420`
- **mRNA:** `724907, 724906, 702678, 702676, 702677, 905420`
- **Vector:** `702866, 724905`

Fallback LIKE on `drug_source_value`:
`'%pfizer%' / '%biontech%' / '%comirnaty%' / '%moderna%' / '%spikevax%' / '%janssen%' / '%johnson%' / '%astrazeneca%' / '%vaxzevria%'`.

| Feature | Meaning |
|---|---|
| f_ext_vax_any | any pre-index vaccination |
| f_ext_vax_dose_count | numeric dose count |
| f_ext_vax_boosted | dose_count ≥ 3 |
| f_ext_vax_mrna_any | any mRNA |
| f_ext_vax_vector_any | any vector |

### 2E. Additional acute symptoms (ancestor-based on `condition_occurrence`)

| Feature | Ancestor IDs |
|---|---|
| f_ext_sym_palpitations | `315078, 77670` |
| f_ext_sym_neurocog | `46271045, 4107230, 443432` |
| f_ext_sym_insomnia | `436962, 435524` |
| f_ext_sym_tachycardia | `444070` |

### 2F. Chronic viral hepatitis (pre-index comorbidities)

| Feature | Ancestor ID |
|---|---|
| f_ext_hepatitis_b | `4281232` |
| f_ext_hepatitis_c | `197494` |

### 2G. SARS-CoV-2 serology (pre-index + acute)

LOINC panels (resolved to concept IDs via `vocabulary_id='LOINC'`):

```python
SARS_COV2_AB_LOINCS   = ["94661-6", "94563-4", "94769-7", "94505-5", "94762-2", "94564-2"]
SARS_COV2_N_AB_LOINCS = ["94720-0", "94504-8", "94761-4", "94506-3", "96118-2"]
SARS_COV2_S_AB_LOINCS = ["94509-7", "94551-9", "96831-0", "94507-1", "94505-5"]
```

Positivity via `value_as_concept_id`:
- Positive: `45884084`
- Negative: `45877985`

| Feature | Meaning |
|---|---|
| f_ext_sars_cov2_ab_ever_positive | ≥1 positive result |
| f_ext_sars_cov2_ab_ever_negative | ≥1 negative result |
| f_ext_sars_cov2_ab_wave_at_first_positive | wave (1–6) of first positive |
| f_ext_sars_cov2_ab_n_tests | count of serology tests |

### 2H. Engagement controls (always forced past prevalence + Boruta)

Counts of encounters / distinct problem list entries / distinct medications in
the 1y pre-index window (e.g. `f_eng_n_encounters_pre_index_1y`). Used to
control ascertainment bias.

---

## 3. `mechsig_viral_enh`

`baseline_ext` **plus** both v1 (`VIRAL_INDICATOR_LIST`) and v2
(`ENH_VIRAL_INDICATOR_LIST`) viral indicators. Each row is an `IndicatorSpec`
consumed by `_extract_indicator()` in
[mech_signals_common.py](../mech_signals_common.py) — see [§8](#8-indicator-kind--concept-source-glossary)
for kind semantics.

### 3A. v1 Viral (`indicator_definitions.py` → `VIRAL_INDICATOR_LIST`)

| Indicator | Kind | Source | Codes | `source_value` LIKE | Thresholds |
|---|---|---|---|---|---|
| REPEATED_POS_SARS_COV2_TEST | MEAS_REPEAT_AT_LEAST_N | measurement | same PCR concepts | — | `value_as_concept_id IN (45884084, 45877985)`; count ≥ 2 |
| PERSISTENT_POS_SARS_COV2_STRICT | MEAS_PERSISTENT_POSITIVITY | measurement | DIRECT `(706169, 586526, 706170, 706163, 723476)` + LOINC `94500-6, 94309-2, 94759-8, 94660-8, 94306-8, 94531-1, 94534-5, 94565-9, 94308-4, 94558-4, 95406-5, 94640-0, 94769-7` | value text fallback (`positive/detected/negative/not detected`) when concept value missing | index positive as anchor + ≥1 in-window positive at day ≥21 + no intervening negative |
| PERSISTENT_POS_SARS_COV2_ANY | MEAS_PERSISTENT_POSITIVITY | measurement | same expanded SARS-CoV-2 concept set as strict variant | same value text fallback | ≥2 positives, no intervening negative (no day anchor) |
| REPEATED_COVID_DX | COND_EPISODES_GAP | condition | DIRECT `(37311061,)` | — | gap=30 d, min=2 |
| EBV_PCR_ANY_VALUE | MEAS_ANY_RECORDED | measurement | LOINC `32585-2, 43730-1, 47982-4, 100677-4, 100678-2` + DIRECT `(3014258, 3050637, 3037329, 3043849)` | `%ebv%, %epstein%barr%` | `require_any_value` |
| CMV_PCR_ANY_VALUE | MEAS_ANY_RECORDED | measurement | DESCENDANTS of `37172169` | — | `require_any_value` |
| CMV_DX | COND_ANY_RECORDED | condition | DIRECT `(440032,)` | — | — |
| HHV6_PCR_ANY_VALUE | MEAS_ANY_RECORDED | measurement | DIRECT `(3031625, 3049401, 3052811, 1761324, 3029493, 3965815)` | `%hhv%6%, %hhv-6%, %human herpesvirus 6%` | — |
| REMDESIVIR | DRUG_ANY_RECORDED | drug | DESCENDANTS `(37499271,)` | — | — |
| NIRMATRELVIR | DRUG_ANY_RECORDED | drug | DESCENDANTS `(702530,)` | — | — |
| RITONAVIR | DRUG_ANY_RECORDED | drug | DESCENDANTS `(1748921,)` | — | — |
| OTHER_COVID_DAAS | DRUG_ANY_RECORDED | drug | DESCENDANTS `(21603127,)` | — | — |
| ANY_COVID_ANTIVIRAL | DRUG_ANY_RECORDED | drug | DESCENDANTS `(37499271, 702530, 1748921, 21603127)` | — | — |
| REPEATED_COVID_ANTIVIRAL | DRUG_EPISODES_GAP | drug | same antivirals | — | gap=14 d, min=2 |

### 3C. Viral persistence changes applied (current)

- `ANY_POS_SARS_COV2_TEST` is now positive-only (`45884084`) where used; no longer conflated with negative-result rows.
- Two new persistence indicators are active in v1:
   - `PERSISTENT_POS_SARS_COV2_STRICT` (index-anchored, post-acute day gate, no negative between positives)
   - `PERSISTENT_POS_SARS_COV2_ANY` (same no-intervening-negative logic, no day anchor)
- Persistent positivity now uses an expanded SARS-CoV-2 concept set (direct OMOP test IDs + added LOINC PCR/antigen panels and gene-target assays).
- Persistent positivity handler includes `value_source_value` text fallback (detected/positive/negative/not detected) when `value_as_concept_id` is null.
- v1 viral list was cluster-purified:
   - moved to immuno: `INFLUENZA_A_DX`, `INFLUENZA_B_DX`, `RSV_DX`, `HEPATITIS_E_DX`, `RECURRENT_RESPIRATORY_INFECTION`
   - moved to baseline ext (comorbidity confounders): `HEPATITIS_B_DX`, `HEPATITIS_C_DX`
   - dropped as workup-intensity proxies: `BRONCHOSCOPY`, `GI_BIOPSY`, `LIVER_BIOPSY`
   - decomposed into v2 subtype indicators: `SARS_COV2_ANTIBODY`, `EBV_ANTIBODY`

### 3B. v2 Enhanced Viral (`ENH_VIRAL_INDICATOR_LIST`)

| Indicator | Kind | Source | Codes | `source_value` LIKE |
|---|---|---|---|---|
| LATE_POS_SARS_COV2 | MEAS_ANY_RECORDED | measurement | DIRECT `(706169, 586526, 706170, 706163, 723476)` + `value_as_concept_id IN (45884084)` | — |
| SARS_COV2_NUCLEOCAPSID_IGG | MEAS_ANY_RECORDED | measurement | DIRECT `(723478, 40771922)` | `%nucleocapsid%igg%, %sars%cov%nucleocapsid%, %covid%nucleocapsid%` |
| SARS_COV2_SPIKE_IGG | MEAS_ANY_RECORDED | measurement | DIRECT `(40763481, 4132298, 4196936, 4211116)` | `%spike%igg%, %sars%cov%igg%, %covid%igg%, %sars%igg%` |
| SARS_COV2_IGM | MEAS_ANY_RECORDED | measurement | NAME_PATTERN `%SARS%COV%2%IGM%` (Measurement domain) | `%sars%cov%igm%, %covid%igm%` |
| EBV_PCR_ANY | MEAS_ANY_RECORDED | measurement | LOINC `32585-2` | — |
| EBV_EARLY_ANTIGEN | MEAS_ANY_RECORDED | measurement | LOINC `24007-2, 30339-6` | `%ebv%early antigen%, %ebv%ea%igg%, %ebv%ea-d%` |
| EBV_VCA_IGM | MEAS_ANY_RECORDED | measurement | LOINC `7885-7` | `%ebv%vca%igm%, %ebv%capsid%igm%` |
| CMV_PCR_ANY | MEAS_ANY_RECORDED | measurement | DESCENDANTS of `37172169` | — |
| GI_DX_ANY | COND_ANY_RECORDED | condition | DESCENDANTS `(196523, 200219, 27674)` | — |
| RESP_DX_ANY | COND_ANY_RECORDED | condition | DESCENDANTS `(254761, 312437)` | — |

Removed (0% prevalence in Mount Sinai OMOP): `SARS_COV2_NEUTRALIZING`.

### 3D. Defined but inactive (not in active viral lists)

From `ViralIndicator` (defined in code, not currently included in `VIRAL_INDICATOR_LIST`):

- `ANY_SARS_COV2_TEST`
- `ANY_POS_SARS_COV2_TEST`
- `EBV_ANTIBODY`
- `INFLUENZA_A_DX`
- `INFLUENZA_B_DX`
- `HEPATITIS_B_DX`
- `HEPATITIS_C_DX`
- `HEPATITIS_E_DX`
- `RSV_DX`
- `BRONCHOSCOPY`
- `GI_BIOPSY`
- `LIVER_BIOPSY`
- `SARS_COV2_ANTIBODY`
- `RECURRENT_RESPIRATORY_INFECTION`

From `EnhViralIndicator`: none inactive (`ENH_VIRAL_INDICATOR_LIST = list(EnhViralIndicator)`).

---

## 4. `mechsig_immuno_enh`

`baseline_ext` **plus** v1 (`IMMUNO_INDICATOR_LIST`) + v2
(`ENH_IMMUNO_INDICATOR_LIST`). When part of `mechsig_all_enh` the numeric
lab / CBC / composite features routed to *immuno* (see [§6](#6-mechsig_all_enh)) are also added.

### 4A. v1 Immuno

| Indicator | Kind | Source | Codes | `source_value` LIKE | Thresholds |
|---|---|---|---|---|---|
| ANY_IL6 | MEAS_OR_OBS_ANY | measurement + observation | LOINC `26881-3` + DIRECT `(4332015, 42529420)` + obs `(4150054,)` | — | any value |
| ANY_TNFA | MEAS_OR_OBS_ANY | measurement + observation | NAME_PATTERN `%TNF%ALPHA%, %TUMOR NECROSIS FACTOR%` + DIRECT `(4225604,)` + obs `(4216487,)` | — | any |
| ANY_IL12 | MEAS_ANY_RECORDED | measurement | DESCENDANTS `(37046229,)` | — | any |
| ANY_LYMPH_ABS | MEAS_ANY_RECORDED | measurement | LOINC `731-0` + DIRECT `(37208689,)` + DESCENDANTS `(4254663,)` | — | any numeric |
| ANY_NEUT_ABS | MEAS_ANY_RECORDED | measurement | LOINC `751-8` + DIRECT `(4148615,)` | — | any numeric |
| NLR_COMPUTABLE | MEAS_PAIRED_SAME_DATE_NUMERIC | measurement | left = LOINC `731-0, 26474-7`; right = LOINC `751-8, 26499-4` | — | same-date numeric pair |
| PERSISTENT_CRP | MEAS_ELEV_EPISODES_GAP | measurement | LOINC `1988-5, 30522-7` + DIRECT `(4208414,)` | — | gap=30 d, min=2 |
| NEUTROPHIL_ANTIBODY_OBS | OBS_ANY_RECORDED | observation | — | `%neutrophil%antibod%` | — |
| IMMUNOMOD_DRUGS | DRUG_INGREDIENT_DESC | drug | RxNorm ingredients: `prednisone, prednisolone, methylprednisolone, dexamethasone, hydrocortisone, methotrexate, azathioprine, mycophenolate mofetil, leflunomide, sulfasalazine, hydroxychloroquine, adalimumab, infliximab, etanercept, rituximab, tocilizumab, abatacept, ustekinumab, tofacitinib, baricitinib, upadacitinib` | paired brand-name LIKE patterns | — |
| PNEUMONIA_DX | COND_ANY_RECORDED | condition | DIRECT `(255848,)` | — | — |
| LYME_DISEASE_DX | COND_ANY_RECORDED | condition | DIRECT `(440638,)` | — | — |
| OPPORTUNISTIC_INFECTION | COND_ANY_RECORDED | condition | DESCENDANTS `(433701, 437663, 4140111, 432584, 440704)` | — | — |
| INFLUENZA_A_DX | COND_ANY_RECORDED | condition | DIRECT `(40483537,)` | — | — |
| INFLUENZA_B_DX | COND_ANY_RECORDED | condition | DIRECT `(4266367,)` | — | — |
| RSV_DX | COND_ANY_RECORDED | condition | DIRECT `(437222,)` | — | — |
| HEPATITIS_E_DX | COND_ANY_RECORDED | condition | DIRECT `(45769824,)` | — | — |
| RECURRENT_RESPIRATORY_INFECTION | COND_EPISODES_GAP | condition | DESCENDANTS `(4103703, 255848, 260139, 258780)` | — | gap=30 d, min=2 |

### 4B. v2 Enhanced Immuno (`ENH_IMMUNO_INDICATOR_LIST`)

| Indicator | Kind | Source | Codes | `source_value` LIKE |
|---|---|---|---|---|
| COMPLEMENT_C3_ORDERED | MEAS_ANY_RECORDED | measurement | LOINC `4485-9` | `%complement c3%, %complement 3%` |
| COMPLEMENT_C4_ORDERED | MEAS_ANY_RECORDED | measurement | LOINC `4498-2` | `%complement c4%, %complement 4%` |
| COMPLEMENT_CH50_ORDERED | MEAS_ANY_RECORDED | measurement | LOINC `4532-8` + NAME_PATTERN `%COMPLEMENT%CH50%, %COMPLEMENT%AH50%, %COMPLEMENT%TOTAL%HEMOLYTIC%` | `%ch50%, %ah50%, %total hemolytic complement%` |
| ANA_ORDERED | MEAS_ANY_RECORDED | measurement | LOINC `8061-4` | `%antinuclear%antibod%, %ana %screen%, %ana %titer%` |
| RF_ORDERED | MEAS_ANY_RECORDED | measurement | LOINC `11572-5` | `%rheumatoid factor%` |
| ANTI_CCP_ORDERED | MEAS_ANY_RECORDED | measurement | LOINC `53027-9` + NAME_PATTERN `%ANTI%CCP%, %CYCLIC CITRULLINATED PEPTIDE%` | `%anti%ccp%, %cyclic citrullinated%` |
| ANTIPHOSPHOLIPID_ORDERED | MEAS_ANY_RECORDED | measurement | NAME_PATTERN `%ANTIPHOSPHOLIPID%, %ANTICARDIOLIPIN%, %BETA-2 GLYCOPROTEIN%, %B2 GLYCOPROTEIN%, %LUPUS ANTICOAGULANT%` | same patterns lowercased |
| NEW_RHEUM_DX | COND_ANY_RECORDED | condition | DESCENDANTS `(257628, 80809, 134442, 4058824, 81893)` (RA, SLE, dermatomyositis, Sjögren, vasculitis) | — |
| JAK_INHIBITOR | DRUG_INGREDIENT_DESC | drug | RxNorm ingredients: `tofacitinib, baricitinib, ruxolitinib, upadacitinib` | `%tofacitinib%, %xeljanz%, %baricitinib%, %olumiant%, %ruxolitinib%` |
| DMARD_INITIATION | DRUG_INGREDIENT_DESC | drug | `methotrexate, hydroxychloroquine, sulfasalazine, leflunomide, azathioprine, mycophenolate` | per-drug brand-name LIKE |
| IVIG_EXPOSURE | DRUG_INGREDIENT_DESC | drug | `immune globulin` | `%ivig%, %gamunex%, %gammagard%, %privigen%, %octagam%, %immune globulin%intravenous%` |
| ANY_IL1B | MEAS_ANY_RECORDED | measurement | LOINC `47032-8` + NAME_PATTERN `%INTERLEUKIN%1%BETA%` | `%il-1%beta%, %il1b%, %interleukin 1b%` |
| CD4_COUNT | MEAS_ANY_RECORDED | measurement | LOINC `24467-3` + DIRECT `(37396514,)` | `%cd4%count%, %cd4%abs%, %t-helper%` |
| DSDNA_ANTIBODY | MEAS_ANY_RECORDED | measurement | LOINC `5130-0` | `%dsdna%, %double stranded dna%, %anti-dna%` |
| THYROID_AUTOANTIBODY | MEAS_ANY_RECORDED | measurement | LOINC `5382-7, 8098-6` | `%thyroid peroxidase%ab%, %anti%tpo%, %thyroglobulin%ab%` |

Removed (0–0.02% prevalence): `ANY_IFNG`, `CD8_COUNT`.

### 4C. Defined but inactive (not in active immuno lists)

From `ImmunoIndicator` (defined in code, not currently included in `IMMUNO_INDICATOR_LIST`):

- `ELEV_CRP`
- `ELEV_ESR`
- `ELEV_FERRITIN`
- `ANY_IL6_OBS`
- `ANY_TNFA_OBS`
- `IMMUNE_RELATED_DX`
- `SPECIALIST_VISITS`

From `EnhImmunoIndicator` (defined in code, not currently included in `ENH_IMMUNO_INDICATOR_LIST`):

- `RHEUMATOLOGY_VISIT`
- `IMMUNOLOGY_VISIT`

---

## 5. `mechsig_endo_enh`

`baseline_ext` **plus** v1 (`ENDO_INDICATOR_LIST`) + v2
(`ENH_ENDO_INDICATOR_LIST`).

### 5A. v1 Endo

| Indicator | Kind | Source | Codes / Patterns | Thresholds |
|---|---|---|---|---|
| ANTICOAG_THERAPY | DRUG_INGREDIENT_DESC | drug | RxNorm ingredients: `HEPARIN, WARFARIN, APIXABAN, RIVAROXABAN, DABIGATRAN ETEXILATE, EDOXABAN, ENOXAPARIN, DALTEPARIN, FONDAPARINUX` | — |
| REPEATED_ANTICOAG_EXPOSURE | DRUG_REPEAT_AT_LEAST_N | drug | same ingredients | count ≥ 2 |
| REPEATED_DDIMER_TESTING | MEAS_REPEAT_AT_LEAST_N | measurement | LOINC `48065-7, 3246-6, 71425-3` | count ≥ 2 |
| APHERESIS | PROC_NAME_PATTERN | procedure | `%PLASMAPHERESIS%, %THERAPEUTIC PLASMA EXCHANGE%, %PLASMA EXCHANGE%` + LIKE `%apheresis%, %plasmapheresis%, %plasma exchange%` | — |

### 5B. v2 Enhanced Endo

| Indicator | Kind | Source | Codes / Patterns |
|---|---|---|---|
| VWF_ORDERED | MEAS_ANY_RECORDED | measurement | LOINC `6014-5, 27816-8`; LIKE `%von willebrand%, %vwf%` |
| FACTOR_VIII_ORDERED | MEAS_ANY_RECORDED | measurement | LOINC `3209-4`; LIKE `%factor viii%, %factor 8%activity%` |
| DVT_PE | COND_NAME_PATTERN | condition | `%DEEP VEIN THROMBOSIS%, %PULMONARY EMBOLISM%, %PULMONARY THROMBOEMBOLISM%` |
| ARTERIAL_THROMBOSIS | COND_ANY_RECORDED | condition | DESCENDANTS `(312327, 4310996, 373503)` (MI, cerebral infarction, TIA) |
| MYOCARDITIS_PERICARDITIS | COND_ANY_RECORDED | condition | DESCENDANTS `(314383, 320116)`; LIKE `%myocardit%, %pericardit%` |
| AKI_DIAGNOSIS | COND_ANY_RECORDED | condition | DESCENDANTS `(197320,)` |
| PROTEINURIA_DX | COND_NAME_PATTERN | condition | `%PROTEINURIA%, %ALBUMINURIA%` |
| POTS_ORTHOSTATIC | COND_NAME_PATTERN | condition | `%POSTURAL ORTHOSTATIC TACHYCARDIA%, %ORTHOSTATIC HYPOTENSION%, %ORTHOSTATIC INTOLERANCE%`; LIKE `%pots%, %orthostatic%, %dysautonomia%, %postural tachycardia%` |
| ANTIPLATELET_THERAPY | DRUG_INGREDIENT_DESC | drug | `clopidogrel, prasugrel, ticagrelor, dipyridamole`; LIKE `%clopidogrel%, %plavix%, %ticagrelor%, %brilinta%` |
| ARRHYTHMIA_DX | COND_ANY_RECORDED | condition | DESCENDANTS `(313217, 314665, 316999, 4068155, 4103295)` |

Removed: `ADAMTS13_ORDERED`, `EXERCISE_INTOLERANCE_DX` (0% prevalence).
Dropped from active list as workup / redundancy proxies: `PT_INR_ORDERED`, `APTT_ORDERED`, `VASCULAR_IMAGING`, `CARDIAC_IMAGING_ORDERED`.

### 5C. v2 Exploratory (also attached to `mechsig_all_enh`)

| Indicator | Kind | Source | Codes / Patterns |
|---|---|---|---|
| TRYPTASE_ORDERED | MEAS_ANY_RECORDED | measurement | DIRECT `(4007807,)`; LIKE `%tryptase%` |
| H1_H2_BLOCKER | DRUG_INGREDIENT_DESC | drug | `cetirizine, loratadine, fexofenadine, diphenhydramine, hydroxyzine, famotidine, ranitidine` with brand-name LIKE |
| LEUKOTRIENE_ANTAGONIST | DRUG_INGREDIENT_DESC | drug | `montelukast, zafirlukast`; LIKE `%singulair%` |
| URTICARIA_DX | COND_NAME_PATTERN | condition | `%URTICARIA%, %FLUSHING%, %ANGIOEDEMA%` |

### 5D. Defined but inactive (not in active endo lists)

From `EndoIndicator` (defined in code, not currently included in `ENDO_INDICATOR_LIST`):

- `THROMBOTIC_EVENTS`
- `MICROVASCULAR_INJURY`
- `COAG_ACTIVATION_ANY`
- `COAG_DDIMER_ELEV_PROXY`
- `PLATELETS_ANY`
- `COMPLEMENT_ACTIVITY`

From `EnhEndoIndicator` (defined in code, not currently included in `ENH_ENDO_INDICATOR_LIST`):

- `VASCULAR_IMAGING`
- `PT_INR_ORDERED`
- `APTT_ORDERED`
- `CARDIAC_IMAGING_ORDERED`

---

## 6. `mechsig_all_enh`

Superset: `baseline_ext` + **every** v1 + v2 indicator above + all quantitative
labs, CBC indices, composites, and multi-window temporal-divergence features.
Built by `assemble_enhanced_feature_matrix()` and
`build_enhanced_model_configs()` in
[enhanced_mech_signals.py](../enhanced_mech_signals.py).

### 6A. Numeric Lab Trajectories (`NUMERIC_LAB_SPECS`)

Per lab, per feature window `{ws}_{we}` the extractor emits `_peak`, `_median`,
`_last`, `_abnormal_count`, `_abnormal_any` columns
(`f_lab_{window}_{name}_{stat}`).

| Lab | LOINC | DIRECT | Abnormal cut-off | Unit |
|---|---|---|---|---|
| crp | `1988-5, 30522-7, 71426-1` | — | `>10` | mg/L |
| esr | `30341-2, 18184-2` | — | `>20` | mm/hr |
| ferritin | `2276-4` | `4176561` | `>300` | ng/mL |
| ddimer | `48065-7, 48066-5, 48067-3, 71427-9` | — | `>0.5` | µg/mL FEU |
| fibrinogen | `3255-7` | — | `>400` | mg/dL |
| albumin | `1751-7` | — | `<3.5` | g/dL |
| ldh | `2532-0, 14805-6` | — | `>250` | U/L |
| creatinine | `2160-0` | — | `>1.2` | mg/dL |
| ast | `1920-8` | `4189605` | `>40` | U/L |
| alt | `1742-6` | `4189605` | `>40` | U/L |
| troponin | `6598-7, 10839-9, 49563-0, 89579-7` | — | `>0.04` | ng/mL |
| nt_probnp | `33762-6, 83107-3` | — | `>125` | pg/mL |
| lactate | `2524-7` | — | `>2` | mmol/L |
| platelets | `777-3, 26515-7` | `4267147` | `<150` or `>400` | 10³/µL |
| pt_inr | `34714-6, 6301-6` | — | `>1.1` | INR |

### 6B. CBC Components (`CBC_SPECS`) and Derived Indices

LOINC codes used to pull counts:

| Component | LOINC |
|---|---|
| neutrophils_abs | `751-8, 26499-4` |
| lymphocytes_abs | `731-0, 26474-7` |
| monocytes_abs | `742-7, 26484-6` |
| eosinophils_abs | `711-2, 26449-9` |
| rdw | `788-0, 30385-9` |

Derived (per window):

- `f_cbc_{w}_nlr_peak / _median` — neutrophils / lymphocytes
- `f_cbc_{w}_plr_peak / _median` — platelets / lymphocytes
- `f_cbc_{w}_sii_peak / _median` — (platelets × neutrophils) / lymphocytes
- `f_cbc_{w}_persistent_lymphopenia` — ≥2 measurements with lymph `<1.0`
- `f_cbc_{w}_monocytosis` — ≥1 measurement with mono `>1.0`
- `f_cbc_{w}_eosinophilia` — ≥1 measurement with eos `>0.5`
- `f_cbc_{w}_nlr_slope` — OLS slope of NLR over time
- `f_cbc_{w}_lymph_delta`, `f_cbc_{w}_lymph_slope`

### 6C. Composite Features

Complement + coagulation (from labs & binary signals):

- `f_comp_{w}_complement_ddimer` — complement present ∧ ddimer abnormal
- `f_comp_{w}_complement_crp_ddimer` — all three abnormal
- `f_comp_{w}_complement_crp_ddimer_count` — 0–3 count

Organ-injury (from labs):

- `f_comp_{w}_cardiac_injury` — troponin_abnormal ∨ nt_probnp_abnormal
- `f_comp_{w}_cardiac_injury_score` — max severity ratio
- `f_comp_{w}_renal_injury` — creatinine_abnormal
- `f_comp_{w}_renal_injury_ratio` — creatinine ÷ 1.2

GI-persistence cluster (window 30–180 d, built in
`build_composite_features()`):

| Feature | Source | Codes / LIKE |
|---|---|---|
| f_comp_gi_chronic_diarrhea | condition | DESCENDANTS `(196523, 4091513, 196152)` |
| f_comp_gi_abdominal_pain | condition | DESCENDANTS `(200219, 4103703)` |
| f_comp_gi_nausea_vomiting | condition | DESCENDANTS `(27674, 4101344)` |
| f_comp_gi_endoscopy | procedure | `concept_name LIKE '%ENDOSCOP%, %COLONOSCOP%, %SIGMOIDOSCOP%'` |
| f_comp_gi_biopsy | procedure | LIKE `%BIOPSY%GI%, %BIOPSY%INTESTIN%, %BIOPSY%COLON%, %BIOPSY%GASTRIC%` |
| f_comp_gi_ppi | drug | ingredients `omeprazole, pantoprazole, lansoprazole, esomeprazole, rabeprazole` |
| f_comp_gi_antidiarrheal | drug | ingredients `loperamide, bismuth, cholestyramine, diphenoxylate` |
| f_comp_gi_any | derived | OR of the above |
| f_comp_gi_burden_score | derived | count (0–7) |

### 6D. Temporal-Divergence Composites (multi-window)

Built by `build_temporal_divergence_composites()`. Fires when window
`start_day ≥ post_acute_days` (default 30) **and**
`end_day ≤ max_window_end_day` (leakage guard).

| Composite | Substrings matched on binary signals | Mechanism |
|---|---|---|
| f_comp_td_prolonged_inflammation | `immuno_elev_crp, immuno_elev_esr` | immuno |
| f_comp_td_post_acute_respiratory | `enh_viral_resp_dx_any` | viral |
| f_comp_td_sustained_coag | `endo_coag_activation_any, endo_coag_ddimer_elev_proxy, endo_repeated_ddimer_testing` | endo |
| f_comp_td_emerging_dysautonomia | `enh_endo_pots_orthostatic` | endo |
| f_comp_td_late_autoimmune | `enh_immuno_ana_ordered, enh_immuno_antiphospholipid_ordered, enh_immuno_rheumatology_visit, enh_immuno_new_rheum_dx` | immuno |
| f_comp_td_persistent_cytokine | `immuno_any_il6, immuno_any_tnfa` | immuno |
| f_comp_td_unresolved_thrombosis | `enh_endo_dvt_pe, endo_thrombotic_events` | endo |
| f_comp_td_multisystem_late | count of mechanisms (0–3) with ≥1 post-acute hit | all |

### 6E. Routing of labs / CBC / composites into mech-signal configs

`build_enhanced_model_configs()` assigns each quantitative feature to **exactly
one** single-mechanism config; `mechsig_all_enh` receives all of them.

- **→ immuno:** crp, esr, ferritin, albumin, ldh, lactate, NLR, SII,
  persistent_lymphopenia, monocytosis, eosinophilia, complement + CRP
  composites.
- **→ endo:** ddimer, fibrinogen, pt_inr, aptt, troponin, nt_probnp,
  creatinine, platelets, ast, alt, PLR, RDW, cardiac/renal-injury composites,
  coag + thrombosis composites, dysautonomia composites.
- **→ viral:** GI-persistence composites, post-acute respiratory divergence.
- **Only in `mechsig_all_enh`:** `f_comp_td_multisystem_late`.

---

## 7. OMOP2OBO / HPO symptom mapping

When `use_hpo_symptoms=True`, Family B symptoms are resolved through the
**OMOP2OBO v2.0.0 N3C Enclave** mapping artifact instead of SNOMED
`concept_ancestor`.

- **Artifact:** `data/omop2obo/OMOP2OBO_v2.0.0_N3C_Enclave_CSV_concept_set_expression_items.csv`
- **Module:** [omop2obo_mapping.py](../omop2obo_mapping.py)

### 7A. CSV columns consumed

| Column | Use |
|---|---|
| `concept_id` | OMOP standard concept ID |
| `ontology_id` | HPO curie (e.g. `HP:0001945`) |
| `ontology_label` | HPO term label |
| `mapping_category` | quality tier (see below) |

### 7B. Quality-tier filter (`TRUSTED_CATEGORIES`)

Kept by default:

- Automatic One-to-One Concept
- Automatic One-to-Many Concept
- Automatic One-to-One Ancestor
- Automatic One-to-Many Ancestor
- Manual One-to-One Concept
- Manual One-to-Many Concept

Excluded: *Cosine Similarity One-to-One Concept*.

### 7C. Target HPO set

Core 14 Antony symptoms + 5 extras (listed in [§1B](#1b-acute-symptoms--family-b)). Each
`f_sym_hpo_*` feature is fired when any of the OMOP concept IDs mapped to the
target HPO curie appears in the patient's acute-window
`condition_occurrence`.

### 7D. Runtime flow (`extract_acute_symptoms_hpo`)

1. Load CSV, filter to `TRUSTED_CATEGORIES`.
2. `build_concept_to_hpo_lookup()` → `dict[concept_id] → set[HP:xxxx]`.
3. Query acute-window `condition_occurrence` for the cohort — one
   `(person_id, condition_concept_id)` row per patient/condition.
4. For each row, look up HPO terms; if any match a target HPO, set
   `f_sym_hpo_<symptom> = 1`.
5. Result merged into the feature matrix alongside (or replacing) the
   SNOMED-ancestor `f_sym_*` columns.

### 7E. Reverse map used for sanity checks

`SNOMED_ANCESTOR_TO_HPO` maps each SNOMED ancestor concept ID back to its HPO
curie, enabling validation that the HPO path reproduces the SNOMED-based
cohort extraction.

---

## 8. Indicator-kind & concept-source glossary

### 8A. `IndicatorKind` → SQL shape

Defined in [mech_signals_common.py](../mech_signals_common.py).

| Kind | Semantics |
|---|---|
| `MEAS_ANY_RECORDED` | ≥1 row in `measurement` matching concepts |
| `MEAS_NUMERIC_ANY` | ≥1 measurement row with `value_as_number IS NOT NULL` |
| `MEAS_NUMERIC_THRESHOLD` | ≥1 measurement with `value_as_number ≥ threshold` |
| `MEAS_ELEVATED_OR_ANY` | `value_as_number > range_high` **or** any recorded |
| `MEAS_PAIRED_SAME_DATE_NUMERIC` | two measurement concepts on same `measurement_date`, both numeric |
| `MEAS_REPEAT_AT_LEAST_N` | `GROUP BY person_id HAVING COUNT(*) ≥ N` |
| `MEAS_ELEV_EPISODES_GAP` | ≥N elevated measurements separated by ≥`gap_days` |
| `MEAS_OR_OBS_ANY` | UNION of measurement + observation rows |
| `OBS_ANY_RECORDED` | ≥1 row in `observation` |
| `COND_ANY_RECORDED` | ≥1 row in `condition_occurrence` |
| `COND_EPISODES_GAP` | ≥N condition episodes with gap clustering |
| `COND_NAME_PATTERN` | `UPPER(concept_name) LIKE` any pattern |
| `PROC_ANY_RECORDED` | ≥1 row in `procedure_occurrence` |
| `PROC_NAME_PATTERN` | procedure name LIKE any pattern |
| `DRUG_ANY_RECORDED` | ≥1 row in `drug_exposure` |
| `DRUG_INGREDIENT_DESC` | ingredient via `concept_ancestor` descendants |
| `DRUG_REPEAT_AT_LEAST_N` | `COUNT(*) ≥ N` in `drug_exposure` |
| `DRUG_EPISODES_GAP` | ≥N drug episodes with gap clustering |
| `VISIT_SPECIALTY_NAME` | provider specialty LIKE pattern via `visit_occurrence ↔ provider` |

### 8B. `ConceptSourceKind` → concept resolution

| Kind | Resolution |
|---|---|
| `DIRECT_IDS` | use the provided concept IDs as-is |
| `LOINC_CODES` | `WHERE vocabulary_id='LOINC' AND concept_code IN (...)` |
| `DESCENDANTS` | `SELECT descendant_concept_id FROM concept_ancestor WHERE ancestor_concept_id IN (...)` |
| `NAME_PATTERN` | `UPPER(concept_name) LIKE` against `concept`, optionally filtered by `domain_id` |
| `MIXED` | union of `DIRECT_IDS` + `LOINC_CODES` |

### 8C. Source-value fallback

In addition to concept-based lookup, `source_value_patterns` are applied as

```sql
OR LOWER(<domain>_source_value) LIKE '<pattern1>'
OR LOWER(<domain>_source_value) LIKE '<pattern2>' ...
```

to catch locally-coded strings that never resolve to a standard concept
(common for in-house lab panels, brand-name drug entries, and non-OMOP-mapped
notes).

### 8D. Caching

Resolved concept ID sets are memoised in `_CONCEPT_ID_CACHE`
(process-lifetime) to avoid repeat `concept`/`concept_ancestor` round-trips.

---

## Appendix: OMOP domain → where each feature family lives

| Domain | Used by |
|---|---|
| `condition_occurrence` | Family A/B, extended comorbidities/symptoms, all `COND_*` indicators |
| `measurement` | BMI/BP/SpO₂, serology, all numeric labs, CBC, all `MEAS_*` indicators |
| `observation` | smoking / alcohol / vent-mode / ICU markers, all `OBS_*` indicators |
| `drug_exposure` | Family C, vaccination, corticosteroids, antivirals, anticoagulants, all `DRUG_*` indicators |
| `procedure_occurrence` | Family E (IMV, ECMO), biopsies, apheresis, imaging, all `PROC_*` indicators |
| `visit_occurrence` | LOS, ICU/ED classification, specialty visits |

All window clauses use the cohort temp table `"#antony_cohort"`
(`acute_start`, `acute_end`, `covid_index_date`) joined per-patient, so every
feature is deterministically time-boxed.

"""The configuration builder produces the sets of thesis Table 3.2 and 3.9 with the right composition."""
import pytest

from pasc.features.enhanced import build_enhanced_model_configs

THESIS_CONFIGS = {
    "baseline_antony", "baseline_antony_eng", "baseline_ext",
    "mechsig_viral", "mechsig_immuno", "mechsig_endo", "mechsig_all",
    "baseline_ext_no_td", "mechsig_all_no_td",
    "baseline_ext_no_eng", "mechsig_viral_no_eng", "mechsig_immuno_no_eng",
    "mechsig_endo_no_eng", "mechsig_all_no_eng",
}


@pytest.fixture
def families():
    return {
        "antony_comorbidities": ["f_comor_a", "f_comor_b"],
        "antony_symptoms": ["f_sym_hpo_x"],
        "antony_drugs": ["f_drug_d"],
        "antony_demographics": ["f_age", "f_sex_female"],
        "antony_treatment": ["f_tx_los"],
        "ext_features": ["f_ext_bmi_last", "f_ext_index_year_month"],
        "engagement_controls": ["f_eng_n_encounters_pre_index_1y", "f_eng_has_pcp_pre_index"],
        "mechsig_viral": ["f_ind_viral_w0_90_a"],
        "mechsig_immuno": ["f_ind_immuno_w0_90_b"],
        "mechsig_endo": ["f_ind_endo_w0_90_c"],
        "mechsig_exploratory": ["f_ind_expl_w0_90_tryptase_ordered"],
        "numeric_labs": ["f_lab_w0_90_crp_peak", "f_lab_w0_90_troponin_peak"],
        "cbc_indices": ["f_cbc_w0_90_nlr_peak", "f_cbc_w0_90_plr_peak"],
        "lab_trajectories": [],
        "temporal_trends": [],
        "composites": ["f_comp_w0_90_complement_ddimer", "f_comp_td_multisystem_late",
                       "f_comp_td_post_acute_respiratory"],
        "cross_window_summaries": [],
    }


def test_all_thesis_configurations_are_built(families):
    configs = build_enhanced_model_configs(families)
    assert THESIS_CONFIGS <= set(configs)


def test_configurations_nest_as_in_table_3_2(families):
    c = build_enhanced_model_configs(families)
    antony, ext = set(c["baseline_antony"]), set(c["baseline_ext"])
    assert antony < set(c["baseline_antony_eng"]) < ext
    for single in ("mechsig_viral", "mechsig_immuno", "mechsig_endo"):
        assert ext < set(c[single]) <= set(c["mechsig_all"])


def test_single_cluster_configurations_share_only_the_baseline(families):
    c = build_enhanced_model_configs(families)
    ext = set(c["baseline_ext"])
    added = {k: set(c[k]) - ext for k in ("mechsig_viral", "mechsig_immuno", "mechsig_endo")}
    assert added["mechsig_viral"].isdisjoint(added["mechsig_immuno"])
    assert added["mechsig_viral"].isdisjoint(added["mechsig_endo"])
    assert added["mechsig_immuno"].isdisjoint(added["mechsig_endo"])


def test_non_cluster_features_appear_in_mechsig_all_only(families):
    c = build_enhanced_model_configs(families)
    for col in ("f_ind_expl_w0_90_tryptase_ordered", "f_comp_td_multisystem_late"):
        assert col in c["mechsig_all"]
        for single in ("mechsig_viral", "mechsig_immuno", "mechsig_endo", "baseline_ext"):
            assert col not in c[single]


def test_refits_strip_exactly_one_family(families):
    c = build_enhanced_model_configs(families)
    for base in ("baseline_ext", "mechsig_all"):
        no_td = c[f"{base}_no_td"]
        assert not any(col.startswith("f_comp_td_") for col in no_td)
        assert set(no_td) == {col for col in c[base] if not col.startswith("f_comp_td_")}
        no_eng = c[f"{base}_no_eng"]
        assert not any(col.startswith("f_eng_") for col in no_eng)
        assert set(no_eng) == {col for col in c[base] if not col.startswith("f_eng_")}


def test_engagement_controls_are_in_every_configuration_from_baseline_ext_on(families):
    c = build_enhanced_model_configs(families)
    eng = set(families["engagement_controls"])
    for name in THESIS_CONFIGS - {"baseline_antony"}:
        if name.endswith("_no_eng"):
            assert eng.isdisjoint(c[name])
        else:
            assert eng <= set(c[name])

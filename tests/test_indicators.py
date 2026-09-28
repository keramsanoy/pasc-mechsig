"""The indicator catalogue matches thesis Table 3.3: 24 viral, 32 immunoinflammatory, 14 endothelial."""
from pasc.features.enhanced import (
    ENH_ENDO_INDICATOR_LIST,
    ENH_EXPLORATORY_INDICATOR_LIST,
    ENH_IMMUNO_INDICATOR_LIST,
    ENH_VIRAL_INDICATOR_LIST,
)
from pasc.features.indicators import ENDO_INDICATOR_LIST, IMMUNO_INDICATOR_LIST, VIRAL_INDICATOR_LIST


def _names(*lists):
    return [ind.name for lst in lists for ind in lst]


def test_cluster_sizes_match_table_3_3():
    assert len(VIRAL_INDICATOR_LIST) + len(ENH_VIRAL_INDICATOR_LIST) == 24
    assert len(IMMUNO_INDICATOR_LIST) + len(ENH_IMMUNO_INDICATOR_LIST) == 32
    assert len(ENDO_INDICATOR_LIST) + len(ENH_ENDO_INDICATOR_LIST) == 14
    assert len(ENH_EXPLORATORY_INDICATOR_LIST) == 4  # mast-cell indicators outside the clusters


def test_indicator_names_are_unique_within_and_across_clusters():
    names = _names(VIRAL_INDICATOR_LIST, ENH_VIRAL_INDICATOR_LIST,
                   IMMUNO_INDICATOR_LIST, ENH_IMMUNO_INDICATOR_LIST,
                   ENDO_INDICATOR_LIST, ENH_ENDO_INDICATOR_LIST,
                   ENH_EXPLORATORY_INDICATOR_LIST)
    assert len(names) == len(set(names)) == 74


def test_every_indicator_names_a_cdm_event_table():
    tables = {"CDMPHI.measurement", "CDMPHI.observation", "CDMPHI.condition_occurrence",
              "CDMPHI.procedure_occurrence", "CDMPHI.drug_exposure", "CDMPHI.visit_occurrence"}
    for lst in (VIRAL_INDICATOR_LIST, ENH_VIRAL_INDICATOR_LIST, IMMUNO_INDICATOR_LIST,
                ENH_IMMUNO_INDICATOR_LIST, ENDO_INDICATOR_LIST, ENH_ENDO_INDICATOR_LIST,
                ENH_EXPLORATORY_INDICATOR_LIST):
        for ind in lst:
            spec = ind.value
            assert spec.table in tables or spec.constant_zero, (ind.name, spec.table)

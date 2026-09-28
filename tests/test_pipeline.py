"""One repeat of the modelling pipeline on synthetic data (thesis Table 3.5)."""
import numpy as np
import pytest

from pasc.modeling.pipeline import balance_training_set, prevalence_filter, run_single_iteration


def test_balancing_keeps_every_case_and_matches_controls():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = (rng.random(500) < 0.1).astype(int)
    Xb, yb = balance_training_set(X, y, random_state=1)
    assert yb.sum() == y.sum()
    assert (yb == 0).sum() == y.sum()


def test_prevalence_filter_drops_rare_binaries_but_keeps_forced_and_continuous():
    X = np.zeros((200, 3))
    X[:1, 0] = 1                      # binary, 0.5 % prevalence -> dropped
    X[:1, 1] = 1                      # same, but forced -> kept
    X[:, 2] = np.arange(200) / 10.0   # continuous -> kept
    kept_idx, kept, n_dropped = prevalence_filter(X, ["rare", "rare_forced", "cont"],
                                                  min_prev=0.01, forced_features=["rare_forced"])
    assert kept == ["rare_forced", "cont"] and n_dropped == 1 and kept_idx == [1, 2]


@pytest.mark.slow
def test_single_iteration_is_seeded_and_reproducible():
    rng = np.random.default_rng(42)
    n, p = 600, 8
    X = rng.normal(size=(n, p))
    X[:, 4:] = (rng.random((n, p - 4)) < 0.3).astype(float)
    logit = 1.5 * X[:, 0] - 1.0 * X[:, 5] - 2.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    names = [f"f_{i}" for i in range(p)]

    def run():
        return run_single_iteration(
            X, y, names, iteration=1, model_type="RF", cohort_name="test",
            random_state_base=42, boruta_max_iter=10, boruta_n_estimators=50,
            compute_shap=True, forced_features=["f_7"], boruta_perc=97,
        )

    a, b = run(), run()
    assert a.seed == 42 + 1000
    assert 0.5 < a.auroc <= 1.0
    assert a.auroc == b.auroc and a.auprc == b.auprc
    assert a.selected_features == b.selected_features
    assert "f_7" in a.selected_features
    assert a.shap_values is not None and a.shap_values.shape[1] == len(a.shap_feature_names)

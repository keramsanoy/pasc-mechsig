"""Paired comparisons (thesis 3.8) on a synthetic all_iterations table."""
import numpy as np
import pandas as pd

from pasc.analysis.paired import GROUPS, WINDOWS, benjamini_hochberg, paired_diff, paired_tests


def _iterations(n=100, lift=0.02, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in WINDOWS:
        for it in range(1, n + 1):
            base = 0.80 + rng.normal(0, 0.01)
            rows.append(dict(config="baseline_ext", feature_window=w, iteration=it, auroc=base, auprc=base / 8))
            rows.append(dict(config="mechsig_all", feature_window=w, iteration=it,
                             auroc=base + lift + rng.normal(0, 0.002), auprc=base / 8))
    return pd.DataFrame(rows)


def test_paired_diff_pairs_on_iteration():
    df = _iterations()
    d = paired_diff(df, "mechsig_all", "baseline_ext", "w0_90", "auroc")
    assert len(d) == 100 and abs(d.median() - 0.02) < 0.002


def test_paired_tests_report_one_row_per_window_and_metric_with_bh():
    out = paired_tests(_iterations(), groups={"H3H4_plain": GROUPS["H3H4_plain"]})
    assert len(out) == 2 * len(WINDOWS)
    auroc = out[out.metric == "AUROC"]
    assert (auroc.bh_group_size == len(WINDOWS)).all()
    assert auroc["sig_bh_0.05"].all() and (auroc.share_positive > 0.95).all()
    auprc = out[out.metric == "AUPRC"]
    assert (auprc["median"].abs() < 1e-9).all() and not auprc["sig_bh_0.05"].any()


def test_bh_is_monotone_and_bounded():
    p = np.array([0.001, 0.04, 0.03, 0.5])
    adj = benjamini_hochberg(p)
    assert np.all(adj >= p) and np.all(adj <= 1)
    assert adj[0] <= adj[2] <= adj[1] <= adj[3]

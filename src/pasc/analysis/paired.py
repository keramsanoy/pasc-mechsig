"""Paired comparisons between configurations (thesis 3.8).

Every configuration is fitted on the same 100 divisions of the same patients
(seed 42 + 1000*i for repeat i), so two configurations are compared repeat by
repeat: the paired difference of AUROC or AUPRC, its median and interquartile
range, the share of repeats in which it is positive, a Wilcoxon signed-rank
test, and Benjamini–Hochberg correction within the group of comparisons that is
reported together (separately for AUROC and AUPRC).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

WINDOWS = ["w0_21", "w0_30", "w0_60", "w0_90", "w30_60", "w60_90"]
METRICS = ["auroc", "auprc"]

# One correction group per reported table; each (config_a, config_b) pair is
# tested in every window, so a group of one pair has six tests per metric.
GROUPS = {
    "H2": [("baseline_ext", "baseline_antony")],
    "H3H4_plain": [("mechsig_all", "baseline_ext")],
    "H3H4_no_td": [("mechsig_all_no_td", "baseline_ext_no_td")],
    "H3H4_no_eng": [("mechsig_all_no_eng", "baseline_ext_no_eng")],
    "clusters": [("mechsig_viral", "baseline_ext"),
                 ("mechsig_immuno", "baseline_ext"),
                 ("mechsig_endo", "baseline_ext")],
    "eng_step": [("baseline_antony_eng", "baseline_antony")],
}


def benjamini_hochberg(p) -> np.ndarray:
    """BH-adjusted p-values (step-up), holding the expected false-discovery share."""
    try:  # scipy >= 1.11
        from scipy.stats import false_discovery_control
        return np.asarray(false_discovery_control(np.asarray(p, dtype=float), method="bh"))
    except ImportError:  # pragma: no cover
        p = np.asarray(p, dtype=float)
        n = len(p)
        order = np.argsort(p)
        ranked = p[order] * n / (np.arange(n) + 1)
        adj = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(n)
        out[order] = np.minimum(adj, 1.0)
        return out


def paired_diff(df: pd.DataFrame, a: str, b: str, window: str, metric: str) -> pd.Series:
    """Per-repeat difference metric(a) - metric(b) in one window, indexed by iteration."""
    A = df[(df.config == a) & (df.feature_window == window)].set_index("iteration")[metric]
    B = df[(df.config == b) & (df.feature_window == window)].set_index("iteration")[metric]
    return (A - B).dropna()


def paired_tests(df: pd.DataFrame, groups: dict | None = None, windows=WINDOWS, metrics=METRICS) -> pd.DataFrame:
    """One row per (group, metric, pair, window) with the paired statistics and BH-adjusted p.

    ``df`` is ``all_iterations.csv`` as written by scripts/run_main_analysis.py:
    one row per (config, feature_window, iteration) with ``auroc`` and ``auprc``.
    """
    groups = GROUPS if groups is None else groups
    rows = []
    for group, pairs in groups.items():
        for metric in metrics:
            block = []
            for a, b in pairs:
                for w in windows:
                    d = paired_diff(df, a, b, w, metric)
                    if len(d) == 0:
                        continue
                    p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
                    block.append(dict(group=group, metric=metric.upper(), config_a=a, config_b=b,
                                      window=w, n_pairs=len(d), median=d.median(),
                                      q1=d.quantile(.25), q3=d.quantile(.75),
                                      share_positive=float((d > 0).mean()), p_wilcoxon=p))
            if not block:
                continue
            padj = benjamini_hochberg([r["p_wilcoxon"] for r in block])
            for r, pa in zip(block, padj):
                r["p_bh"] = float(pa)
                r["sig_bh_0.05"] = bool(pa < 0.05)
                r["bh_group_size"] = len(block)
            rows.extend(block)
    return pd.DataFrame(rows)

#!/usr/bin/env python
"""Paired Wilcoxon signed-rank + Benjamini-Hochberg for the thesis comparisons (thesis 3.8).

Reads results/main/perc<PERC>/all_iterations.csv (written by run_enhanced_mechsig.py), pairs
configurations by iteration (identical splits, seeds 42 + 1000*i), and writes one row per
(comparison, window, metric) with the median paired difference, IQR, share of repeats with a
positive difference, raw Wilcoxon p, BH-adjusted p and the size of the correction group.

Correction groups (one per reported table, AUROC and AUPRC corrected separately):
  H2            baseline_ext vs baseline_antony              6 windows
  H3/H4 plain   mechsig_all vs baseline_ext                  6 windows
  H3/H4 no_td   mechsig_all_no_td vs baseline_ext_no_td      6 windows
  H3/H4 no_eng  mechsig_all_no_eng vs baseline_ext_no_eng    6 windows
  clusters      {viral,immuno,endo} vs baseline_ext          18 (3 x 6)
  eng_step      baseline_antony_eng vs baseline_antony       6 windows (decomposition of H2)

Replaces the unpaired Mann-Whitney files mw_*.csv, which must not be cited in the thesis.

Usage:
  python scripts/paired_wilcoxon_bh.py            # perc97, writes paired_wilcoxon_bh.csv next to all_iterations.csv
  python scripts/paired_wilcoxon_bh.py --perc 97 --out /path/to/file.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
from pasc_paths import MAIN_RESULTS_DIR  # noqa: E402

try:  # scipy >= 1.11
    from scipy.stats import false_discovery_control

    def _bh(p):
        return false_discovery_control(p, method="bh")
except ImportError:  # manual BH
    def _bh(p):
        p = np.asarray(p, dtype=float)
        n = len(p)
        order = np.argsort(p)
        ranked = p[order] * n / (np.arange(n) + 1)
        adj = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(n)
        out[order] = np.minimum(adj, 1.0)
        return out

WINDOWS = ["w0_21", "w0_30", "w0_60", "w0_90", "w30_60", "w60_90"]
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
METRICS = ["auroc", "auprc"]


def paired_diff(df, a, b, window, metric):
    A = df[(df.config == a) & (df.feature_window == window)].set_index("iteration")[metric]
    B = df[(df.config == b) & (df.feature_window == window)].set_index("iteration")[metric]
    return (A - B).dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perc", type=int, default=97)
    ap.add_argument("--results-root", default=str(MAIN_RESULTS_DIR))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    perc_dir = os.path.join(args.results_root, f"perc{args.perc}")
    df = pd.read_csv(os.path.join(perc_dir, "all_iterations.csv"))
    rows = []
    for group, pairs in GROUPS.items():
        for metric in METRICS:
            block = []
            for a, b in pairs:
                for w in WINDOWS:
                    d = paired_diff(df, a, b, w, metric)
                    if len(d) == 0:
                        continue
                    p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
                    block.append(dict(group=group, metric=metric.upper(), config_a=a, config_b=b,
                                      window=w, n_pairs=len(d), median=d.median(),
                                      q1=d.quantile(.25), q3=d.quantile(.75),
                                      share_positive=float((d > 0).mean()), p_wilcoxon=p))
            padj = _bh([r["p_wilcoxon"] for r in block])
            for r, pa in zip(block, padj):
                r["p_bh"] = pa
                r["sig_bh_0.05"] = bool(pa < 0.05)
                r["bh_group_size"] = len(block)
            rows.extend(block)
    out = pd.DataFrame(rows)
    out_path = args.out or os.path.join(perc_dir, "paired_wilcoxon_bh.csv")
    out.to_csv(out_path, index=False)
    pd.set_option("display.width", 220)
    print(out.round(4).to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Paired Wilcoxon signed-rank + Benjamini-Hochberg for the thesis comparisons (thesis 3.8).

Reads results/main/perc<PERC>/all_iterations.csv (written by scripts/run_main_analysis.py), pairs
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
  python scripts/paired_tests.py            # perc97, writes paired_wilcoxon_bh.csv next to all_iterations.csv
  python scripts/paired_tests.py --perc 97 --out /path/to/file.csv
"""
import argparse
import os

import pandas as pd

import _bootstrap  # noqa: E402,F401  (makes `pasc` importable from a plain clone)
from pasc.config.paths import MAIN_RESULTS_DIR  # noqa: E402

from pasc.analysis.paired import paired_tests  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perc", type=int, default=97)
    ap.add_argument("--results-root", default=str(MAIN_RESULTS_DIR))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    perc_dir = os.path.join(args.results_root, f"perc{args.perc}")
    df = pd.read_csv(os.path.join(perc_dir, "all_iterations.csv"))
    out = paired_tests(df)
    out_path = args.out or os.path.join(perc_dir, "paired_wilcoxon_bh.csv")
    out.to_csv(out_path, index=False)
    pd.set_option("display.width", 220)
    print(out.round(4).to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

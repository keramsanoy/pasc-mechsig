#!/usr/bin/env python
"""
scripts/make_lgbm_table.py -- thesis Table B.5 (LightGBM robustness) from the final-cohort rerun.

Reads results/lgbm_robustness/all_iterations_lgbm.csv and the reference Random Forest
iterations_*.csv for the same windows, pairs each mechanism arm against its MATCHED reference
on `iteration` (identical splits, seeds 42 + 1000*i) and reports, per arm, window and estimator:
median paired delta, IQR, share of repeats with delta > 0, paired Wilcoxon p and BH-adjusted p.

  plain    mechsig_all        - baseline_ext
  no_td    mechsig_all_no_td  - baseline_ext_no_td
  no_eng   mechsig_all_no_eng - baseline_ext_no_eng

The statistics mirror scripts/paired_tests.py exactly (same wilcoxon call, same BH), and
each arm is its own correction group, as in that script. BH family: the two LightGBM tests per
(arm, measure), one per window.

The RF rows are recomputed ONLY as a check: the plain arm against the published Table 5.x values
and every arm against perc97/paired_wilcoxon_bh.csv. If any disagree, this script stops.

Usage:
  python scripts/make_lgbm_table.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon



import _bootstrap  # noqa: E402,F401  (makes `pasc` importable from a plain clone)
from pasc.config.paths import REPO_ROOT, MAIN_RESULTS_DIR, LGBM_RESULTS_DIR  # noqa: E402
from pasc.analysis.paired import benjamini_hochberg as _bh  # noqa: E402

BASE = str(REPO_ROOT)

LGBM_DIR = str(LGBM_RESULTS_DIR)
RF_DIR = os.path.join(str(MAIN_RESULTS_DIR), "perc97")
WINDOWS = ["w0_90", "w60_90"]
WINDOW_TEX = {"w0_90": "w0--90", "w60_90": "w60--90"}
METRICS = ["auroc", "auprc"]
TOL = 5e-4

ARM_PAIRS = {
    "plain":  ("mechsig_all", "baseline_ext"),
    "no_td":  ("mechsig_all_no_td", "baseline_ext_no_td"),
    "no_eng": ("mechsig_all_no_eng", "baseline_ext_no_eng"),
}
ARM_WIL_GROUP = {"plain": "H3H4_plain", "no_td": "H3H4_no_td", "no_eng": "H3H4_no_eng"}
ARM_TEX = {"plain": "LightGBM",
           "no_td": "LightGBM, composites removed",
           "no_eng": "LightGBM, engagement stripped"}

# Published Table 5.x values the plain-arm RF recomputation must reproduce.
RF_EXPECTED_PLAIN = {
    ("w0_90", "auroc"): (0.001, -0.003, 0.006, 57),
    ("w0_90", "auprc"): (-0.005, -0.014, 0.003, 33),
    ("w60_90", "auroc"): (0.020, 0.017, 0.024, 100),
    ("w60_90", "auprc"): (0.004, -0.003, 0.010, 68),
}


def die(msg):
    print(f"\nABORT: {msg}\n", flush=True)
    raise SystemExit(1)


# ----------------------------------------------------------------------------- loading
def load_lgbm():
    path = os.path.join(LGBM_DIR, "all_iterations_lgbm.csv")
    if not os.path.exists(path):
        die(f"missing {path}; run scripts/run_lgbm_check.py first")
    return pd.read_csv(path)


def load_rf(configs):
    frames = []
    for w in WINDOWS:
        for cfg in configs:
            path = os.path.join(RF_DIR, w, "strict", f"iterations_{cfg}.csv")
            if not os.path.exists(path):
                die(f"missing RF reference: {path}")
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def paired_diff(df, arm, window, metric):
    """treatment - reference on shared iterations, with a seed-equality check."""
    treat, ref = ARM_PAIRS[arm]
    a = df[(df.feature_window == window) & (df.config == treat)].set_index("iteration")
    b = df[(df.feature_window == window) & (df.config == ref)].set_index("iteration")
    shared = a.index.intersection(b.index)
    if len(shared) == 0:
        die(f"no shared iterations for {arm} at {window}")
    sa, sb = a.loc[shared, "seed"], b.loc[shared, "seed"]
    if not (sa.values == sb.values).all():
        bad = shared[sa.values != sb.values].tolist()
        die(f"{arm} {window}: seeds differ between configs at iterations {bad[:5]}; "
            f"the differences are not paired")
    return (a.loc[shared, metric] - b.loc[shared, metric]).dropna()


def describe(d):
    return dict(
        n_pairs=int(len(d)),
        median=float(d.median()),
        q25=float(d.quantile(0.25)),
        q75=float(d.quantile(0.75)),
        n_positive=int((d > 0).sum()),
        p_wilcoxon=float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0,
    )


# ----------------------------------------------------------------------------- formatting
def tex(x):
    return f"${x:+.3f}$"


def cell(r):
    return f"{tex(r['median'])} [{tex(r['q25'])}, {tex(r['q75'])}]"


def txt(r):
    return (f"{r['median']:+.3f} [{r['q25']:+.3f}, {r['q75']:+.3f}]  "
            f"{r['n_positive']}/{r['n_pairs']}")


def main():
    lgbm = load_lgbm()
    arms = [a for a in ARM_PAIRS if all(c in set(lgbm.config) for c in ARM_PAIRS[a])]
    if not arms:
        die("all_iterations_lgbm.csv contains none of the arm config pairs")
    skipped = [a for a in ARM_PAIRS if a not in arms]
    if skipped:
        print(f"note: no LightGBM rows for arm(s) {skipped}; reporting {arms}\n")

    configs = sorted({c for a in arms for c in ARM_PAIRS[a]})
    rf = load_rf(configs)

    missing = [w for w in WINDOWS if w not in set(lgbm.feature_window)]
    if missing:
        die(f"all_iterations_lgbm.csv has no rows for {missing}")

    # ---------------------------------------------------------------- RF check (must reproduce)
    print("=" * 78)
    print("RF CHECK -- recomputed from iterations_*.csv, must reproduce the published values")
    print("=" * 78)
    wil_path = os.path.join(RF_DIR, "paired_wilcoxon_bh.csv")
    wil = pd.read_csv(wil_path) if os.path.exists(wil_path) else None
    rf_stats, failures = {}, []

    for arm in arms:
        print(f"  [{arm}]  {ARM_PAIRS[arm][0]} - {ARM_PAIRS[arm][1]}")
        for w in WINDOWS:
            for m in METRICS:
                r = describe(paired_diff(rf, arm, w, m))
                rf_stats[(arm, w, m)] = r
                notes = []

                if arm == "plain":
                    med_e, q1_e, q3_e, pos_e = RF_EXPECTED_PLAIN[(w, m)]
                    ok = (abs(r["median"] - med_e) < TOL and abs(r["q25"] - q1_e) < TOL
                          and abs(r["q75"] - q3_e) < TOL and r["n_positive"] == pos_e)
                    notes.append("table OK" if ok else "table MISMATCH")
                    if not ok:
                        failures.append((arm, w, m, "published table"))

                if wil is not None:
                    sel = wil[(wil.group == ARM_WIL_GROUP[arm])
                              & (wil.metric.astype(str).str.upper() == m.upper())
                              & (wil.window == w)]
                    if len(sel):
                        row = sel.iloc[0]
                        ok = (abs(float(row["median"]) - r["median"]) < TOL
                              and abs(float(row["q1"]) - r["q25"]) < TOL
                              and abs(float(row["q3"]) - r["q75"]) < TOL
                              and abs(float(row["share_positive"])
                                      - r["n_positive"] / r["n_pairs"]) < 1e-9)
                        notes.append("wilcoxon_bh OK" if ok else "wilcoxon_bh MISMATCH")
                        if not ok:
                            failures.append((arm, w, m, "paired_wilcoxon_bh.csv"))
                    else:
                        notes.append("no wilcoxon_bh row")

                print(f"      {w:<7} {m.upper():<5} {txt(r)}   {' | '.join(notes)}")

    if failures:
        die(f"the Random Forest recomputation does not reproduce the published values "
            f"({failures}). The reference run or the pairing logic has changed -- stopping "
            f"before any LightGBM numbers are reported.")
    print("  all RF values reproduced.\n")

    # ---------------------------------------------------------------- LightGBM + BH
    rows = []
    for arm in arms:
        for m in METRICS:
            block = [dict(arm=arm, window=w, estimator="LGBM", measure=m.upper(),
                          **describe(paired_diff(lgbm, arm, w, m)))
                     for w in WINDOWS]
            for r, padj in zip(block, _bh([r["p_wilcoxon"] for r in block])):
                r["p_bh"] = float(padj)
                r["sig_bh_0.05"] = bool(padj < 0.05)
                r["bh_group_size"] = len(block)
            rows.extend(block)
            for w in WINDOWS:  # RF rows travel with the table, uncorrected (check only)
                r = dict(arm=arm, window=w, estimator="RF", measure=m.upper(),
                         **rf_stats[(arm, w, m)])
                r["p_bh"] = np.nan
                r["sig_bh_0.05"] = pd.NA
                r["bh_group_size"] = pd.NA
                rows.append(r)

    table = pd.DataFrame(rows, columns=["arm", "window", "estimator", "measure", "n_pairs",
                                        "median", "q25", "q75", "n_positive", "p_wilcoxon",
                                        "p_bh", "sig_bh_0.05", "bh_group_size"])
    out_path = os.path.join(LGBM_DIR, "table_B5_lgbm.csv")
    table.to_csv(out_path, index=False)

    print("=" * 78)
    print("PAIRED LIFT  treatment - matched reference")
    print("=" * 78)
    pd.set_option("display.width", 240)
    print(table.round(4).to_string(index=False))

    # ---------------------------------------------------------------- absolute performance
    print("\n" + "=" * 78)
    print("ABSOLUTE PERFORMANCE -- median [IQR] over the repeats, and median features kept")
    print("=" * 78)
    abs_rows = []
    for w in WINDOWS:
        for est, df in (("LGBM", lgbm), ("RF", rf)):
            for cfg in configs:
                s = df[(df.feature_window == w) & (df.config == cfg)]
                if s.empty:
                    continue
                rec = dict(window=w, estimator=est, config=cfg, n=len(s))
                for m in METRICS:
                    rec[f"{m}_median"] = float(s[m].median())
                    rec[f"{m}_q25"] = float(s[m].quantile(0.25))
                    rec[f"{m}_q75"] = float(s[m].quantile(0.75))
                rec["n_features_median"] = float(s["n_features_boruta"].median())
                abs_rows.append(rec)
                print(f"  {w:<7} {est:<5} {cfg:<20} n={len(s):>3}  "
                      f"AUROC {rec['auroc_median']:.3f} [{rec['auroc_q25']:.3f}, {rec['auroc_q75']:.3f}]   "
                      f"AUPRC {rec['auprc_median']:.3f} [{rec['auprc_q25']:.3f}, {rec['auprc_q75']:.3f}]   "
                      f"features {rec['n_features_median']:.0f}")
    abs_path = os.path.join(LGBM_DIR, "table_B5_absolute.csv")
    pd.DataFrame(abs_rows).to_csv(abs_path, index=False)

    # ---------------------------------------------------------------- LaTeX
    print("\n" + "=" * 78)
    print("LaTeX rows (Table B.5)")
    print("=" * 78)
    width = max(len(ARM_TEX[a]) for a in arms)
    for arm in arms:
        for w in WINDOWS:
            a = table[(table.arm == arm) & (table.window == w) & (table.estimator == "LGBM")
                      & (table.measure == "AUROC")].iloc[0]
            p = table[(table.arm == arm) & (table.window == w) & (table.estimator == "LGBM")
                      & (table.measure == "AUPRC")].iloc[0]
            print(f"{WINDOW_TEX[w]:<7} & {ARM_TEX[arm]:<{width}} & {cell(a)} "
                  f"& {a['n_positive']}/{a['n_pairs']} & {cell(p)} "
                  f"& {p['n_positive']}/{p['n_pairs']} \\\\")

    print(f"\nSaved: {os.path.relpath(out_path, BASE)}")
    print(f"Saved: {os.path.relpath(abs_path, BASE)}")


if __name__ == "__main__":
    main()

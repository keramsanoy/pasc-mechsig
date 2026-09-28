#!/usr/bin/env python
"""Calibration and recalibration of the primary configurations (thesis 3.7, 4.6, B.5).

Refits one configuration (default mechsig_all; set PASC_CALIB_CONFIG=baseline_ext)
at w0-90 on the same 100 divisions as the main run (seeds 42 + 1000*i, identical
balancing, Boruta and grid search via antony_pipeline.run_single_iteration) and
stores the held-out predictions of every repeat. For each repeat it then

  * assesses the raw probabilities (calibration-in-the-large, slope, ICI;
    Van Calster et al. 2019, Austin & Steyerberg 2019),
  * splits the test part in half stratified by outcome, fits a logistic
    recalibration (and, for comparison, isotonic and a prior shift) on one half
    and evaluates on the other (thesis Appendix A.3),
  * summarises the metrics across repeats.

Inputs : cohort_parquets/enhanced_w0_90_strict_all_patients.parquet + families.json
         (written by scripts/run_main_analysis.py)
Outputs: results/calibration/<config>/predictions_<config>.parquet   (patient-level, never commit)
         results/calibration/<config>/calibration_metrics.csv
         results/calibration/<config>/calibration_summary.md
The decision-curve analysis and the calibration figure are drawn from the stored
predictions in notebooks/make_thesis_figures.ipynb.

Run:
  PASC_CALIB_CONFIG=mechsig_all  python scripts/run_calibration.py
  PASC_CALIB_CONFIG=baseline_ext python scripts/run_calibration.py
  PASC_CALIB_N=5 PASC_CALIB_DRY=1 ... for a smoke test
"""
import os
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401  (makes `pasc` importable from a plain clone)
from pasc.config.paths import COHORT_DIR as _COHORT_DIR, CALIBRATION_RESULTS_DIR

print("=" * 80)
print("Probability calibration of the primary configurations (w0-90)")
print("=" * 80)

from pasc.features.enhanced import build_enhanced_model_configs
from pasc.modeling.pipeline import run_single_iteration

import pasc.modeling.calibration as cal

# ===========================================================================
# Configuration
# ===========================================================================
WINDOW = "w0_90"
COHORT_MODE = "strict"
MODEL_TYPE = "RF"
HEADLINE_CONFIG = os.environ.get("PASC_CALIB_CONFIG", "mechsig_all")

RANDOM_STATE_BASE = 42
N_ITERATIONS = int(os.environ.get("PASC_CALIB_N", "100"))
BORUTA_MAX_ITER = 50
BORUTA_N_ESTIMATORS = 500   # matches run_main_analysis.py
BORUTA_PERC = 97            # matches run_main_analysis.py
LOWESS_FRAC = 0.66
# DCA threshold grid: true prevalence ~1% -> clinically relevant low-threshold range.
DCA_THRESHOLDS = np.arange(0.0, 0.10 + 1e-9, 0.005)

# The main run withholds the index-date calendar feature from modelling
# (thesis 3.6; PASC_DROP_INDEX_CALENDAR in scripts/run_main_analysis.py). Mirror it
# here so the calibrated model is the model whose discrimination is reported.
DROP_INDEX_CALENDAR = os.environ.get("PASC_DROP_INDEX_CALENDAR", "1") == "1"
CALENDAR_DROP_COLS = ["f_ext_index_year_month"]

DRY = os.environ.get("PASC_CALIB_DRY", "0") == "1"
if DRY:
    print("\n[dry-run] PASC_CALIB_DRY=1 -> single seed (iteration 1)")
    N_ITERATIONS = 1

COHORT_DIR = str(_COHORT_DIR)
OUT_DIR = os.path.join(str(CALIBRATION_RESULTS_DIR), HEADLINE_CONFIG)
os.makedirs(OUT_DIR, exist_ok=True)

print("\nConfiguration:")
print(f"  WINDOW={WINDOW}  COHORT_MODE={COHORT_MODE}  MODEL_TYPE={MODEL_TYPE}")
print(f"  HEADLINE_CONFIG={HEADLINE_CONFIG}  N_ITERATIONS={N_ITERATIONS}")
print(f"  DROP_INDEX_CALENDAR={DROP_INDEX_CALENDAR}")
print(f"  OUT_DIR={OUT_DIR}")


# ===========================================================================
# Modelled population: the cached w0-90 matrix written by scripts/run_main_analysis.py
# (already gated: index <= cutoff, no PASC code < 90 d, controls >= 365 d follow-up).
# ===========================================================================
def build_strict_frame():
    feat_path = os.path.join(COHORT_DIR, f"enhanced_{WINDOW}_{COHORT_MODE}_all_patients.parquet")
    fam_path = os.path.join(COHORT_DIR, f"enhanced_{WINDOW}_{COHORT_MODE}_families.json")
    for p in (feat_path, fam_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required artefact missing: {p}")
    feature_df = pd.read_parquet(feat_path)
    with open(fam_path) as fh:
        feature_families = json.load(fh)
    assert feature_df["label"].isin([0, 1]).all()
    if DROP_INDEX_CALENDAR:
        dropped = [c for c in CALENDAR_DROP_COLS if c in feature_df.columns]
        feature_df = feature_df.drop(columns=dropped)
        for fam, cols in feature_families.items():
            feature_families[fam] = [c for c in cols if c not in CALENDAR_DROP_COLS]
        print(f"[frame] withheld from modelling: {dropped}")
    # The MSHS run modelled 168,345 patients with 2,102 cases (thesis 4.1);
    # another extract or site will differ, so this is a readout, not a check.
    print(f"[frame] strict enhanced store: n={len(feature_df):,} "
          f"pos={int(feature_df['label'].sum()):,} "
          f"prev={100 * feature_df['label'].mean():.3f}%")
    return feature_df, feature_families


print("\n[1] Loading strict frame + families...")
frame, feature_families = build_strict_frame()
PI = float(frame["label"].mean())   # true outcome prevalence
print(f"[frame] true prevalence pi = {PI:.5f}  (logit={np.log(PI / (1 - PI)):.4f})")

# ===========================================================================
# 2. Reconstruct configs exactly as the main run does + select the headline one
# ===========================================================================
print("\n[2] Reconstructing model configs...")
all_configs = build_enhanced_model_configs(feature_families)

if HEADLINE_CONFIG not in all_configs:
    raise KeyError(f"Headline config {HEADLINE_CONFIG!r} not built. "
                   f"Available (sample): {list(all_configs)[:8]} ...")
cfg_features = [c for c in all_configs[HEADLINE_CONFIG] if c in frame.columns]
n_missing = len(all_configs[HEADLINE_CONFIG]) - len(cfg_features)
print(f"[config] {HEADLINE_CONFIG}: {len(all_configs[HEADLINE_CONFIG])} features "
      f"({len(cfg_features)} present in frame, {n_missing} missing)")

# Engagement controls are forced past the prevalence filter and Boruta, as in the main run.
engagement_cols = list(feature_families.get("engagement_controls", []) or []) or None
print(f"[config] forced features (engagement controls): {engagement_cols}")

X = frame[cfg_features].to_numpy(dtype=np.float32)
y = frame["label"].to_numpy(dtype=int)

# ===========================================================================
# 3. Collect held-out predictions per seed (locked pipeline, RF, no SHAP)
# ===========================================================================
print("\n[3] Refitting headline config per seed and collecting held-out predictions...")
pred_frames = []
for it in range(1, N_ITERATIONS + 1):
    ir = run_single_iteration(
        X, y, cfg_features,
        iteration=it,
        model_type=MODEL_TYPE,
        cohort_name=f"{HEADLINE_CONFIG}__calibration",
        random_state_base=RANDOM_STATE_BASE,
        boruta_max_iter=BORUTA_MAX_ITER,
        boruta_n_estimators=BORUTA_N_ESTIMATORS,
        compute_shap=False,
        forced_features=engagement_cols,
        boruta_perc=BORUTA_PERC,
    )
    pred_frames.append(pd.DataFrame({
        "seed": ir.seed,
        "y_true": np.asarray(ir.y_test, dtype=int),
        "p_raw": np.asarray(ir.y_pred_proba, dtype=float),
    }))

preds = pd.concat(pred_frames, ignore_index=True)
pred_path = os.path.join(OUT_DIR, f"predictions_{HEADLINE_CONFIG}.parquet")
preds.to_parquet(pred_path, index=False)
print(f"[3] Saved pooled predictions: {pred_path}  "
      f"({len(preds):,} rows, {preds['seed'].nunique()} seeds)")
print(f"[3] pooled raw: mean(p)={preds['p_raw'].mean():.4f} "
      f"obs_rate={preds['y_true'].mean():.4f}  "
      f"(overforecast factor ~{preds['p_raw'].mean() / max(preds['y_true'].mean(), 1e-9):.1f}x)")

# ===========================================================================
# 4. Per-seed assessment + recalibration (disjoint cal/eval split)
# ===========================================================================
print("\n[4] Per-seed assessment + recalibration...")
per_seed_rows = []   # tidy long: row_type=per_seed
for seed, g in preds.groupby("seed"):
    yt = g["y_true"].to_numpy(dtype=int)
    pr = g["p_raw"].to_numpy(dtype=float)

    # Headline raw miscalibration on the FULL test set (method='raw_full').
    raw_full = cal.assess(yt, pr, frac=LOWESS_FRAC)
    raw_full.update({"seed": int(seed), "method": "raw_full",
                     "n_eval": int(len(yt)), "pos_eval": int(yt.sum())})

    # Fair pre/post on the disjoint eval split: raw, logistic, isotonic, prior_shift.
    eval_rows = cal.recalibrate_and_assess_seed(
        yt, pr, seed=int(seed), pi=PI, frac=LOWESS_FRAC, cal_size=0.5
    )

    for r in [raw_full, *eval_rows]:
        for metric in cal._METRIC_KEYS:
            if metric in r:
                per_seed_rows.append({
                    "row_type": "per_seed", "method": r["method"], "metric": metric,
                    "seed": int(seed), "value": r[metric],
                    "mean": np.nan, "sd": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan,
                    "n_seeds": np.nan,
                })
    print(f"  seed={int(seed)}: raw_full slope={raw_full['slope']:.3f} "
          f"citl={raw_full['citl']:+.3f} ici={raw_full['ici']:.4f}")

per_seed_df = pd.DataFrame(per_seed_rows)

# Summary rows (mean + 95% CI across seeds) per (method, metric).
# Reuse summarize_across_seeds by reshaping per-seed long -> wide per row.
wide_for_summary = (
    per_seed_df.pivot_table(index=["method", "seed"], columns="metric",
                            values="value", aggfunc="first")
    .reset_index()
)
summary_df = cal.summarize_across_seeds(wide_for_summary.to_dict("records"))
summary_df.insert(0, "row_type", "summary")
summary_df["seed"] = np.nan
summary_df["value"] = np.nan
summary_df = summary_df[["row_type", "method", "metric", "seed", "value",
                         "mean", "sd", "ci95_lo", "ci95_hi", "n_seeds"]]

metrics_out = pd.concat([per_seed_df, summary_df], ignore_index=True)
metrics_path = os.path.join(OUT_DIR, "calibration_metrics.csv")
metrics_out.to_csv(metrics_path, index=False)
print(f"[4] Saved metrics: {metrics_path} "
      f"({len(per_seed_df)} per-seed rows + {len(summary_df)} summary rows)")

# ===========================================================================
# 5. Console readout of the key numbers
# ===========================================================================
def _sm(method, metric):
    r = summary_df[(summary_df.method == method) & (summary_df.metric == metric)]
    if r.empty:
        return float("nan"), float("nan"), float("nan")
    return float(r["mean"].iloc[0]), float(r["ci95_lo"].iloc[0]), float(r["ci95_hi"].iloc[0])

print("\n" + "=" * 80)
print("CALIBRATION SUMMARY (mean [95% CI] across seeds)")
print("=" * 80)
hdr = f"{'method':<14} {'slope':>22} {'CITL':>22} {'ICI':>20}"
print(hdr)
for method in ["raw_full", "raw", "logistic", "isotonic", "prior_shift"]:
    sl, sl_lo, sl_hi = _sm(method, "slope")
    ci, ci_lo, ci_hi = _sm(method, "citl")
    ic, ic_lo, ic_hi = _sm(method, "ici")
    print(f"{method:<14} {sl:>7.3f} [{sl_lo:+.3f},{sl_hi:+.3f}] "
          f"{ci:>7.3f} [{ci_lo:+.3f},{ci_hi:+.3f}] "
          f"{ic:>7.4f} [{ic_lo:.4f},{ic_hi:.4f}]")

# ===========================================================================
# 6. Calibration summary markdown
# ===========================================================================
raw_sl = _sm("raw_full", "slope")
raw_citl = _sm("raw_full", "citl")
raw_ici = _sm("raw_full", "ici")
raw_mpo = _sm("raw_full", "mean_pred_minus_obs")
log_sl = _sm("logistic", "slope")
log_citl = _sm("logistic", "citl")
log_ici = _sm("logistic", "ici")

md = f"""# Calibration readout — `{HEADLINE_CONFIG}` (strict cohort, w0-90, 100 seeded repeats)

**Model:** RF with 1:1 undersampling, pasc.modeling.pipeline, {preds['seed'].nunique()} seeds
(`seed = 42 + i*1000`). True outcome prevalence **pi = {PI:.4f}** ({100 * PI:.2f}%).
Pooled mean predicted probability = **{preds['p_raw'].mean():.4f}** vs observed rate
**{preds['y_true'].mean():.4f}** (overforecast factor ~{preds['p_raw'].mean() / max(preds['y_true'].mean(), 1e-9):.0f}x).

## Raw model (expected miscalibration)
The undersampled RF is miscalibrated by construction: its probabilities target the 50%
training prevalence, not the true ~{100 * PI:.1f}%. The dominant defect is therefore a prior
(intercept) shift, not a slope problem: on the full held-out test sets the raw model shows
**CITL = {raw_citl[0]:+.3f}** [{raw_citl[1]:+.3f}, {raw_citl[2]:+.3f}] (ideal = 0; large
*negative* value = systematic overforecasting, because the offset log-odds are far too high),
mean(p)-mean(y) = {raw_mpo[0]:+.4f}, **calibration slope = {raw_sl[0]:.3f}**
[{raw_sl[1]:+.3f}, {raw_sl[2]:+.3f}] (ideal = 1; with 1:1 undersampling the discrimination
ordering is roughly preserved so the slope stays near 1 rather than collapsing), and
**ICI = {raw_ici[0]:.4f}** [{raw_ici[1]:.4f}, {raw_ici[2]:.4f}]. The flexible curve sits far
below the diagonal (observed event rate ≪ predicted probability across the whole range).

## Recalibration (fit on a disjoint 50% split, evaluated on the held-out 50%)
**Logistic (Platt) recalibration** restores calibration: **slope = {log_sl[0]:.3f}**
[{log_sl[1]:+.3f}, {log_sl[2]:+.3f}] -> ~1, **CITL = {log_citl[0]:+.3f}**
[{log_citl[1]:+.3f}, {log_citl[2]:+.3f}] -> ~0, and **ICI = {log_ici[0]:.4f}**
[{log_ici[1]:.4f}, {log_ici[2]:.4f}] (down from {raw_ici[0]:.4f}). Isotonic and the analytic
prior-shift map are reported alongside in `calibration_metrics.csv` for comparison.

## Headline
- Raw: slope **{raw_sl[0]:.3f}**, CITL **{raw_citl[0]:+.3f}**, ICI **{raw_ici[0]:.4f}** (miscalibrated, as expected).
- Post logistic recalibration: slope **{log_sl[0]:.3f}**, CITL **{log_citl[0]:+.3f}**, ICI **{log_ici[0]:.4f}** (the fix).

The decision-curve analysis over thresholds 0-10% and the calibration curves are drawn
from `predictions_{HEADLINE_CONFIG}.parquet` in notebooks/make_thesis_figures.ipynb.
"""
md_path = os.path.join(OUT_DIR, "calibration_summary.md")
with open(md_path, "w") as fh:
    fh.write(md)
print(f"\n[6] Saved summary: {md_path}")

print("\n" + "=" * 80)
print("CALIBRATION RUN COMPLETE")
print("=" * 80)

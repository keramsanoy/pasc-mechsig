#!/usr/bin/env python
"""
run_lgbm_final.py -- LightGBM robustness check of the mechanism lift (thesis 3.6.1, 4.3.3, Table B.5).

Standalone. Imports the pipeline primitives from antony_pipeline.py and the config builder
from enhanced_mech_signals.py and never redefines them. Reads the Random Forest outputs of
run_enhanced_mechsig.py (results/main/perc97/) and writes only under results/lgbm_robustness/.

WHAT IT ANSWERS
---------------
For each mechanism arm at w0_90 and w60_90: the paired LightGBM difference on the SAME 100
divisions the Random Forest used. Everything except the final estimator is identical to
antony_pipeline.run_single_iteration, so the delta isolates the estimator.

  plain    mechsig_all        - baseline_ext
  no_td    mechsig_all_no_td  - baseline_ext_no_td       (composites removed)
  no_eng   mechsig_all_no_eng - baseline_ext_no_eng      (engagement stripped)

Each arm is its own BH correction group, exactly as scripts/paired_wilcoxon_bh.py treats them.

PARITY
------
  data        cohort_parquets/enhanced_<window>_strict_all_patients.parquet (+ _families.json),
              loaded as run_enhanced_mechsig.py does with PASC_REEXTRACT=0
  calendar    the PASC_DROP_INDEX_CALENDAR=1 drop list is removed from frame / families / configs
  configs     enhanced_mech_signals.build_enhanced_model_configs(), builder order
  split       train_test_split(test_size=0.2, random_state=42+i*1000, stratify=y)
  balance     antony_pipeline.balance_training_set(..., random_state=seed)
  selection   PINNED to the RF's features_<config>_RF.json for that iteration (default), or
              refit in-fold with PASC_LGBM_REFIT_BORUTA=1
  impute      SimpleImputer(strategy="median") fitted on the balanced training rows
  tuning      GridSearchCV(scoring="accuracy", cv=StratifiedKFold(5, shuffle, seed), refit=True)
  metrics     roc_auc_score / average_precision_score on the untouched imbalanced test part

The split depends only on y and the seed, so it is identical to the RF's split for that iteration
and identical across the two configs -- which is what makes the delta paired.

The grid mirrors antony_pipeline.RF_PARAM_GRID setting for setting (36 combinations):
  n_estimators [100, 300, 500] | max_depth [5, 10, 20, -1] | min_child_samples [1, 5, 10]
max_depth=-1 is LightGBM's unlimited, the analogue of the forest's max_depth=None; every other
LightGBM setting stays at its default (learning_rate 0.1, num_leaves 31). No scale_pos_weight:
balancing already handles the imbalance, same as the forest.

PARALLELISM
-----------
Iterations run in parallel across processes (joblib/loky); everything inside a worker is
single-threaded, so there is no nested oversubscription. n_jobs never changes results.
Each finished iteration is appended and flushed immediately, so Ctrl-C keeps all completed work
and a rerun resumes (PASC_SKIP_EXISTING, on by default).

Interactive use on a compute node. BOTH resource flags matter:
  * `span[hosts=1]` -- without it LSF scatters the slots over several machines and the shell
    gets a single core, so the workers all contend for one CPU.
  * enough memory -- every loky worker is a full Python with numpy/pandas/lightgbm resident
    (~250 MB each), so 15 workers need ~4 GB of interpreters ALONE, on top of the parent.
    A 4 GB job limit is killed with TERM_MEMLIMIT before the first iteration finishes.

    tmux new -s lgbm                      # tmux on the LOGIN node, never inside the job:
                                          # an LSF job kills the tmux server when it ends
    bsub -P <allocation> -q <queue> -n 16 -W 12:00 \\
         -R "span[hosts=1]" -R "rusage[mem=8000]" -Is /bin/bash
    conda activate <env> && python run_lgbm_final.py

ENVIRONMENT
-----------
  PASC_LGBM_WINDOWS       default "w0_90,w60_90"
  PASC_LGBM_ARMS          default "plain,no_td,no_eng"
  PASC_LGBM_WORKERS       default min(15, LSB_DJOB_NUMPROC-1); one core left for the shell
  PASC_LGBM_FAST=1        3 iterations and a one-cell grid (smoke test)
  PASC_LGBM_MAX_ITERS=n   first n iterations, FULL grid (timing probe)
  PASC_LGBM_REFIT_BORUTA=1  refit prevalence_filter + Boruta in-fold instead of pinning
  PASC_SKIP_EXISTING      default 1
  PASC_LGBM_SUMMARIZE     run scripts/summarize_lgbm_table.py at the end (default: only on a
                          complete, full-grid run of both windows)
"""

import os

# Cap per-process thread pools BEFORE importing numpy / lightgbm. On a shared node the per-user
# process limit is low; LightGBM's OpenMP pool multiplied across loky workers exhausts it and
# pthread_create returns EAGAIN. One thread per worker keeps the total ~= WORKERS.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

# antony_pipeline reads this at import time and hands it to boruta_select's internal forest.
# We parallelise OUTSIDE, so every worker must stay single-threaded.
os.environ.setdefault("ANTONY_N_JOBS", "1")

import csv
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pyarrow_pq
import joblib
import sklearn
import lightgbm
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from pasc_paths import REPO_ROOT, COHORT_DIR as _COHORT_DIR, MAIN_RESULTS_DIR, LGBM_RESULTS_DIR  # noqa: E402

BASE = str(REPO_ROOT)
if BASE not in sys.path:
    sys.path.insert(0, BASE)

# Locked primitives: import, never redefine.
from antony_pipeline import balance_training_set, boruta_select, prevalence_filter  # noqa: E402
from enhanced_mech_signals import build_enhanced_model_configs  # noqa: E402


# ----------------------------------------------------------------------------- configuration
COHORT_MODE = "strict"

# Each arm is a (treatment, reference) pair. Mechanism lift is ALWAYS measured against the
# matched reference -- the _no_td arm against baseline_ext_no_td, the _no_eng arm against
# baseline_ext_no_eng -- never against baseline_ext or baseline_antony.
ARM_PAIRS = {
    "plain":  ("mechsig_all", "baseline_ext"),
    "no_td":  ("mechsig_all_no_td", "baseline_ext_no_td"),
    "no_eng": ("mechsig_all_no_eng", "baseline_ext_no_eng"),
}
N_ITERATIONS = 100
RANDOM_STATE_BASE = 42
BORUTA_PERC = 97
BORUTA_MAX_ITER = 50
BORUTA_N_ESTIMATORS = 500

COHORT_DIR = str(_COHORT_DIR)
RF_RESULTS_DIR = os.path.join(str(MAIN_RESULTS_DIR), f"perc{BORUTA_PERC}")
OUT_DIR = str(LGBM_RESULTS_DIR)

# Optional guard: the MSHS run modelled 168,345 patients / 2,102 cases (thesis 4.1).
# Set PASC_LGBM_EXPECT="168345,2102" to refuse a cache that does not match.
_expect = os.environ.get("PASC_LGBM_EXPECT", "").strip()
EXPECTED_N_ROWS, EXPECTED_N_POS = (tuple(int(x) for x in _expect.split(",")) if _expect else (None, None))

# run_enhanced_mechsig.py, PASC_DROP_INDEX_CALENDAR=1.
CALENDAR_DROP_COLS = ["f_ext_index_year_month"]

LGBM_PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 10, 20, -1],
    "min_child_samples": [1, 5, 10],
}
LGBM_FAST_GRID = {"n_estimators": [100], "max_depth": [5], "min_child_samples": [10]}

WINDOWS = [w.strip() for w in
           os.environ.get("PASC_LGBM_WINDOWS", "w0_90,w60_90").split(",") if w.strip()]
ARMS = [a.strip() for a in
        os.environ.get("PASC_LGBM_ARMS", "plain,no_td,no_eng").split(",") if a.strip()]
CONFIGS = []
for _arm in ARMS:
    for _cfg in reversed(ARM_PAIRS.get(_arm, ())):  # reference first, so it runs first
        if _cfg not in CONFIGS:
            CONFIGS.append(_cfg)
FAST = os.environ.get("PASC_LGBM_FAST", "0") == "1"
REFIT_BORUTA = os.environ.get("PASC_LGBM_REFIT_BORUTA", "0") == "1"
SKIP_EXISTING = os.environ.get("PASC_SKIP_EXISTING", "1") == "1"
MAX_ITERS = int(os.environ.get("PASC_LGBM_MAX_ITERS", "0") or 0)

SELECTION_SOURCE = "refit_boruta" if REFIT_BORUTA else "pinned_rf"
PARAM_GRID = LGBM_FAST_GRID if FAST else LGBM_PARAM_GRID
N_ITER_RUN = 3 if FAST else (MAX_ITERS if MAX_ITERS else N_ITERATIONS)

# Smoke/probe runs must never land in the real output tree: with PASC_SKIP_EXISTING on, their
# one-cell-grid or partial rows would be kept by the subsequent full run.
if FAST or MAX_ITERS:
    OUT_DIR = os.path.join(OUT_DIR, "_smoke")

CSV_COLUMNS = ["cohort", "model", "iteration", "seed", "auroc", "auprc", "n_features_initial",
               "n_features_filtered", "n_features_boruta", "best_params", "feature_window",
               "mode", "config", "boruta_perc", "selection_source"]


def _default_workers():
    n = os.environ.get("LSB_DJOB_NUMPROC", "")
    n = int(n) if n.isdigit() else (os.cpu_count() or 2)
    return max(1, min(15, n - 1))


WORKERS = int(os.environ.get("PASC_LGBM_WORKERS", str(_default_workers())))

_COMPLETE_RUN = (not FAST and not MAX_ITERS and sorted(WINDOWS) == ["w0_90", "w60_90"]
                 and sorted(ARMS) == ["no_eng", "no_td", "plain"])
SUMMARIZE = os.environ.get("PASC_LGBM_SUMMARIZE", "1" if _COMPLETE_RUN else "0") == "1"


def die(msg):
    print(f"\nABORT: {msg}\n", flush=True)
    raise SystemExit(1)


# ----------------------------------------------------------------------------- data loading
def load_window(window):
    """Load the cached cohort as run_enhanced_mechsig.py does with PASC_REEXTRACT=0.

    Only the columns the two configs actually need are materialised: every loky worker
    inherits this frame's memory, and LSF caps the whole job, so the parent must stay lean.
    """
    pq_path = os.path.join(COHORT_DIR, f"enhanced_{window}_{COHORT_MODE}_all_patients.parquet")
    fam_path = os.path.join(COHORT_DIR, f"enhanced_{window}_{COHORT_MODE}_families.json")
    for p in (pq_path, fam_path):
        if not os.path.exists(p):
            die(f"missing cohort cache: {p}")

    schema_names = list(pyarrow_pq.ParquetFile(pq_path).schema_arrow.names)
    with open(fam_path) as fh:
        families = json.load(fh)

    configs = build_enhanced_model_configs(families)

    dropped = []
    for col in CALENDAR_DROP_COLS:
        if col in schema_names:
            schema_names.remove(col)
            dropped.append(col)
        for fam, cols in families.items():
            if col in cols:
                families[fam] = [c for c in cols if c != col]
        for cfg, cols in configs.items():
            if col in cols:
                configs[cfg] = [c for c in cols if c != col]

    available = set(schema_names)
    needed, seen = ["label"], {"label"}
    for cfg in CONFIGS:
        for c in configs[cfg]:
            if c in available and c not in seen:
                seen.add(c)
                needed.append(c)
    df = pd.read_parquet(pq_path, columns=needed)

    n, n_pos = len(df), int(df["label"].sum())
    if EXPECTED_N_ROWS is not None and (n != EXPECTED_N_ROWS or n_pos != EXPECTED_N_POS):
        die(f"{pq_path}\n       has {n} rows / {n_pos} positives, expected {EXPECTED_N_ROWS} / "
            f"{EXPECTED_N_POS} (PASC_LGBM_EXPECT). Refusing to run on a different cache.")

    return dict(window=window, df=df, families=families, configs=configs, parquet=pq_path,
                families_path=fam_path, dropped=dropped,
                n_rows=n, n_pos=n_pos, n_parquet_cols=len(schema_names) + len(dropped))


def rf_dir(window):
    return os.path.join(RF_RESULTS_DIR, window, COHORT_MODE)


def load_rf_iterations(window, config):
    path = os.path.join(rf_dir(window), f"iterations_{config}.csv")
    if not os.path.exists(path):
        die(f"missing RF reference: {path}")
    d = pd.read_csv(path)
    return {int(r.iteration): r for r in d.itertuples()}


def load_rf_selections(window, config):
    path = os.path.join(rf_dir(window), f"features_{config}_RF.json")
    if not os.path.exists(path):
        die(f"missing RF selections: {path}")
    with open(path) as fh:
        records = json.load(fh)
    return {int(r["iteration"]): list(r["selected_features"]) for r in records}


# ----------------------------------------------------------------------------- startup report
def print_window_report(wd):
    w = wd["window"]
    print(f"\n{'=' * 78}\nWINDOW {w} ({COHORT_MODE})\n{'=' * 78}")
    print(f"  parquet          : {os.path.relpath(wd['parquet'], BASE)}")
    print(f"  parquet columns  : {wd['n_parquet_cols']}")
    print(f"  loaded           : {wd['df'].shape[0]} rows x {wd['df'].shape[1]} cols "
          f"(label + the union the two configs need)")
    print(f"  label            : {wd['n_pos']} positives "
          f"({100.0 * wd['n_pos'] / wd['n_rows']:.2f}%)")
    print(f"  families         : {len(wd['families'])}")
    for fam, cols in wd["families"].items():
        print(f"      {fam:<26} {len(cols):>4}")
    eng = wd["families"].get("engagement_controls", [])
    print(f"  engagement ctrls : {eng}")
    print(f"  calendar dropped : {wd['dropped'] or 'none present in this parquet'}")
    for cfg in CONFIGS:
        cols = [c for c in wd["configs"][cfg] if c in wd["df"].columns]
        print(f"  config {cfg:<14} {len(cols)} columns available "
              f"({len(wd['configs'][cfg])} in the builder list)")


# ----------------------------------------------------------------------------- one iteration
def fit_iteration(X, y, iteration, param_grid, feature_names, sel_idx=None, forced_features=None,
                  boruta_perc=BORUTA_PERC, boruta_max_iter=BORUTA_MAX_ITER,
                  boruta_n_estimators=BORUTA_N_ESTIMATORS, random_state_base=RANDOM_STATE_BASE):
    """One repeat. `sel_idx` pins the RF's selection; None refits prevalence filter + Boruta."""
    seed = random_state_base + iteration * 1000

    if sel_idx is None:
        Xw, names = X, list(feature_names)
    else:
        Xw = np.ascontiguousarray(X[:, sel_idx])  # column subset first; the split only uses y+seed
        names = [feature_names[i] for i in sel_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        Xw, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train_bal, y_train_bal = balance_training_set(X_train, y_train, random_state=seed)

    n_initial = X_train.shape[1]
    if sel_idx is None:
        kept_idx, kept_names, _ = prevalence_filter(X_train_bal, names,
                                                    forced_features=forced_features)
        X_train_bal, X_test = X_train_bal[:, kept_idx], X_test[:, kept_idx]
        n_filtered = len(kept_idx)
        pick_idx, sel_names = boruta_select(
            X_train_bal, y_train_bal, kept_names, random_state=seed,
            max_iter=boruta_max_iter, n_estimators=boruta_n_estimators,
            forced_features=forced_features, boruta_perc=boruta_perc,
        )
        X_train_bal, X_test = X_train_bal[:, pick_idx], X_test[:, pick_idx]
    else:
        n_filtered, sel_names = n_initial, names

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_bal)
    X_test_imp = imputer.transform(X_test)

    # n_jobs=1 only; passing num_threads too would emit LightGBM's alias warning on every fit.
    search = GridSearchCV(
        estimator=LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1),
        param_grid=param_grid,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train_imp, y_train_bal)
    proba = search.best_estimator_.predict_proba(X_test_imp)[:, 1]

    return dict(
        iteration=iteration,
        seed=seed,
        auroc=float(roc_auc_score(y_test, proba)),
        auprc=float(average_precision_score(y_test, proba)),
        n_features_initial=int(n_initial),
        n_features_filtered=int(n_filtered),
        n_features_boruta=int(len(sel_names)),
        best_params=dict(search.best_params_),
        selected_features=list(sel_names),
    )


# ----------------------------------------------------------------------------- driver
def _make_parallel(n_jobs):
    for mode in ("generator_unordered", "generator"):
        try:
            return Parallel(n_jobs=n_jobs, backend="loky", return_as=mode)
        except (TypeError, ValueError):
            continue
    return Parallel(n_jobs=n_jobs, backend="loky")


def _done_iterations(path):
    if not (SKIP_EXISTING and os.path.exists(path) and os.path.getsize(path) > 0):
        return set()
    try:
        d = pd.read_csv(path)
    except Exception:
        return set()
    return set(d["iteration"].astype(int)) if "iteration" in d.columns else set()


def _fmt_eta(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def run_config(wd, config):
    """Run every outstanding iteration of one (window, config) and stream rows to disk."""
    window = wd["window"]
    df, configs = wd["df"], wd["configs"]

    cfg_cols = [c for c in configs[config] if c in df.columns]
    rf_rows = load_rf_iterations(window, config)
    rf_sel = load_rf_selections(window, config)

    rf_initial = int(rf_rows[1].n_features_initial)
    if len(cfg_cols) != rf_initial:
        die(f"{window}/{config}: config has {len(cfg_cols)} available columns but the RF row "
            f"records n_features_initial={rf_initial}. The parquet or the config builder has "
            f"drifted; the comparison would no longer be estimator-only.")

    out_dir = os.path.join(OUT_DIR, window, COHORT_MODE)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"iterations_{config}_LGBM.csv")

    iterations = list(range(1, N_ITER_RUN + 1))
    done = _done_iterations(out_path)
    todo = [i for i in iterations if i not in done]

    print(f"\n[{window}/{config}] {len(cfg_cols)} config columns | "
          f"{len(todo)} of {len(iterations)} iterations to run "
          f"({len(done)} already on disk) | selection={SELECTION_SOURCE} | "
          f"{len(PARAM_GRID['n_estimators']) * len(PARAM_GRID['max_depth']) * len(PARAM_GRID['min_child_samples'])}"
          f" grid cells | {WORKERS} workers", flush=True)
    if not todo:
        return out_path

    # The _no_eng arm deliberately has no engagement controls, so nothing is force-kept there.
    forced = (None if config.endswith("_no_eng")
              else (wd["families"].get("engagement_controls", []) or None))

    if REFIT_BORUTA:
        feature_names = cfg_cols
        X = df[cfg_cols].to_numpy(dtype=np.float32)
        col_index = None
    else:
        # Union of the pinned columns: one shared matrix, per-iteration positional indices.
        needed, order = set(), []
        for i in todo:
            names = rf_sel[i]
            missing = [n for n in names if n not in df.columns]
            if missing:
                die(f"{window}/{config} iteration {i}: pinned features absent from the "
                    f"calendar-dropped frame: {missing[:5]}"
                    f"{' ...' if len(missing) > 5 else ''}")
            rf_seed = int(rf_rows[i].seed)
            if rf_seed != RANDOM_STATE_BASE + i * 1000:
                die(f"{window}/{config} iteration {i}: RF seed {rf_seed} != "
                    f"{RANDOM_STATE_BASE + i * 1000}; the splits would not be paired.")
            for n in names:
                if n not in needed:
                    needed.add(n)
                    order.append(n)
        feature_names = [c for c in cfg_cols if c in needed]
        X = df[feature_names].to_numpy(dtype=np.float32)
        pos = {c: k for k, c in enumerate(feature_names)}
        col_index = {i: [pos[n] for n in rf_sel[i]] for i in todo}

    y = df["label"].to_numpy(dtype=int)

    # Appending without skip-existing would duplicate iterations, so start the file over instead.
    fh = open(out_path, "a" if SKIP_EXISTING else "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
    if fh.tell() == 0:
        writer.writeheader()
        fh.flush()

    selections = {}
    t0, n_done = time.time(), 0
    try:
        tasks = (
            delayed(fit_iteration)(
                X, y, i, PARAM_GRID, feature_names,
                sel_idx=(None if col_index is None else col_index[i]),
                forced_features=forced,
            )
            for i in todo
        )
        for res in _make_parallel(WORKERS)(tasks):
            i = res["iteration"]
            selections[i] = res["selected_features"]
            row = dict(
                cohort=f"{window}/{COHORT_MODE}/{config} (perc={BORUTA_PERC}, lgbm)",
                model="LGBM",
                iteration=i,
                seed=res["seed"],
                auroc=res["auroc"],
                auprc=res["auprc"],
                n_features_initial=(res["n_features_initial"] if REFIT_BORUTA
                                    else int(rf_rows[i].n_features_initial)),
                n_features_filtered=(res["n_features_filtered"] if REFIT_BORUTA
                                     else int(rf_rows[i].n_features_filtered)),
                n_features_boruta=res["n_features_boruta"],
                best_params=str(res["best_params"]),
                feature_window=window,
                mode=COHORT_MODE,
                config=config,
                boruta_perc=BORUTA_PERC,
                selection_source=SELECTION_SOURCE,
            )
            writer.writerow(row)
            fh.flush()
            os.fsync(fh.fileno())

            n_done += 1
            elapsed = time.time() - t0
            eta = elapsed / n_done * (len(todo) - n_done)
            msg = (f"  [{window}/{config}] iter {i:>3} | auroc={res['auroc']:.4f} "
                   f"auprc={res['auprc']:.4f} nfeat={res['n_features_boruta']:>3} | "
                   f"{n_done}/{len(todo)} done, {_fmt_eta(elapsed)} elapsed, eta {_fmt_eta(eta)}")
            if REFIT_BORUTA:
                same = set(res["selected_features"]) == set(rf_sel[i])
                msg += f" | boruta {'==' if same else '!='} RF"
            print(msg, flush=True)
    except KeyboardInterrupt:
        print(f"\n  interrupted -- {n_done} iterations written to "
              f"{os.path.relpath(out_path, BASE)}; rerun to resume", flush=True)
        raise
    finally:
        fh.close()

    if REFIT_BORUTA and selections:
        sel_path = os.path.join(out_dir, f"features_{config}_LGBM.json")
        records = [{"iteration": i, "selected_features": selections[i]}
                   for i in sorted(selections)]
        with open(sel_path, "w") as f:
            json.dump(records, f, indent=2)
        n_same = sum(1 for i in selections if set(selections[i]) == set(rf_sel[i]))
        print(f"  [{window}/{config}] Boruta matched the RF selection as a set in "
              f"{n_same}/{len(selections)} iterations -> "
              f"{os.path.relpath(sel_path, BASE)}", flush=True)

    return out_path


def write_manifest(window_detail, paths):
    manifest = dict(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        script=os.path.basename(__file__),
        purpose="LightGBM robustness rerun on the final cohort (thesis Table B.5)",
        windows=WINDOWS,
        arms={a: dict(treatment=ARM_PAIRS[a][0], reference=ARM_PAIRS[a][1]) for a in ARMS},
        configs=CONFIGS,
        cohort_mode=COHORT_MODE,
        n_iterations=N_ITER_RUN,
        random_state_base=RANDOM_STATE_BASE,
        seed_formula="42 + iteration * 1000",
        selection_source=SELECTION_SOURCE,
        boruta_perc=BORUTA_PERC,
        boruta_max_iter=BORUTA_MAX_ITER if REFIT_BORUTA else None,
        boruta_n_estimators=BORUTA_N_ESTIMATORS if REFIT_BORUTA else None,
        param_grid=PARAM_GRID,
        grid_size=(len(PARAM_GRID["n_estimators"]) * len(PARAM_GRID["max_depth"])
                   * len(PARAM_GRID["min_child_samples"])),
        gridsearch=dict(scoring="accuracy", cv="StratifiedKFold(5, shuffle=True, "
                                              "random_state=seed)", refit=True),
        estimator="LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1)",
        calendar_drop_cols=CALENDAR_DROP_COLS,
        workers=WORKERS,
        versions=dict(python=sys.version.split()[0], lightgbm=lightgbm.__version__,
                      scikit_learn=sklearn.__version__, joblib=joblib.__version__,
                      numpy=np.__version__, pandas=pd.__version__),
        rf_reference_dir=os.path.relpath(RF_RESULTS_DIR, BASE),
        windows_detail=window_detail,
        outputs=[os.path.relpath(p, BASE) for p in paths],
        notes=[
            "max_depth=-1 is LightGBM's unlimited depth, the analogue of the forest's "
            "max_depth=None; num_leaves stays at its default 31.",
            "num_threads is deliberately NOT passed alongside n_jobs=1: it is an alias and "
            "LightGBM 4.x would warn on every fit. Threads are capped via OMP_NUM_THREADS=1.",
            "Iterations are parallelised across processes; GridSearchCV and LightGBM both run "
            "with n_jobs=1, which does not affect results.",
        ],
    )
    path = os.path.join(OUT_DIR, "manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def main():
    print(f"run_lgbm_final.py | {datetime.now().isoformat(timespec='seconds')}")
    print(f"  lightgbm {lightgbm.__version__} | scikit-learn {sklearn.__version__} | "
          f"joblib {joblib.__version__}")
    unknown = [a for a in ARMS if a not in ARM_PAIRS]
    if unknown:
        die(f"unknown arm(s) {unknown}; PASC_LGBM_ARMS accepts {sorted(ARM_PAIRS)}")
    print(f"  windows={WINDOWS} arms={ARMS} iterations={N_ITER_RUN} "
          f"selection={SELECTION_SOURCE} workers={WORKERS}"
          f"{' | FAST (one-cell grid)' if FAST else ''}")
    for a in ARMS:
        t, r = ARM_PAIRS[a]
        print(f"      {a:<7} {t} - {r}")
    print(f"  memory: {WORKERS} worker interpreters at ~250 MB each need ~"
          f"{0.25 * WORKERS + 1.0:.1f} GB plus the parent; a smaller LSF MEMLIMIT is killed "
          f"with TERM_MEMLIMIT (lower PASC_LGBM_WORKERS if the job is tight)")
    os.makedirs(OUT_DIR, exist_ok=True)

    paths, window_detail = [], {}
    for window in WINDOWS:
        wd = load_window(window)
        print_window_report(wd)
        window_detail[window] = dict(
            parquet=os.path.relpath(wd["parquet"], BASE),
            families=os.path.relpath(wd["families_path"], BASE),
            n_rows=wd["n_rows"],
            n_positives=wd["n_pos"],
            calendar_cols_dropped=wd["dropped"],
            config_sizes={c: len([x for x in wd["configs"][c] if x in wd["df"].columns])
                          for c in CONFIGS},
        )
        for config in CONFIGS:
            paths.append(run_config(wd, config))
        del wd  # one window's frame resident at a time; LSF caps the whole job

    frames = [pd.read_csv(p) for p in paths if os.path.exists(p) and os.path.getsize(p) > 0]
    combined = os.path.join(OUT_DIR, "all_iterations_lgbm.csv")
    if frames:
        # workers finish out of order; restore a deterministic row order
        allit = pd.concat(frames, ignore_index=True)
        allit = allit.sort_values(["feature_window", "config", "iteration"]).reset_index(drop=True)
        allit.to_csv(combined, index=False)
        print(f"\nWrote {os.path.relpath(combined, BASE)} ({len(allit)} rows)")
        paths.append(combined)

    manifest = write_manifest(window_detail, paths)
    print(f"Wrote {os.path.relpath(manifest, BASE)}")

    if SUMMARIZE:
        print("\nRunning scripts/summarize_lgbm_table.py ...\n", flush=True)
        subprocess.run([sys.executable, os.path.join(BASE, "scripts", "summarize_lgbm_table.py")],
                       cwd=BASE, check=False)
    else:
        print("\nNext: python scripts/summarize_lgbm_table.py")


if __name__ == "__main__":
    main()

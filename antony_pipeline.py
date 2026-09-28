"""
Modeling pipeline for the Antony et al. replication.

Implements:
- 10-iteration repeated stratified hold-out (80/20)
- Training set balancing (subsample negatives to match positives)
- Prevalence filter (<1%) on training set only
- Boruta feature selection on training set only
- Nested 5-fold CV grid search (optimize accuracy)
- Logistic Regression and Random Forest classifiers
- Evaluation on imbalanced test set (AUROC, AUPRC)
- SHAP value computation per iteration
"""

import os
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field
from typing import Optional

# Configurable parallelism. Avoids "can't start new thread" on HPC nodes where
# nested n_jobs=-1 (GridSearchCV x RandomForest) exhausts the thread/process limit.
# Override with env var ANTONY_N_JOBS; default is conservative (4).
N_JOBS = int(os.environ.get("ANTONY_N_JOBS", "4"))

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from boruta import BorutaPy


# ============================================================================
# Data structures for results
# ============================================================================

@dataclass
class IterationResult:
    """Results from a single hold-out iteration."""
    iteration: int
    model_type: str           # 'LR' or 'RF'
    cohort_name: str
    auroc: float
    auprc: float
    n_features_before_filter: int
    n_features_after_filter: int
    n_features_after_boruta: int
    best_params: dict
    selected_features: list
    y_test: np.ndarray
    y_pred_proba: np.ndarray
    seed: int = 0
    shap_values: Optional[np.ndarray] = None
    shap_feature_names: Optional[list] = None


@dataclass
class PipelineResults:
    """Aggregated results across all iterations for one cohort+model."""
    cohort_name: str
    model_type: str
    iterations: list = field(default_factory=list)

    @property
    def aurocs(self):
        return np.array([it.auroc for it in self.iterations])

    @property
    def auprcs(self):
        return np.array([it.auprc for it in self.iterations])

    def summary(self):
        """Return median and IQR as a dict."""
        def _stats(arr):
            return {
                "median": float(np.median(arr)),
                "q25": float(np.percentile(arr, 25)),
                "q75": float(np.percentile(arr, 75)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            }
        return {
            "cohort": self.cohort_name,
            "model": self.model_type,
            "n_iterations": len(self.iterations),
            "auroc": _stats(self.aurocs),
            "auprc": _stats(self.auprcs),
        }


# ============================================================================
# Hyperparameter grids
#
# Aligned with Antony et al. (Ref 35 — scikit-learn 3.1 / GridSearchCV):
#   RF: n_estimators ∈ {100,300,500}, max_depth ∈ {5,10,20,None},
#       min_samples_leaf ∈ {1,5,10}
#   LR: C ∈ {0.001..100}, penalty ∈ {l1,l2}
# Inner CV uses scoring='accuracy' per Antony's protocol.
# ============================================================================

LR_PARAM_GRID = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["saga"],
    "max_iter": [2000],
}

RF_PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 10, 20, None],
    "min_samples_leaf": [1, 5, 10],
}


# ============================================================================
# Helper functions
# ============================================================================

def balance_training_set(X_train, y_train, random_state=42):
    """
    Subsample negatives to match the number of positives.

    Returns:
        X_balanced, y_balanced (numpy arrays)
    """
    rng = np.random.default_rng(random_state)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    n_pos = len(pos_idx)
    if n_pos == 0 or len(neg_idx) == 0:
        return X_train, y_train

    # Subsample negatives to match positives
    if len(neg_idx) > n_pos:
        neg_sampled = rng.choice(neg_idx, size=n_pos, replace=False)
    else:
        neg_sampled = neg_idx

    selected = np.concatenate([pos_idx, neg_sampled])
    rng.shuffle(selected)

    return X_train[selected], y_train[selected]


def prevalence_filter(X_train, feature_names, min_prev=0.01, forced_features=None):
    """
    Remove binary features present in <min_prev of training patients.

    Non-binary features (e.g., age, LOS) are always kept.
    Features in *forced_features* are always kept regardless of prevalence.

    Returns:
        kept_indices, kept_names, n_dropped
    """
    forced_set = set(forced_features) if forced_features else set()
    n = X_train.shape[0]
    kept_indices = []
    kept_names = []
    n_dropped = 0

    for i, name in enumerate(feature_names):
        # Always keep forced features
        if name in forced_set:
            kept_indices.append(i)
            kept_names.append(name)
            continue

        col = X_train[:, i]
        unique_vals = np.unique(col[~np.isnan(col)])

        # If binary (only 0 and 1), check prevalence
        if set(unique_vals).issubset({0.0, 1.0}):
            prev = np.nanmean(col)
            if prev < min_prev:
                n_dropped += 1
                continue

        kept_indices.append(i)
        kept_names.append(name)

    return kept_indices, kept_names, n_dropped


def boruta_select(X_train, y_train, feature_names, random_state=42,
                  max_iter=100, n_estimators=500, forced_features=None,
                  boruta_perc=100):
    """
    Run Boruta feature selection using a Random Forest estimator.

    Args:
        forced_features: list of feature names that are always retained
            regardless of Boruta's decision (e.g. engagement controls).
        boruta_perc: BorutaPy `perc` threshold (default 100 = faithful, strictest).
            Lower values (e.g. 90) loosen selection by comparing against a
            percentile of shadow-feature importances rather than the max.

    Returns:
        selected_indices, selected_names
    """
    # Impute NaNs for Boruta (it cannot handle missing values)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_train).astype(np.float32)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=7,
        n_jobs=N_JOBS,
        random_state=random_state,
    )

    selector = BorutaPy(
        estimator=rf,
        n_estimators=n_estimators,
        max_iter=max_iter,
        random_state=random_state,
        perc=boruta_perc,
        two_step=True,
        verbose=0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(X_imp, y_train)

    mask = selector.support_
    selected_indices = np.where(mask)[0].tolist()
    selected_names = [feature_names[i] for i in selected_indices]

    # Force-include mandatory features (e.g. engagement controls)
    if forced_features:
        forced_set = set(forced_features)
        for i, name in enumerate(feature_names):
            if name in forced_set and i not in selected_indices:
                selected_indices.append(i)
                selected_names.append(name)
        selected_indices.sort()
        selected_names = [feature_names[i] for i in selected_indices]

    # Fallback: if Boruta selects nothing, keep all
    if len(selected_indices) == 0:
        selected_indices = list(range(len(feature_names)))
        selected_names = list(feature_names)

    return selected_indices, selected_names


def compute_shap_values(model, X_test, model_type, max_shap_samples=2000):
    """
    Compute SHAP values for a trained model on the test set.

    Args:
        max_shap_samples: If X_test has more rows than this, subsample
            to reduce computation time. Set to None to disable.

    Returns:
        shap_values (numpy array): SHAP values for the positive class.
    """
    try:
        import shap

        # Subsample test set for efficiency
        if max_shap_samples is not None and X_test.shape[0] > max_shap_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(X_test.shape[0], max_shap_samples, replace=False)
            X_shap = X_test[idx]
        else:
            X_shap = X_test

        if model_type == "LR":
            # Use LinearExplainer for logistic regression
            masker = shap.maskers.Independent(X_shap, max_samples=min(100, X_shap.shape[0]))
            explainer = shap.LinearExplainer(model, masker)
            sv = explainer.shap_values(X_shap)
        else:
            # Use TreeExplainer for Random Forest
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_shap)
            # For binary classification, take the positive class
            if isinstance(sv, list) and len(sv) == 2:
                sv = sv[1]

        return sv
    except Exception as e:
        print(f"  WARNING: SHAP computation failed: {e}")
        return None


# ============================================================================
# Single iteration
# ============================================================================

def run_single_iteration(
    X, y,
    feature_names,
    iteration,
    model_type,
    cohort_name,
    random_state_base=42,
    boruta_max_iter=100,
    boruta_n_estimators=500,
    compute_shap=True,
    forced_features=None,
    boruta_perc=100,
):
    """
    Execute one iteration of the Antony pipeline.

    Steps:
      1. Stratified 80/20 split
      2. Balance training set
      3. Prevalence filter on training
      4. Boruta on training
      5. Nested 5-fold CV grid search
      6. Retrain best model on full balanced training
      7. Evaluate on imbalanced test set

    Args:
        forced_features: list of feature names always retained past Boruta
            (e.g. engagement controls).

    Returns:
        IterationResult
    """
    seed = random_state_base + iteration * 1000
    feature_names = list(feature_names)

    print(f"\n  --- Iteration {iteration} | {model_type} | {cohort_name} ---")

    # 1. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    n_features_before = X_train.shape[1]
    print(f"    Split: train={len(y_train):,} (pos={y_train.sum():,}), "
          f"test={len(y_test):,} (pos={y_test.sum():,})")

    # 2. Balance training set
    X_train_bal, y_train_bal = balance_training_set(X_train, y_train, random_state=seed)
    print(f"    Balanced training: {len(y_train_bal):,} "
          f"(pos={y_train_bal.sum():,}, neg={(y_train_bal == 0).sum():,})")

    # 3. Prevalence filter (on balanced training)
    kept_idx, kept_names, n_dropped = prevalence_filter(X_train_bal, feature_names,
                                                        forced_features=forced_features)
    X_train_bal = X_train_bal[:, kept_idx]
    X_test_filt = X_test[:, kept_idx]
    n_after_filter = len(kept_names)
    print(f"    Prevalence filter: {n_features_before} → {n_after_filter} "
          f"(dropped {n_dropped})")

    # 4. Boruta (on balanced training)
    sel_idx, sel_names = boruta_select(
        X_train_bal, y_train_bal, kept_names,
        random_state=seed,
        max_iter=boruta_max_iter,
        n_estimators=boruta_n_estimators,
        forced_features=forced_features,
        boruta_perc=boruta_perc,
    )
    X_train_sel = X_train_bal[:, sel_idx]
    X_test_sel = X_test_filt[:, sel_idx]
    n_after_boruta = len(sel_names)
    print(f"    Boruta: {n_after_filter} → {n_after_boruta} features selected")

    # 5. Impute for model fitting
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_sel)
    X_test_imp = imputer.transform(X_test_sel)

    # 6. Grid search with nested 5-fold CV
    if model_type == "LR":
        # Standardize for LR
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        base_model = LogisticRegression(random_state=seed)
        param_grid = LR_PARAM_GRID
        X_gs_train = X_train_scaled
        X_gs_test = X_test_scaled
    else:
        # Inner model single-threaded to avoid nested parallelism with GridSearchCV
        base_model = RandomForestClassifier(random_state=seed, n_jobs=1)
        param_grid = RF_PARAM_GRID
        X_gs_train = X_train_imp
        X_gs_test = X_test_imp

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=N_JOBS,
        refit=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(X_gs_train, y_train_bal)

    best_params = grid_search.best_params_
    print(f"    Best params: {best_params}")

    # 7. The grid search already refits the best model on all training data
    best_model = grid_search.best_estimator_

    # 8. Evaluate on imbalanced test set
    y_pred_proba = best_model.predict_proba(X_gs_test)[:, 1]
    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    print(f"    AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

    # 9. SHAP values
    shap_vals = None
    if compute_shap:
        shap_vals = compute_shap_values(best_model, X_gs_test, model_type)

    return IterationResult(
        iteration=iteration,
        model_type=model_type,
        cohort_name=cohort_name,
        auroc=auroc,
        auprc=auprc,
        n_features_before_filter=n_features_before,
        n_features_after_filter=n_after_filter,
        n_features_after_boruta=n_after_boruta,
        best_params=best_params,
        selected_features=sel_names,
        y_test=y_test,
        y_pred_proba=y_pred_proba,
        seed=seed,
        shap_values=shap_vals,
        shap_feature_names=sel_names,
    )


# ============================================================================
# Full pipeline: 10 iterations x {LR, RF}
# ============================================================================

def run_full_pipeline(
    feature_df,
    feature_families,
    cohort_name,
    n_iterations=10,
    model_types=("LR", "RF"),
    random_state_base=42,
    boruta_max_iter=100,
    boruta_n_estimators=500,
    compute_shap=True,
    feature_subset=None,
    forced_features=None,
    feature_columns=None,
    boruta_perc=100,
):
    """
    Run the full Antony pipeline for a cohort.

    Args:
        feature_df: DataFrame with person_id, label, and feature columns
        feature_families: dict mapping family name -> list of feature columns
        cohort_name: e.g. 'all_patients', 'inpatients', 'outpatients'
        n_iterations: Number of repeated hold-out iterations (default: 10)
        model_types: Tuple of model types to train (default: ('LR', 'RF'))
        random_state_base: Base random state for reproducibility
        boruta_max_iter: Max Boruta iterations
        boruta_n_estimators: Number of estimators for Boruta RF
        compute_shap: Whether to compute SHAP values
        feature_subset: If provided, only use features from these families.
                       e.g., ['comorbidities', 'demographics']
        forced_features: list of feature names always retained past Boruta.
                       These are never dropped by prevalence filter or Boruta.
                       Typically engagement controls (f_eng_*).
        feature_columns: If provided, use this exact list of column names
                        instead of expanding from feature_subset/families.
                        Takes precedence over feature_subset.

    Returns:
        dict mapping model_type -> PipelineResults
    """
    print("\n" + "=" * 80)
    print(f"ANTONY PIPELINE: {cohort_name}")
    print("=" * 80)

    # Determine which features to use
    if feature_columns is not None:
        all_features = list(feature_columns)
        print(f"  Explicit feature_columns: {len(all_features)} columns")
    elif feature_subset is not None:
        all_features = []
        for family_name in feature_subset:
            all_features.extend(feature_families.get(family_name, []))
        print(f"  Feature subset: {feature_subset}")
    else:
        all_features = []
        for cols in feature_families.values():
            all_features.extend(cols)
        print(f"  Using all feature families")

    # Remove any features not present in the DataFrame
    available = [c for c in all_features if c in feature_df.columns]
    if len(available) < len(all_features):
        print(f"  WARNING: {len(all_features) - len(available)} features not found in DataFrame")
    all_features = available

    print(f"  Total features: {len(all_features)}")
    print(f"  Cohort size: {len(feature_df):,}")
    print(f"  Label distribution: {feature_df['label'].value_counts().to_dict()}")

    # Prepare numpy arrays
    X = feature_df[all_features].to_numpy(dtype=np.float32)
    y = feature_df["label"].to_numpy(dtype=int)

    # Run iterations for each model type
    all_results = {}
    for model_type in model_types:
        print(f"\n{'=' * 60}")
        print(f"MODEL: {model_type} | COHORT: {cohort_name}")
        print(f"{'=' * 60}")

        pipeline_results = PipelineResults(
            cohort_name=cohort_name,
            model_type=model_type,
        )

        for it in range(1, n_iterations + 1):
            result = run_single_iteration(
                X=X,
                y=y,
                feature_names=all_features,
                iteration=it,
                model_type=model_type,
                cohort_name=cohort_name,
                random_state_base=random_state_base,
                boruta_max_iter=boruta_max_iter,
                boruta_n_estimators=boruta_n_estimators,
                compute_shap=compute_shap,
                forced_features=forced_features,
                boruta_perc=boruta_perc,
            )
            pipeline_results.iterations.append(result)

        all_results[model_type] = pipeline_results

        # Print summary
        s = pipeline_results.summary()
        print(f"\n  {model_type} SUMMARY ({cohort_name}):")
        print(f"    AUROC — median: {s['auroc']['median']:.4f} "
              f"[IQR: {s['auroc']['q25']:.4f}–{s['auroc']['q75']:.4f}]")
        print(f"    AUPRC — median: {s['auprc']['median']:.4f} "
              f"[IQR: {s['auprc']['q25']:.4f}–{s['auprc']['q75']:.4f}]")

    return all_results


# ============================================================================
# Results export
# ============================================================================

def results_to_dataframe(all_results):
    """
    Convert a dict of {model_type: PipelineResults} to a summary DataFrame.

    Returns:
        DataFrame with one row per iteration per model.
    """
    rows = []
    for model_type, pr in all_results.items():
        for it in pr.iterations:
            rows.append({
                "cohort": it.cohort_name,
                "model": it.model_type,
                "iteration": it.iteration,
                "seed": it.seed,
                "auroc": it.auroc,
                "auprc": it.auprc,
                "n_features_initial": it.n_features_before_filter,
                "n_features_filtered": it.n_features_after_filter,
                "n_features_boruta": it.n_features_after_boruta,
                "best_params": str(it.best_params),
            })
    return pd.DataFrame(rows)


def print_summary_table(results_dict):
    """
    Print a formatted summary table across all cohorts and models.

    Args:
        results_dict: dict of {cohort_name: {model_type: PipelineResults}}
    """
    print("\n" + "=" * 100)
    print("FINAL RESULTS SUMMARY (Median [IQR])")
    print("=" * 100)
    print(f"{'Cohort':20s} | {'Model':5s} | {'AUROC':30s} | {'AUPRC':30s} | {'n_feat (Boruta)':15s}")
    print("-" * 100)

    for cohort_name, model_dict in results_dict.items():
        for model_type, pr in model_dict.items():
            s = pr.summary()
            auroc_str = f"{s['auroc']['median']:.4f} [{s['auroc']['q25']:.4f}–{s['auroc']['q75']:.4f}]"
            auprc_str = f"{s['auprc']['median']:.4f} [{s['auprc']['q25']:.4f}–{s['auprc']['q75']:.4f}]"

            boruta_counts = [it.n_features_after_boruta for it in pr.iterations]
            feat_str = f"{np.median(boruta_counts):.0f} [{np.percentile(boruta_counts, 25):.0f}–{np.percentile(boruta_counts, 75):.0f}]"

            print(f"{cohort_name:20s} | {model_type:5s} | {auroc_str:30s} | {auprc_str:30s} | {feat_str:15s}")

    print("=" * 100)


# ============================================================================
# Cross-site analysis stub (Antony Fig 6 analogue)
# ============================================================================

def run_cross_site_analysis(
    feature_df,
    feature_families,
    site_column="care_site_id",
    model_types=("LR", "RF"),
    random_state_base=42,
    results_dir=None,
):
    """
    Within-MSHS cross-site replication: train on one care site, test on
    the remaining sites.  Analogous to Antony's Fig 6 cross-partner
    evaluation.

    Requires ``feature_df`` to contain a ``site_column`` populated by
    :func:`antony_cohort.extract_care_site`.

    Returns:
        DataFrame with columns: train_site, model, auroc, auprc,
        n_train, n_test.
    """
    if site_column not in feature_df.columns:
        raise ValueError(
            f"'{site_column}' not found in feature_df. "
            f"Run antony_cohort.extract_care_site() first."
        )

    sites = feature_df[site_column].dropna().unique()
    if len(sites) < 2:
        print(f"  WARNING: Only {len(sites)} site(s) found — need ≥2 for cross-site.")
        return pd.DataFrame()

    print(f"\n{'=' * 80}")
    print(f"CROSS-SITE ANALYSIS: {len(sites)} sites")
    print(f"{'=' * 80}")

    all_features = []
    for cols in feature_families.values():
        all_features.extend(cols)
    all_features = [c for c in all_features if c in feature_df.columns]

    rows = []
    for train_site in sites:
        train_mask = feature_df[site_column] == train_site
        test_mask = feature_df[site_column] != train_site

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        if n_train < 50 or n_test < 50:
            print(f"  Skipping site {train_site}: train={n_train}, test={n_test}")
            continue

        X_train = feature_df.loc[train_mask, all_features].to_numpy(dtype=np.float32)
        y_train = feature_df.loc[train_mask, "label"].to_numpy(dtype=int)
        X_test = feature_df.loc[test_mask, all_features].to_numpy(dtype=np.float32)
        y_test = feature_df.loc[test_mask, "label"].to_numpy(dtype=int)

        if y_train.sum() == 0 or y_test.sum() == 0:
            print(f"  Skipping site {train_site}: no positives in train or test")
            continue

        print(f"\n  Train site={train_site} (n={n_train}, pos={y_train.sum()}) "
              f"→ Test others (n={n_test}, pos={y_test.sum()})")

        # Balance training
        X_train_bal, y_train_bal = balance_training_set(X_train, y_train)

        # Impute
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_bal)
        X_test_imp = imputer.transform(X_test)

        for model_type in model_types:
            if model_type == "LR":
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train_imp)
                X_te = scaler.transform(X_test_imp)
                base_model = LogisticRegression(
                    C=1.0, penalty="l2", solver="saga", max_iter=2000,
                    random_state=random_state_base,
                )
            else:
                X_tr, X_te = X_train_imp, X_test_imp
                base_model = RandomForestClassifier(
                    n_estimators=300, max_depth=10, min_samples_leaf=5,
                    n_jobs=N_JOBS, random_state=random_state_base,
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                base_model.fit(X_tr, y_train_bal)

            y_proba = base_model.predict_proba(X_te)[:, 1]
            auroc = roc_auc_score(y_test, y_proba)
            auprc = average_precision_score(y_test, y_proba)
            print(f"    {model_type}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

            rows.append({
                "train_site": int(train_site),
                "model": model_type,
                "auroc": auroc,
                "auprc": auprc,
                "n_train": n_train,
                "n_test": n_test,
            })

    result_df = pd.DataFrame(rows)

    if results_dir and not result_df.empty:
        import os
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, "cross_site_results.csv")
        result_df.to_csv(path, index=False)
        print(f"\n  Saved: {path}")

    return result_df

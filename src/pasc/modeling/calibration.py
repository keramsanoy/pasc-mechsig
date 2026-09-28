#!/usr/bin/env python
"""Calibration assessment, recalibration and decision-curve analysis (thesis 3.7, Appendix A.3).

Pure, importable functions. No file I/O, no plotting, no global state. The driver
(``scripts/run_calibration.py``) and the figure notebook (``notebooks/make_thesis_figures.ipynb``) own those.

References
---------
- Van Calster B, et al. "Calibration: the Achilles heel of predictive analytics."
  BMC Medicine 2019;17:230. (CITL, intercept/slope, flexible curve, ICI/E50/E90)
- Austin PC, Steyerberg EW. "The Integrated Calibration Index (ICI)..." Stat Med 2019.
- Vickers AJ, Elkin EB. "Decision curve analysis." Med Decis Making 2006;26:565-74.

Design notes
------------
- All metrics are computed on a single (y_true, p) vector; the driver loops seeds
  and aggregates with :func:`summarize_across_seeds`.
- ``logit``/``expit`` clip probabilities to ``[EPS, 1-EPS]`` so log-odds is finite.
- Recalibration is always fit on a *disjoint* split (``cal``) and scored on a held-out
  ``eval`` split, so post-recalibration metrics carry no in-sample optimism.
"""
from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.special import expit, logit as _scipy_logit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from statsmodels.nonparametric.smoothers_lowess import lowess

EPS = 1e-6


def _unpenalized_logreg():
    """Unpenalized logistic regression, robust across sklearn versions.

    sklearn 1.8 deprecates penalty=None in favour of C=np.inf but still emits a
    cosmetic UserWarning; we fit unpenalized (needed for an honest calibration
    slope) and silence only that message.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)


# ===========================================================================
# Primitives
# ===========================================================================
def clip_proba(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Clip probabilities to [eps, 1-eps] for finite log-odds."""
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Log-odds with clipping (avoids +/-inf at 0/1)."""
    return _scipy_logit(clip_proba(p, eps))


# ===========================================================================
# Assessment metrics (Van Calster hierarchy)
# ===========================================================================
def calibration_in_the_large(y_true: np.ndarray, p: np.ndarray) -> dict:
    """CITL: intercept a0 from logit(P(y))=a0+offset(logit(p)), slope fixed to 1.

    Fit by a 1-parameter logistic regression with the log-odds of ``p`` as a fixed
    offset (no coefficient). Also returns the simple mean difference mean(p)-mean(y).
    """
    z = logit(p).reshape(-1, 1)
    y = np.asarray(y_true, dtype=int)
    # Intercept-only logistic regression with z as offset: statsmodels handles
    # offsets cleanly; use it to avoid hand-rolling Newton steps.
    import statsmodels.api as sm

    const = np.ones_like(z)
    try:
        model = sm.GLM(y, const, family=sm.families.Binomial(), offset=z.ravel())
        res = model.fit()
        a0 = float(res.params[0])
    except Exception:
        a0 = float("nan")
    return {
        "citl": a0,
        "mean_pred_minus_obs": float(np.mean(p) - np.mean(y)),
        "mean_pred": float(np.mean(p)),
        "obs_rate": float(np.mean(y)),
    }


def calibration_intercept_slope(y_true: np.ndarray, p: np.ndarray) -> dict:
    """Fit logit(P(y=1)) = a + b*logit(p). Returns intercept a and slope b.

    b=1 and a=0 is ideal. b<1 indicates overfitting / over-extreme probabilities
    (the expected RF+undersampling signature).
    """
    z = logit(p).reshape(-1, 1)
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return {"intercept": float("nan"), "slope": float("nan")}
    # Unpenalised logistic regression on the single predictor logit(p).
    lr = _unpenalized_logreg()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        lr.fit(z, y)
    return {"intercept": float(lr.intercept_[0]), "slope": float(lr.coef_[0, 0])}


def flexible_calibration_curve(
    y_true: np.ndarray, p: np.ndarray, frac: float = 0.66, grid: Optional[np.ndarray] = None
) -> dict:
    """Nonparametric lowess of y on p (flexible calibration curve).

    Returns the lowess fit g(p) evaluated at each observation (``g_obs``, aligned to
    the input order) and, optionally, on a supplied probability ``grid`` for plotting.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y_true, dtype=float)
    # lowess returns sorted (x, yhat); request return_sorted=False to align to input.
    g_obs = lowess(y, p, frac=frac, it=0, return_sorted=False)
    g_obs = np.clip(g_obs, 0.0, 1.0)
    out = {"p_obs": p, "g_obs": g_obs}
    if grid is not None:
        sm_sorted = lowess(y, p, frac=frac, it=0, return_sorted=True)
        xs, ys = sm_sorted[:, 0], np.clip(sm_sorted[:, 1], 0.0, 1.0)
        out["grid"] = np.asarray(grid, dtype=float)
        out["g_grid"] = np.interp(grid, xs, ys)
    return out


def calibration_error_indices(y_true: np.ndarray, p: np.ndarray, frac: float = 0.66) -> dict:
    """ICI / E50 / E90 from the flexible (lowess) curve.

    With g(p) the lowess prediction at each point:
      ICI = mean(|g(p)-p|); E50 = median(|g(p)-p|); E90 = 90th percentile(|g(p)-p|).
    """
    curve = flexible_calibration_curve(y_true, p, frac=frac)
    diff = np.abs(curve["g_obs"] - curve["p_obs"])
    return {
        "ici": float(np.mean(diff)),
        "e50": float(np.median(diff)),
        "e90": float(np.percentile(diff, 90)),
    }


def assess(y_true: np.ndarray, p: np.ndarray, frac: float = 0.66) -> dict:
    """Full §3 assessment block on one (y_true, p) vector."""
    out = {}
    out.update(calibration_in_the_large(y_true, p))
    out.update(calibration_intercept_slope(y_true, p))
    out.update(calibration_error_indices(y_true, p, frac=frac))
    return out


# ===========================================================================
# Recalibration maps (fit on cal, return a callable applied to eval)
# ===========================================================================
def fit_logistic_recalibration(y_cal: np.ndarray, p_cal: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Platt-style: LogisticRegression of y on logit(p). Returns map p_raw -> p_corr."""
    z = logit(p_cal).reshape(-1, 1)
    lr = _unpenalized_logreg()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        lr.fit(z, np.asarray(y_cal, dtype=int))

    def _map(p_new: np.ndarray) -> np.ndarray:
        zz = logit(p_new).reshape(-1, 1)
        return lr.predict_proba(zz)[:, 1]

    _map.kind = "logistic"  # type: ignore[attr-defined]
    return _map


def fit_isotonic_recalibration(y_cal: np.ndarray, p_cal: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Isotonic regression of y on p_raw (monotone, out_of_bounds='clip')."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(p_cal, dtype=float), np.asarray(y_cal, dtype=int))

    def _map(p_new: np.ndarray) -> np.ndarray:
        return iso.predict(np.asarray(p_new, dtype=float))

    _map.kind = "isotonic"  # type: ignore[attr-defined]
    return _map


def prior_shift_map(pi: float) -> Callable[[np.ndarray], np.ndarray]:
    """Analytic prior-shift: logit(p_corr)=logit(p_raw)+log(pi/(1-pi)).

    Training prevalence is 0.5 (1:1 undersampling), so its log-odds is 0 and drops
    out. No fitting; map p_raw directly. ``pi`` is the true outcome prevalence.
    """
    shift = float(np.log(pi / (1.0 - pi)))

    def _map(p_new: np.ndarray) -> np.ndarray:
        return expit(logit(p_new) + shift)

    _map.kind = "prior_shift"  # type: ignore[attr-defined]
    _map.shift = shift  # type: ignore[attr-defined]
    return _map


# ===========================================================================
# Per-seed recalibration: split test -> cal/eval, fit on cal, assess on eval
# ===========================================================================
def recalibrate_and_assess_seed(
    y_test: np.ndarray,
    p_raw: np.ndarray,
    seed: int,
    pi: float,
    frac: float = 0.66,
    cal_size: float = 0.5,
) -> list[dict]:
    """One seed: split test set 50/50 (stratified), fit each map on cal, assess eval.

    Returns a list of metric dicts, one per (method, phase). Phases:
      - method='raw',         phase='eval'   : raw p on the eval split (baseline)
      - method='logistic',    phase='eval'   : logistic-recalibrated
      - method='isotonic',    phase='eval'   : isotonic-recalibrated
      - method='prior_shift', phase='eval'   : analytic prior-shift
    All assessed on the SAME eval split so methods are directly comparable.
    """
    y_test = np.asarray(y_test, dtype=int)
    p_raw = np.asarray(p_raw, dtype=float)

    strat = y_test if (y_test.sum() >= 2 and (len(y_test) - y_test.sum()) >= 2) else None
    idx = np.arange(len(y_test))
    cal_idx, eval_idx = train_test_split(
        idx, test_size=(1.0 - cal_size), random_state=seed, stratify=strat
    )
    y_cal, p_cal = y_test[cal_idx], p_raw[cal_idx]
    y_eval, p_eval = y_test[eval_idx], p_raw[eval_idx]

    maps = {
        "logistic": fit_logistic_recalibration(y_cal, p_cal),
        "isotonic": fit_isotonic_recalibration(y_cal, p_cal),
        "prior_shift": prior_shift_map(pi),
    }

    rows = []

    def _row(method, p_vec):
        m = assess(y_eval, p_vec, frac=frac)
        m.update({"seed": seed, "method": method, "phase": "eval",
                  "n_eval": int(len(y_eval)), "pos_eval": int(y_eval.sum())})
        return m

    rows.append(_row("raw", p_eval))
    for name, mp in maps.items():
        rows.append(_row(name, mp(p_eval)))
    return rows


# ===========================================================================
# Aggregation across seeds (mean + 95% CI)
# ===========================================================================
_T95 = 1.959963984540054  # normal approx; per-seed n is small (10) but symmetric CI is fine


def _mean_ci(vals: np.ndarray) -> tuple[float, float, float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    m = float(np.mean(vals))
    if n == 1:
        return m, 0.0, m, m
    sd = float(np.std(vals, ddof=1))
    half = _T95 * sd / np.sqrt(n)
    return m, sd, m - half, m + half


_METRIC_KEYS = ["citl", "mean_pred_minus_obs", "intercept", "slope",
                "ici", "e50", "e90", "mean_pred", "obs_rate"]


def summarize_across_seeds(rows: list[dict]) -> pd.DataFrame:
    """Aggregate per-seed metric rows to mean + 95% CI per (method, metric).

    Input rows must carry 'seed', 'method' and the metric keys. Returns a tidy
    DataFrame: method, metric, mean, sd, ci95_lo, ci95_hi, n_seeds.
    """

    df = pd.DataFrame(rows)
    out = []
    for method, sub in df.groupby("method"):
        for key in _METRIC_KEYS:
            if key not in sub:
                continue
            m, sd, lo, hi = _mean_ci(sub[key].to_numpy())
            out.append({"method": method, "metric": key, "mean": m, "sd": sd,
                        "ci95_lo": lo, "ci95_hi": hi, "n_seeds": int(sub["seed"].nunique())})
    return pd.DataFrame(out)


# ===========================================================================
# Decision-curve analysis (Vickers & Elkin 2006)
# ===========================================================================
def net_benefit(y_true: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Net benefit at each threshold pt: NB = TP/n - (FP/n)*(pt/(1-pt)).

    ``predicted positive`` = p >= pt.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    nb = np.empty(len(thresholds), dtype=float)
    for i, pt in enumerate(thresholds):
        pred = p >= pt
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        w = pt / (1.0 - pt) if pt < 1.0 else np.inf
        nb[i] = tp / n - (fp / n) * w
    return nb


def net_benefit_treat_all(y_true: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Treat-all reference: NB = prev - (1-prev)*(pt/(1-pt))."""
    y = np.asarray(y_true, dtype=int)
    prev = float(np.mean(y))
    w = thresholds / (1.0 - thresholds)
    return prev - (1.0 - prev) * w


def decision_curve_seed(
    y_test: np.ndarray,
    p_raw: np.ndarray,
    p_recal: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    """Per-seed net-benefit curves for raw, recalibrated, treat-all, treat-none."""
    return {
        "raw": net_benefit(y_test, p_raw, thresholds),
        "recal": net_benefit(y_test, p_recal, thresholds),
        "treat_all": net_benefit_treat_all(y_test, thresholds),
        "treat_none": np.zeros(len(thresholds), dtype=float),
    }

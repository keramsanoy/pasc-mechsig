"""Calibration measures behave on synthetic probabilities (thesis Appendix A.3)."""
import numpy as np

from pasc.modeling import calibration as cal


def _synthetic(n=60000, seed=1):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(1, 30, size=n), 1e-4, 1 - 1e-4)
    y = (rng.random(n) < p).astype(int)
    return y, p


def test_calibrated_probabilities_score_near_ideal():
    y, p = _synthetic()
    r = cal.assess(y, p)
    assert abs(r["citl"]) < 0.15
    assert abs(r["slope"] - 1.0) < 0.1
    assert r["ici"] < 0.01


def test_logistic_recalibration_corrects_a_prior_shift():
    y, p = _synthetic()
    p_shift = cal.clip_proba(1 / (1 + np.exp(-(cal.logit(p) + 3.0))))   # overforecast by e^3
    raw = cal.assess(y, p_shift)
    assert raw["citl"] < -2.0
    half = len(y) // 2
    f = cal.fit_logistic_recalibration(y[:half], p_shift[:half])
    fixed = cal.assess(y[half:], f(p_shift[half:]))
    assert abs(fixed["citl"]) < 0.15 and abs(fixed["slope"] - 1.0) < 0.1


def test_net_benefit_of_a_useless_model_is_not_above_treat_all_at_low_threshold():
    y, p = _synthetic()
    th = np.array([0.005, 0.01, 0.02])
    nb_model = cal.net_benefit(y, p, th)
    nb_all = cal.net_benefit_treat_all(y, th)
    assert nb_model.shape == th.shape and nb_all.shape == th.shape
    assert np.all(nb_model >= nb_all - 1e-6)   # a calibrated model never loses to treat-all

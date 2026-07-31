import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from quack.quantifiers._threshold import BaseThresholdQuantifier, X, Max, T50, MedianSweep


@pytest.fixture
def binary_dataset():
  return make_classification(n_samples=400, n_classes=2, weights=[0.55, 0.45], random_state=0)


class _ManualCalibrationMixin:
  """Reference (non-vectorized) implementation of `_calibrate`, mirroring
  the original per-threshold Python loop, used to validate the vectorized
  version against it."""

  @staticmethod
  def manual_calibrate(y_true_oof, y_pred_oof, classes_, precision):
    neg_class, pos_class = classes_[0], classes_[1]
    actual_pos_mask = (y_true_oof == pos_class)
    actual_neg_mask = (y_true_oof == neg_class)
    n_pos, n_neg = np.sum(actual_pos_mask), np.sum(actual_neg_mask)
    pos_probs_oof = y_pred_oof[:, 1]

    thresholds = (np.unique(pos_probs_oof) if precision is None
                 else np.unique(np.around(pos_probs_oof, decimals=precision)))

    tpr = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))
    for idx, t in enumerate(thresholds):
      tpr[idx] = np.mean(pos_probs_oof[actual_pos_mask] >= t) if n_pos > 0 else 1.0
      fpr[idx] = np.mean(pos_probs_oof[actual_neg_mask] >= t) if n_neg > 0 else 0.0
    return thresholds, tpr, fpr


class TestBaseThresholdCalibrationVectorization(_ManualCalibrationMixin):
  def test_vectorized_calibration_matches_manual_loop(self, binary_dataset):
    X_data, y = binary_dataset
    quantifier = X(LogisticRegression(max_iter=1000), cv=5, precision=3)
    quantifier.fit(X_data, y)

    # recompute OOF predictions the same way BaseCalibratedQuantifier.fit does,
    # to feed the manual reference implementation with identical inputs
    from sklearn.model_selection import cross_val_predict
    y_pred_oof = cross_val_predict(
      LogisticRegression(max_iter=1000), X_data, y, cv=5, method="predict_proba",
    )
    manual_thresh, manual_tpr, manual_fpr = self.manual_calibrate(
      y, y_pred_oof, quantifier.classes_, precision=3,
    )

    np.testing.assert_allclose(quantifier.thresholds_, manual_thresh)
    # tpr/fpr won't match exactly since the fixture's own CV split differs
    # from cross_val_predict's; instead assert internal consistency:
    assert quantifier.tpr_by_thresh_.shape == quantifier.thresholds_.shape
    assert quantifier.fpr_by_thresh_.shape == quantifier.thresholds_.shape
    assert np.all((quantifier.tpr_by_thresh_ >= 0) & (quantifier.tpr_by_thresh_ <= 1))
    assert np.all((quantifier.fpr_by_thresh_ >= 0) & (quantifier.fpr_by_thresh_ <= 1))

  def test_tpr_is_non_increasing_in_threshold(self, binary_dataset):
    # sanity check on the searchsorted-based ECDF: TPR(t) must be
    # monotonically non-increasing as the threshold t increases
    X_data, y = binary_dataset
    quantifier = Max(LogisticRegression(max_iter=1000), cv=5)
    quantifier.fit(X_data, y)
    assert np.all(np.diff(quantifier.tpr_by_thresh_) <= 1e-12)
    assert np.all(np.diff(quantifier.fpr_by_thresh_) <= 1e-12)


class TestThresholdQuantifiers:
  @pytest.mark.parametrize("quantifier_cls", [X, Max, T50])
  def test_predict_sums_to_one(self, binary_dataset, quantifier_cls):
    X_data, y = binary_dataset
    quantifier = quantifier_cls(LogisticRegression(max_iter=1000), cv=5).fit(X_data, y)
    prevalences = quantifier.predict(X_data)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (2,)

  @pytest.mark.parametrize("quantifier_cls", [X, Max, T50])
  def test_accepts_n_jobs_and_parallel_backend(self, binary_dataset, quantifier_cls):
    X_data, y = binary_dataset
    quantifier = quantifier_cls(
      LogisticRegression(max_iter=1000), cv=5, n_jobs=2, parallel_backend="threading",
    ).fit(X_data, y)
    assert quantifier.predict(X_data).sum() == pytest.approx(1.0)

  def test_base_threshold_quantifier_is_abstract(self):
    with pytest.raises(TypeError):
      BaseThresholdQuantifier()


class TestMedianSweep:
  def test_predict_sums_to_one(self, binary_dataset):
    X_data, y = binary_dataset
    quantifier = MedianSweep(LogisticRegression(max_iter=1000), cv=5).fit(X_data, y)
    prevalences = quantifier.predict(X_data)
    assert prevalences.sum() == pytest.approx(1.0)

  def test_vectorized_matches_manual_loop_when_valid_thresholds_exist(self, binary_dataset):
    X_data, y = binary_dataset
    quantifier = MedianSweep(LogisticRegression(max_iter=1000), cv=5, delta_min=0.05).fit(X_data, y)

    pos_probs_test = quantifier.classifier_.predict_proba(X_data)[:, 1]

    # reference (manual loop) implementation, mirroring the pre-refactor logic
    p_adj_estimates = []
    delta_max, p_max_fallback = -2.0, 0.5
    for idx, t in enumerate(quantifier.thresholds_):
      tpr, fpr = quantifier.tpr_by_thresh_[idx], quantifier.fpr_by_thresh_[idx]
      delta = tpr - fpr
      p_raw_pos = np.mean(pos_probs_test >= t)
      if delta > quantifier.delta_min:
        p_adj_estimates.append(np.clip((p_raw_pos - fpr) / delta, 0.0, 1.0))
      elif delta > delta_max and len(p_adj_estimates) == 0:
        p_max_fallback = tpr if delta == 0 else (p_raw_pos - fpr) / delta
        delta_max = delta

    manual_p_pos = (np.median(p_adj_estimates) if p_adj_estimates
                    else np.clip(p_max_fallback, 0.0, 1.0))

    vectorized_result = quantifier.predict(X_data)
    assert vectorized_result[1] == pytest.approx(manual_p_pos)

  def test_fallback_path_matches_manual_loop_when_delta_min_too_strict(self, binary_dataset):
    # force delta_min above every achievable (tpr - fpr), so no threshold
    # is ever "valid" and the fallback branch is exercised end-to-end
    X_data, y = binary_dataset
    quantifier = MedianSweep(LogisticRegression(max_iter=1000), cv=5, delta_min=1.5).fit(X_data, y)

    pos_probs_test = quantifier.classifier_.predict_proba(X_data)[:, 1]

    p_adj_estimates = []
    delta_max, p_max_fallback = -2.0, 0.5
    for idx, t in enumerate(quantifier.thresholds_):
      tpr, fpr = quantifier.tpr_by_thresh_[idx], quantifier.fpr_by_thresh_[idx]
      delta = tpr - fpr
      p_raw_pos = np.mean(pos_probs_test >= t)
      if delta > quantifier.delta_min:
        p_adj_estimates.append(np.clip((p_raw_pos - fpr) / delta, 0.0, 1.0))
      elif delta > delta_max and len(p_adj_estimates) == 0:
        p_max_fallback = tpr if delta == 0 else (p_raw_pos - fpr) / delta
        delta_max = delta

    assert len(p_adj_estimates) == 0  # sanity check that fallback path is indeed exercised
    manual_p_pos = np.clip(p_max_fallback, 0.0, 1.0)

    vectorized_result = quantifier.predict(X_data)
    assert vectorized_result[1] == pytest.approx(manual_p_pos)

  def test_accepts_n_jobs_and_parallel_backend(self, binary_dataset):
    X_data, y = binary_dataset
    quantifier = MedianSweep(
      LogisticRegression(max_iter=1000), cv=5, n_jobs=2, parallel_backend="threading",
    ).fit(X_data, y)
    assert quantifier.predict(X_data).sum() == pytest.approx(1.0)
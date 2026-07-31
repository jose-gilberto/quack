import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

from quack.quantifiers import CC, PCC, ACC, PACC
from quack.quantifiers.base import normalize_prevalence


class _NoProbaClassifier(BaseEstimator, ClassifierMixin):
  """A classifier deliberately missing `predict_proba`."""

  def fit(self, X, y):
    self.classes_ = np.unique(y)
    return self

  def predict(self, X):
    return np.full(X.shape[0], self.classes_[0])


@pytest.fixture
def binary_dataset():
  return make_classification(n_samples=400, n_classes=2, weights=[0.6, 0.4], random_state=0)


@pytest.fixture
def multiclass_dataset():
  return make_classification(n_samples=400, n_classes=3, n_informative=6, random_state=0)


class TestNormalizePrevalence:
  def test_normalizes_to_sum_one(self):
    result = normalize_prevalence(np.array([3.0, 7.0]), n_classes=2)
    assert result.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(result, [0.3, 0.7])

  def test_does_not_clip_components_above_one_before_normalizing(self):
    # regression test: raw counts/scores aren't individually bounded by 1
    # (e.g. CC's class counts), so clipping each component to 1.0 before
    # dividing by the total would silently distort the resulting ratios
    result = normalize_prevalence(np.array([40.0, 10.0]), n_classes=2)
    np.testing.assert_allclose(result, [0.8, 0.2])

  def test_clips_negative_noise_to_zero(self):
    result = normalize_prevalence(np.array([-1e-9, 1.0]), n_classes=2)
    assert np.all(result >= 0.0)
    assert result.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(result, [0.0, 1.0])

  def test_falls_back_to_uniform_when_all_zero(self):
    result = normalize_prevalence(np.array([0.0, 0.0, 0.0]), n_classes=3)
    np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3])


class TestCC:
  def test_predict_sums_to_one(self, binary_dataset):
    X, y = binary_dataset
    quantifier = CC(LogisticRegression(max_iter=1000)).fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (2,)

  def test_vectorized_counting_matches_manual_dict_approach(self, binary_dataset):
    X, y = binary_dataset
    quantifier = CC(LogisticRegression(max_iter=1000)).fit(X, y)
    y_pred = quantifier.classifier_.predict(X)

    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    pred_counts = dict(zip(unique_pred, counts_pred))
    manual = np.array([pred_counts.get(c, 0) for c in quantifier.classes_], dtype=float)
    manual = normalize_prevalence(manual, quantifier.n_classes_)

    np.testing.assert_allclose(quantifier.predict(X), manual)

  def test_multiclass_supported(self, multiclass_dataset):
    X, y = multiclass_dataset
    quantifier = CC(LogisticRegression(max_iter=1000)).fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.shape == (3,)
    assert prevalences.sum() == pytest.approx(1.0)


class TestPCC:
  def test_predict_sums_to_one(self, binary_dataset):
    X, y = binary_dataset
    quantifier = PCC(LogisticRegression(max_iter=1000)).fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)

  def test_raises_without_predict_proba(self, binary_dataset):
    X, y = binary_dataset
    with pytest.raises(TypeError, match="predict_proba"):
      PCC(_NoProbaClassifier()).fit(X, y)


class TestACC:
  def test_predict_sums_to_one(self, binary_dataset):
    X, y = binary_dataset
    quantifier = ACC(LogisticRegression(max_iter=1000), cv=5).fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (2,)

  def test_raises_value_error_for_multiclass(self, multiclass_dataset):
    X, y = multiclass_dataset
    with pytest.raises(ValueError, match="binary quantification"):
      ACC(LogisticRegression(max_iter=1000)).fit(X, y)

  def test_accepts_n_jobs_and_parallel_backend(self, binary_dataset):
    X, y = binary_dataset
    quantifier = ACC(LogisticRegression(max_iter=1000), cv=5, n_jobs=2, parallel_backend="threading")
    quantifier.fit(X, y)
    assert quantifier.predict(X).sum() == pytest.approx(1.0)


class TestPACC:
  def test_predict_sums_to_one(self, binary_dataset):
    X, y = binary_dataset
    quantifier = PACC(LogisticRegression(max_iter=1000), cv=5).fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)

  def test_raises_value_error_for_multiclass(self, multiclass_dataset):
    X, y = multiclass_dataset
    with pytest.raises(ValueError, match="binary quantification"):
      PACC(LogisticRegression(max_iter=1000)).fit(X, y)

  def test_raises_type_error_without_predict_proba(self, binary_dataset):
    X, y = binary_dataset
    with pytest.raises(TypeError, match="predict_proba"):
      PACC(_NoProbaClassifier()).fit(X, y)

  def test_accepts_n_jobs_and_parallel_backend(self, binary_dataset):
    X, y = binary_dataset
    quantifier = PACC(LogisticRegression(max_iter=1000), cv=5, n_jobs=2, parallel_backend="threading")
    quantifier.fit(X, y)
    assert quantifier.predict(X).sum() == pytest.approx(1.0)
import time
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin

from quack.quantifiers.base import BaseCalibratedQuantifier, BaseMixtureQuantifier


class _DummyCalibratedQuantifier(BaseCalibratedQuantifier):
  """Minimal concrete subclass exercising the base CV/parallel pipeline."""

  def _get_oof_method(self) -> str:
    return "predict"

  def _calibrate(self, y_true_oof, y_pred_oof):
    self.oof_accuracy_ = np.mean(y_true_oof == y_pred_oof)

  def _quantify(self, X):
    y_pred = self.classifier_.predict(X)
    counts = np.array([np.mean(y_pred == c) for c in self.classes_])
    return counts


class _DummyProbaQuantifier(BaseCalibratedQuantifier):
  def _get_oof_method(self) -> str:
    return "predict_proba"

  def _calibrate(self, y_true_oof, y_pred_oof):
    pass

  def _quantify(self, X):
    return self.classifier_.predict_proba(X).mean(axis=0)


class _NoProbaClassifier(BaseEstimator, ClassifierMixin):
  """A classifier deliberately missing `predict_proba`."""

  def fit(self, X, y):
    self.classes_ = np.unique(y)
    return self

  def predict(self, X):
    return np.full(X.shape[0], self.classes_[0])


class _DummyMixtureQuantifier(BaseMixtureQuantifier):
  """Minimal concrete subclass exercising the CVXPY problem caching."""

  def fit(self, X, y):
    self.classes_, counts = np.unique(y, return_counts=True)
    self.n_classes_ = len(self.classes_)
    self.train_prevalence_ = counts / len(y)
    # simple 1-bin-per-feature histogram as the conditional matrix
    self.conditional_matrix_ = np.array([[0.8, 0.2], [0.2, 0.8]])
    return self

  def _compute_score(self, X):
    # pretend every test bag has the same observed frequencies for
    # simplicity; real subclasses derive this from X
    return getattr(self, "_next_test_frequencies", np.array([0.5, 0.5]))


@pytest.fixture
def binary_dataset():
  return make_classification(n_samples=300, n_classes=2, random_state=0)


class TestBaseCalibratedQuantifierParallel:
  def test_sequential_and_parallel_produce_identical_oof(self, binary_dataset):
    X, y = binary_dataset

    seq_quantifier = _DummyCalibratedQuantifier(classifier=LogisticRegression(max_iter=1000), cv=5, n_jobs=None)
    par_quantifier = _DummyCalibratedQuantifier(classifier=LogisticRegression(max_iter=1000), cv=5, n_jobs=2)

    seq_quantifier.fit(X, y)
    par_quantifier.fit(X, y)

    assert seq_quantifier.oof_accuracy_ == pytest.approx(par_quantifier.oof_accuracy_)

  def test_final_classifier_is_fitted_on_full_data(self, binary_dataset):
    X, y = binary_dataset
    quantifier = _DummyCalibratedQuantifier(classifier=LogisticRegression(max_iter=1000), cv=5)
    quantifier.fit(X, y)

    assert hasattr(quantifier.classifier_, "coef_")
    predictions = quantifier.predict(X)
    assert predictions.shape == (2,)
    assert predictions.sum() == pytest.approx(1.0)

  def test_predict_proba_oof_method_works(self, binary_dataset):
    X, y = binary_dataset
    quantifier = _DummyProbaQuantifier(classifier=LogisticRegression(max_iter=1000), cv=5)
    quantifier.fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)

  def test_fails_fast_when_classifier_lacks_predict_proba(self, binary_dataset):
    X, y = binary_dataset
    quantifier = _DummyProbaQuantifier(classifier=_NoProbaClassifier(), cv=5)
    with pytest.raises(TypeError, match="predict_proba"):
      quantifier.fit(X, y)

  def test_parallel_backend_threading_also_matches_sequential(self, binary_dataset):
    X, y = binary_dataset
    seq_quantifier = _DummyCalibratedQuantifier(classifier=LogisticRegression(max_iter=1000), cv=5, n_jobs=None)
    par_quantifier = _DummyCalibratedQuantifier(
      classifier=LogisticRegression(max_iter=1000), cv=5, n_jobs=2, parallel_backend="threading",
    )

    seq_quantifier.fit(X, y)
    par_quantifier.fit(X, y)

    assert seq_quantifier.oof_accuracy_ == pytest.approx(par_quantifier.oof_accuracy_)


class TestBaseMixtureQuantifierConvexCaching:
  def test_repeated_predict_calls_reuse_cached_problem(self):
    quantifier = _DummyMixtureQuantifier(distance_metric="L1")
    quantifier.fit(np.zeros((10, 1)), np.array([0] * 5 + [1] * 5))

    quantifier._next_test_frequencies = np.array([0.5, 0.5])
    p1 = quantifier.predict(np.zeros((4, 1)))
    cached_problem_id_1 = id(quantifier._cvx_problem_)

    quantifier._next_test_frequencies = np.array([0.8, 0.2])
    p2 = quantifier.predict(np.zeros((4, 1)))
    cached_problem_id_2 = id(quantifier._cvx_problem_)

    assert cached_problem_id_1 == cached_problem_id_2  # same object, not rebuilt
    assert p1.sum() == pytest.approx(1.0)
    assert p2.sum() == pytest.approx(1.0)
    assert not np.allclose(p1, p2)  # different test frequencies -> different solution

  def test_cache_rebuilds_when_distance_metric_changes(self):
    quantifier = _DummyMixtureQuantifier(distance_metric="L1")
    quantifier.fit(np.zeros((10, 1)), np.array([0] * 5 + [1] * 5))
    quantifier.predict(np.zeros((4, 1)))
    key_l1 = quantifier._cvx_cache_key_

    quantifier.distance_metric = "L2"
    quantifier.predict(np.zeros((4, 1)))
    key_l2 = quantifier._cvx_cache_key_

    assert key_l1 != key_l2

  def test_updated_conditional_matrix_is_reflected_in_solution(self):
    quantifier = _DummyMixtureQuantifier(distance_metric="L2")
    quantifier.fit(np.zeros((10, 1)), np.array([0] * 5 + [1] * 5))
    # asymmetric test frequencies: with a symmetric target like [0.5, 0.5],
    # both conditional matrices below solve to p=[0.5, 0.5] regardless of
    # their off-diagonal sharpness, which would make this test meaningless
    quantifier._next_test_frequencies = np.array([0.7, 0.3])

    p_before = quantifier.predict(np.zeros((4, 1)))

    # refit with a very different conditional matrix (same shape) and
    # confirm the cached problem picks up the new matrix values, not stale ones
    quantifier.conditional_matrix_ = np.array([[0.99, 0.01], [0.01, 0.99]])
    p_after = quantifier.predict(np.zeros((4, 1)))

    assert not np.allclose(p_before, p_after)

  def test_golden_section_fallback_still_works_without_convex_solver(self):
    quantifier = _DummyMixtureQuantifier(distance_metric="TS", use_convex_solver=False)
    quantifier.fit(np.zeros((10, 1)), np.array([0] * 5 + [1] * 5))
    quantifier._next_test_frequencies = np.array([0.5, 0.5])

    result = quantifier.predict(np.zeros((4, 1)))
    assert result.sum() == pytest.approx(1.0)
    assert np.all(result >= 0.0)


class TestVectorizedTopsoeDistance:
  def test_matches_manual_loop_formula(self):
    quantifier = _DummyMixtureQuantifier(distance_metric="TS")
    quantifier.conditional_matrix_ = np.array([[0.3, 0.1], [0.4, 0.2], [0.3, 0.7]])
    candidate = np.array([0.6, 0.4])
    test_frequencies = np.array([0.25, 0.35, 0.40])

    vectorized_result = quantifier._compute_distance(candidate, test_frequencies)

    projected = quantifier.conditional_matrix_.dot(candidate)
    manual = 0.0
    for i in range(len(projected)):
      if projected[i] != 0:
        manual += projected[i] * np.log(2 * projected[i] / (projected[i] + test_frequencies[i]))
      if test_frequencies[i] != 0:
        manual += test_frequencies[i] * np.log(2 * test_frequencies[i] / (projected[i] + test_frequencies[i]))

    assert vectorized_result == pytest.approx(manual)

  def test_handles_zero_frequencies_without_warning(self, recwarn):
    quantifier = _DummyMixtureQuantifier(distance_metric="TS")
    quantifier.conditional_matrix_ = np.array([[1.0, 0.0], [0.0, 1.0]])
    candidate = np.array([1.0, 0.0])
    test_frequencies = np.array([1.0, 0.0])

    result = quantifier._compute_distance(candidate, test_frequencies)
    assert np.isfinite(result)
    assert len(recwarn) == 0
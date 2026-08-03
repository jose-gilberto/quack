import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from quack.quantifiers._iterators import EM, CDE


@pytest.fixture
def binary_dataset():
  return make_classification(n_samples=400, n_classes=2, weights=[0.6, 0.4], random_state=0)


@pytest.fixture
def multiclass_dataset():
  return make_classification(n_samples=400, n_classes=3, n_informative=6, random_state=0)


class TestEM:
  def test_predict_sums_to_one(self, binary_dataset):
    X, y = binary_dataset
    quantifier = EM(LogisticRegression(max_iter=1000), cv=5).fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (2,)

  def test_multiclass_supported(self, multiclass_dataset):
    X, y = multiclass_dataset
    quantifier = EM(LogisticRegression(max_iter=1000), cv=5).fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.shape == (3,)
    assert prevalences.sum() == pytest.approx(1.0)

  def test_vectorized_quantify_matches_manual_loop_formula(self, binary_dataset):
    X, y = binary_dataset
    quantifier = EM(LogisticRegression(max_iter=1000), cv=5, max_iter=50).fit(X, y)

    predicted_probabilities = quantifier.classifier_.predict_proba(X)
    n_samples = predicted_probabilities.shape[0]

    # reference (manual per-sample loop) implementation, mirroring the pre-refactor logic
    prevalence_new = quantifier.train_prevalence_
    prevalence_old = np.ones(quantifier.train_prevalence_.shape)
    iteration_count = 0
    while (np.linalg.norm(prevalence_old - prevalence_new) > quantifier.epsilon) and iteration_count < quantifier.max_iter:
      prevalence_old = np.array(prevalence_new)
      posterior_matrix = np.array([
        (prevalence_old / quantifier.train_prevalence_) * predicted_probabilities[i]
        for i in range(n_samples)
      ])
      row_sums = np.sum(posterior_matrix, axis=1)[:, np.newaxis]
      posterior_matrix = posterior_matrix / row_sums
      prevalence_new = (1.0 / n_samples) * np.sum(posterior_matrix, axis=0)
      iteration_count += 1

    vectorized_result = quantifier.predict(X)
    np.testing.assert_allclose(vectorized_result, prevalence_new / prevalence_new.sum(), atol=1e-8)

  def test_accepts_n_jobs_and_parallel_backend(self, binary_dataset):
    X, y = binary_dataset
    quantifier = EM(LogisticRegression(max_iter=1000), cv=5, n_jobs=2, parallel_backend="threading").fit(X, y)
    assert quantifier.predict(X).sum() == pytest.approx(1.0)

  def test_converges_close_to_uniform_when_test_matches_train_distribution(self, binary_dataset):
    X, y = binary_dataset
    quantifier = EM(LogisticRegression(max_iter=1000), cv=5).fit(X, y)
    prevalences = quantifier.predict(X)
    # test bag == train bag here, so EM should stay close to train_prevalence_
    np.testing.assert_allclose(prevalences, quantifier.train_prevalence_, atol=0.1)


class TestCDE:
  def test_predict_sums_to_one(self, binary_dataset):
    X, y = binary_dataset
    quantifier = CDE(LogisticRegression(max_iter=1000), cv=5).fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (2,)

  def test_raises_value_error_for_multiclass(self, multiclass_dataset):
    X, y = multiclass_dataset
    with pytest.raises(ValueError, match="binary quantification"):
      CDE(LogisticRegression(max_iter=1000)).fit(X, y)

  def test_vectorized_quantify_matches_manual_apply_along_axis(self, binary_dataset):
    X, y = binary_dataset
    quantifier = CDE(LogisticRegression(max_iter=1000), cv=5, max_iter=50).fit(X, y)

    predicted_probabilities = quantifier.classifier_.predict_proba(X)

    # reference (manual apply_along_axis) implementation, mirroring the pre-refactor logic
    weights = np.ones(2)
    weights_old = np.zeros(2)
    positive_prevalence = 2.0
    iteration_count = 0
    while np.linalg.norm(weights - weights_old) > quantifier.epsilon and iteration_count <= quantifier.max_iter:
      threshold_labels = np.apply_along_axis(
        lambda prob: quantifier.classes_[1] if prob[1] > weights[0] / np.sum(weights) else quantifier.classes_[0],
        axis=1, arr=predicted_probabilities,
      )
      weights_old = np.copy(weights)
      positive_prevalence = np.mean(threshold_labels == quantifier.classes_[1])
      weights[0] = (1.0 - positive_prevalence) / quantifier.train_prevalence_[0]
      weights[1] = positive_prevalence / quantifier.train_prevalence_[1]
      iteration_count += 1

    manual_result = np.array([1.0 - positive_prevalence, positive_prevalence])
    vectorized_result = quantifier.predict(X)
    np.testing.assert_allclose(vectorized_result, manual_result, atol=1e-8)

  def test_warns_when_not_converged(self, binary_dataset):
    X, y = binary_dataset
    # force non-convergence with an unreachable tolerance and a tiny max_iter
    quantifier = CDE(LogisticRegression(max_iter=1000), cv=5, epsilon=1e-300, max_iter=1).fit(X, y)
    with pytest.warns(UserWarning, match="has not converged"):
      quantifier.predict(X)

  def test_accepts_n_jobs_and_parallel_backend(self, binary_dataset):
    X, y = binary_dataset
    quantifier = CDE(LogisticRegression(max_iter=1000), cv=5, n_jobs=2, parallel_backend="threading").fit(X, y)
    assert quantifier.predict(X).sum() == pytest.approx(1.0)
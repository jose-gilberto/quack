import numpy as np
import pytest
from sklearn.datasets import make_classification

from quack.quantifiers._probabilities import ED, _sum_pairwise_distances


@pytest.fixture
def binary_dataset():
  return make_classification(n_samples=200, n_classes=2, weights=[0.6, 0.4], random_state=0)


@pytest.fixture
def multiclass_dataset():
  return make_classification(n_samples=240, n_classes=3, n_informative=6, random_state=0)


class TestEDFitVectorization:
  def test_class_distances_matrix_is_symmetric(self, binary_dataset):
    X, y = binary_dataset
    quantifier = ED().fit(X, y)
    np.testing.assert_allclose(quantifier.class_distances_matrix_, quantifier.class_distances_matrix_.T)

  def test_quadratic_matrix_matches_manual_double_loop(self, multiclass_dataset):
    X, y = multiclass_dataset
    quantifier = ED().fit(X, y)

    A = quantifier.class_distances_matrix_
    last_idx = quantifier.n_classes_ - 1

    # reference (manual double loop) implementation, mirroring the pre-refactor logic
    manual = np.zeros((last_idx, last_idx))
    for i in range(last_idx):
      for j in range(i, last_idx):
        manual[i, j] = -A[i, j] + A[i, last_idx] + A[last_idx, j] - A[last_idx, last_idx]
        if j > i:
          manual[j, i] = manual[i, j]

    np.testing.assert_allclose(quantifier.quadratic_matrix_, manual)

  def test_no_quadratic_matrix_for_binary(self, binary_dataset):
    X, y = binary_dataset
    quantifier = ED().fit(X, y)
    assert quantifier.quadratic_matrix_ is None

  def test_raises_for_single_class(self):
    X = np.zeros((10, 2))
    y = np.zeros(10)
    with pytest.raises(ValueError, match="at least 2 distinct classes"):
      ED().fit(X, y)

  def test_parallel_fit_matches_sequential(self, multiclass_dataset):
    X, y = multiclass_dataset
    seq_quantifier = ED(n_jobs=None).fit(X, y)
    par_quantifier = ED(n_jobs=2, parallel_backend="threading").fit(X, y)

    np.testing.assert_allclose(seq_quantifier.class_distances_matrix_, par_quantifier.class_distances_matrix_)
    np.testing.assert_allclose(seq_quantifier.quadratic_matrix_, par_quantifier.quadratic_matrix_)


class TestEDPredict:
  def test_predict_sums_to_one_binary(self, binary_dataset):
    X, y = binary_dataset
    quantifier = ED().fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (2,)

  def test_predict_sums_to_one_multiclass(self, multiclass_dataset):
    X, y = multiclass_dataset
    quantifier = ED().fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (3,)

  def test_linear_vector_matches_manual_loop(self, multiclass_dataset):
    X, y = multiclass_dataset
    quantifier = ED().fit(X, y)

    A = quantifier.class_distances_matrix_
    last_idx = quantifier.n_classes_ - 1
    # arbitrary but deterministic stand-in for test_cross_distances, just
    # to validate the vectorized formula against the manual loop
    s = np.array([0.4, 0.6, 0.5])

    manual = np.zeros(last_idx)
    for i in range(last_idx):
      manual[i] = -s[i] + A[i, last_idx] + s[last_idx] - A[last_idx, last_idx]

    vectorized = -s[:last_idx] + A[:last_idx, last_idx] + s[last_idx] - A[last_idx, last_idx]
    np.testing.assert_allclose(vectorized, manual)

  def test_parallel_predict_matches_sequential(self, multiclass_dataset):
    X, y = multiclass_dataset
    seq_quantifier = ED(n_jobs=None).fit(X, y)
    par_quantifier = ED(n_jobs=2, parallel_backend="threading").fit(X, y)

    np.testing.assert_allclose(seq_quantifier.predict(X), par_quantifier.predict(X), atol=1e-6)

  def test_binary_clips_when_solution_exceeds_upper_bound(self, monkeypatch, binary_dataset):
    X, y = binary_dataset
    quantifier = ED().fit(X, y)
    # force a distance profile where the analytical p solves to > 1.0
    quantifier.class_distances_matrix_ = np.array([[0.0, 1.0], [1.0, 0.0]])
    call_totals = iter([0.0, 2.0 * X.shape[0] * len(quantifier.train_class_samples_[1])])

    def _fake_sum(samples_i, samples_j):
      return next(call_totals)

    monkeypatch.setattr("quack.quantifiers._probabilities._sum_pairwise_distances", _fake_sum)
    result = quantifier.predict(X)
    np.testing.assert_array_equal(result, [1.0, 0.0])

  def test_binary_clips_when_solution_is_negative(self, monkeypatch, binary_dataset):
    X, y = binary_dataset
    quantifier = ED().fit(X, y)
    quantifier.class_distances_matrix_ = np.array([[0.0, 1.0], [1.0, 0.0]])
    class_0_size = quantifier.train_class_samples_[0].shape[0]
    call_totals = iter([2.0 * X.shape[0] * class_0_size, 0.0])

    def _fake_sum(samples_i, samples_j):
      return next(call_totals)

    monkeypatch.setattr("quack.quantifiers._probabilities._sum_pairwise_distances", _fake_sum)
    result = quantifier.predict(X)
    np.testing.assert_array_equal(result, [0.0, 1.0])


class TestSumPairwiseDistances:
  def test_matches_naive_pairwise_sum(self):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(10, 3))
    b = rng.normal(size=(8, 3))

    from sklearn.metrics import pairwise_distances
    expected = pairwise_distances(a, b).sum()

    assert _sum_pairwise_distances(a, b) == pytest.approx(expected)
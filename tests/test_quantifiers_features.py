import numpy as np
import pytest
from sklearn.datasets import make_classification

from quack.quantifiers._features import HDx, ReadMe, _RawSubspaceMixture


@pytest.fixture
def categorical_dataset():
  rng = np.random.default_rng(0)
  X = rng.integers(0, 5, size=(300, 4)).astype(float)
  y = (X[:, 0] + X[:, 1] > 4).astype(int)
  return X, y


class TestHDxFitVectorization:
  def test_conditional_matrix_matches_manual_triple_loop(self, categorical_dataset):
    X, y = categorical_dataset
    quantifier = HDx().fit(X, y)

    # reference (manual triple loop) implementation, mirroring the pre-refactor logic
    classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    n_features = X.shape[1]
    feature_spaces = [np.unique(X[:, j]) for j in range(n_features)]

    manual_blocks = []
    for j in range(n_features):
      unique_values = feature_spaces[j]
      crosstab_counts = np.zeros((len(unique_values), n_classes))
      for class_idx, class_label in enumerate(classes):
        X_class_subset = X[y == class_label, j]
        for val_idx, val in enumerate(unique_values):
          crosstab_counts[val_idx, class_idx] = np.sum(X_class_subset == val)
      manual_blocks.append(crosstab_counts / class_counts)
    manual_matrix = np.vstack(manual_blocks)

    np.testing.assert_allclose(quantifier.conditional_matrix_, manual_matrix)

  def test_predict_sums_to_one(self, categorical_dataset):
    X, y = categorical_dataset
    quantifier = HDx().fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (2,)

  def test_raises_for_single_class(self):
    X = np.zeros((10, 2))
    y = np.zeros(10)
    with pytest.raises(ValueError, match="at least 2 distinct classes"):
      HDx().fit(X, y)

  def test_compute_score_matches_manual_loop(self, categorical_dataset):
    X, y = categorical_dataset
    quantifier = HDx().fit(X, y)

    n_samples = X.shape[0]
    manual_frequencies = []
    for j in range(X.shape[1]):
      feature_counts = np.array([
        np.count_nonzero(X[:, j] == val) for val in quantifier.feature_spaces_[j]
      ])
      manual_frequencies.append((1.0 / n_samples) * feature_counts)
    manual_result = np.hstack(manual_frequencies)

    np.testing.assert_allclose(quantifier._compute_score(X), manual_result)

  def test_compute_score_ignores_unseen_test_values(self, categorical_dataset):
    X, y = categorical_dataset
    quantifier = HDx().fit(X, y)

    X_test = X.copy()
    X_test[0, 0] = 999.0  # value never seen during training

    result = quantifier._compute_score(X_test)
    # the unseen value contributes 0 to every bin of feature 0 (matching
    # the original np.count_nonzero(X[:, j] == val) semantics: it's
    # simply never counted since it doesn't equal any trained unique value)
    n_unique_feature_0 = len(quantifier.feature_spaces_[0])
    assert result[:n_unique_feature_0].sum() == pytest.approx((X.shape[0] - 1) / X.shape[0])


class TestRawSubspaceMixtureFitVectorization:
  def test_fit_subspace_matches_manual_binary_search_loop(self):
    rng = np.random.default_rng(1)
    X = rng.integers(0, 3, size=(80, 2)).astype(float)
    y = rng.integers(0, 2, size=80)
    classes, class_counts = np.unique(y, return_counts=True)

    vectorized = _RawSubspaceMixture(distance_metric="L2").fit_subspace(X, y, classes, class_counts)

    # reference (manual binary search loop) implementation, mirroring the pre-refactor logic
    unique_rows = np.unique(X, axis=0)
    manual_matrix = np.zeros((unique_rows.shape[0], len(classes)))
    class_to_index = {c: idx for idx, c in enumerate(classes)}
    for i in range(len(y)):
      row_index = _RawSubspaceMixture._binary_search_row(X[i, :], unique_rows)
      manual_matrix[row_index, class_to_index[y[i]]] += 1
    manual_matrix = manual_matrix / class_counts

    np.testing.assert_array_equal(vectorized.unique_rows_, unique_rows)
    np.testing.assert_allclose(vectorized.conditional_matrix_, manual_matrix)

  def test_predict_after_fit_subspace_sums_to_one(self):
    rng = np.random.default_rng(2)
    X = rng.integers(0, 3, size=(60, 2)).astype(float)
    y = rng.integers(0, 2, size=60)
    classes, class_counts = np.unique(y, return_counts=True)

    quantifier = _RawSubspaceMixture(distance_metric="L2").fit_subspace(X, y, classes, class_counts)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)


class TestReadMe:
  def test_predict_sums_to_one(self, categorical_dataset):
    X, y = categorical_dataset
    quantifier = ReadMe(n_subsets=5, n_features=2, random_state=0).fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (2,)

  def test_reproducible_with_same_random_state(self, categorical_dataset):
    X, y = categorical_dataset
    q_a = ReadMe(n_subsets=5, n_features=2, random_state=42).fit(X, y)
    q_b = ReadMe(n_subsets=5, n_features=2, random_state=42).fit(X, y)

    for subset_a, subset_b in zip(q_a.feature_subsets_, q_b.feature_subsets_):
      np.testing.assert_array_equal(subset_a, subset_b)
    np.testing.assert_allclose(q_a.predict(X), q_b.predict(X))

  def test_different_random_states_produce_different_subsets(self, categorical_dataset):
    X, y = categorical_dataset
    q_a = ReadMe(n_subsets=5, n_features=2, random_state=0).fit(X, y)
    q_b = ReadMe(n_subsets=5, n_features=2, random_state=1).fit(X, y)

    all_equal = all(
      np.array_equal(a, b) for a, b in zip(q_a.feature_subsets_, q_b.feature_subsets_)
    )
    assert not all_equal

  def test_accepts_n_jobs_and_parallel_backend(self, categorical_dataset):
    X, y = categorical_dataset
    quantifier = ReadMe(
      n_subsets=5, n_features=2, random_state=0, n_jobs=2, parallel_backend="threading",
    ).fit(X, y)
    assert quantifier.predict(X).sum() == pytest.approx(1.0)

  def test_auto_n_features_low_dimensional(self, categorical_dataset):
    X, y = categorical_dataset  # 4 features -> max(int(4/5), 2) == 2
    quantifier = ReadMe(n_subsets=3, random_state=0).fit(X, y)
    assert quantifier.n_features == 2
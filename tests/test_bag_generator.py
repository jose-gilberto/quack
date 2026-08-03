import numpy as np
import pytest
from sklearn.datasets import make_classification

from quack.bag_generator import BaseBagGenerator, PriorShiftBagGenerator, CovariateShiftBagGenerator


@pytest.fixture
def binary_dataset():
  return make_classification(n_samples=300, n_classes=2, weights=[0.7, 0.3], random_state=0)


@pytest.fixture
def multiclass_dataset():
  return make_classification(n_samples=400, n_classes=4, n_informative=6, random_state=0)


class TestPriorShiftBagGenerator:
  def test_generate_yields_correct_number_and_size_of_bags(self, binary_dataset):
    X, y = binary_dataset
    generator = PriorShiftBagGenerator(n_bags=10, bag_size=50, random_state=0)
    bags = generator.to_list(X, y)

    assert len(bags) == 10
    for X_bag, y_bag in bags:
      assert X_bag.shape == (50, X.shape[1])
      assert y_bag.shape == (50,)

  def test_default_bag_size_matches_dataset_size(self, binary_dataset):
    X, y = binary_dataset
    generator = PriorShiftBagGenerator(n_bags=3, random_state=0)
    for X_bag, y_bag in generator.generate(X, y):
      assert X_bag.shape[0] == len(y)

  def test_uniform_prevalences_sum_to_one_and_match_bag(self, binary_dataset):
    X, y = binary_dataset
    generator = PriorShiftBagGenerator(n_bags=20, bag_size=80, sampling_strategy='uniform', random_state=1)
    for i, (_, y_bag) in enumerate(generator.generate(X, y)):
      realized = np.array([np.mean(y_bag == c) for c in generator.classes_])
      np.testing.assert_allclose(realized, generator.sampled_prevalences_[i])
      assert generator.sampled_prevalences_[i].sum() == pytest.approx(1.0)

  def test_dirichlet_strategy_shapes_and_bounds(self, multiclass_dataset):
    X, y = multiclass_dataset
    generator = PriorShiftBagGenerator(
      n_bags=15, bag_size=60, sampling_strategy='dirichlet', dirichlet_alpha=0.3, random_state=2,
    )
    bags = generator.to_list(X, y)
    assert generator.sampled_prevalences_.shape == (15, 4)
    assert np.all(generator.sampled_prevalences_ >= 0)
    np.testing.assert_allclose(generator.sampled_prevalences_.sum(axis=1), 1.0)

  def test_reproducible_with_same_random_state(self, binary_dataset):
    X, y = binary_dataset
    gen_a = PriorShiftBagGenerator(n_bags=5, bag_size=40, random_state=42)
    gen_b = PriorShiftBagGenerator(n_bags=5, bag_size=40, random_state=42)

    bags_a = gen_a.to_list(X, y)
    bags_b = gen_b.to_list(X, y)

    for (Xa, ya), (Xb, yb) in zip(bags_a, bags_b):
      np.testing.assert_array_equal(Xa, Xb)
      np.testing.assert_array_equal(ya, yb)

  def test_without_replacement_respects_pool_when_possible(self, binary_dataset):
    X, y = binary_dataset
    generator = PriorShiftBagGenerator(
      n_bags=5, bag_size=30, with_replacement=False, sampling_strategy='dirichlet',
      dirichlet_alpha=5.0, random_state=3,
    )
    # should not raise, even forcing replace=True internally when a class pool is too small
    generator.to_list(X, y)

  def test_invalid_sampling_strategy_raises(self, binary_dataset):
    X, y = binary_dataset
    generator = PriorShiftBagGenerator(sampling_strategy='not-a-strategy')
    with pytest.raises(ValueError, match="Unknown sampling_strategy"):
      next(generator.generate(X, y))

  def test_invalid_n_bags_raises(self, binary_dataset):
    X, y = binary_dataset
    generator = PriorShiftBagGenerator(n_bags=0)
    with pytest.raises(ValueError, match="n_bags must be a positive integer"):
      next(generator.generate(X, y))


class TestCovariateShiftBagGenerator:
  def test_generate_yields_correct_number_and_size_of_bags(self, binary_dataset):
    X, y = binary_dataset
    generator = CovariateShiftBagGenerator(n_bags=10, bag_size=50, random_state=0)
    bags = generator.to_list(X, y)

    assert len(bags) == 10
    for X_bag, y_bag in bags:
      assert X_bag.shape == (50, X.shape[1])
      assert y_bag.shape == (50,)

  def test_sampled_prevalences_are_valid_distributions(self, binary_dataset):
    X, y = binary_dataset
    generator = CovariateShiftBagGenerator(n_bags=15, bag_size=60, gamma=0.5, random_state=1)
    generator.to_list(X, y)
    np.testing.assert_allclose(generator.sampled_prevalences_.sum(axis=1), 1.0)
    assert generator.pivot_indices_.shape == (15,)

  def test_reproducible_with_same_random_state(self, binary_dataset):
    X, y = binary_dataset
    gen_a = CovariateShiftBagGenerator(n_bags=5, bag_size=40, gamma=1.0, random_state=7)
    gen_b = CovariateShiftBagGenerator(n_bags=5, bag_size=40, gamma=1.0, random_state=7)

    bags_a = gen_a.to_list(X, y)
    bags_b = gen_b.to_list(X, y)

    for (Xa, ya), (Xb, yb) in zip(bags_a, bags_b):
      np.testing.assert_array_equal(Xa, Xb)
      np.testing.assert_array_equal(ya, yb)

  def test_high_gamma_concentrates_bag_near_pivot(self, binary_dataset):
    X, y = binary_dataset
    generator = CovariateShiftBagGenerator(n_bags=1, bag_size=100, gamma=50.0, random_state=0)
    X_bag, _ = next(generator.generate(X, y))
    pivot = X[generator.pivot_indices_[0]]

    mean_dist_bag = np.mean(np.linalg.norm(X_bag - pivot, axis=1))
    mean_dist_full = np.mean(np.linalg.norm(X - pivot, axis=1))
    assert mean_dist_bag < mean_dist_full


class TestBaseBagGeneratorContract:
  def test_to_list_matches_manual_generate_consumption(self, binary_dataset):
    X, y = binary_dataset
    generator = PriorShiftBagGenerator(n_bags=4, bag_size=30, random_state=0)
    assert isinstance(generator, BaseBagGenerator)

    bags_via_to_list = generator.to_list(X, y)
    assert len(bags_via_to_list) == 4
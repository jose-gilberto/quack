import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from quack.ensembles import EoQ
from quack.quantifiers import CC
from quack.bag_generator import PriorShiftBagGenerator, CovariateShiftBagGenerator


@pytest.fixture
def binary_dataset():
  return make_classification(n_samples=600, n_classes=2, weights=[0.6, 0.4], random_state=0)


@pytest.fixture
def multiclass_dataset():
  return make_classification(n_samples=600, n_classes=3, n_informative=6, random_state=0)


def _make_base_quantifier():
  return CC(LogisticRegression(max_iter=1000))


class TestEoQValidation:
  def test_raises_for_unknown_policy(self, binary_dataset):
    X, y = binary_dataset
    with pytest.raises(ValueError, match="Unknown selection_policy"):
      EoQ(_make_base_quantifier(), selection_policy="not-a-policy").fit(X, y)

  def test_raises_when_red_size_missing_for_ptr(self, binary_dataset):
    X, y = binary_dataset
    with pytest.raises(ValueError, match="requires red_size"):
      EoQ(_make_base_quantifier(), selection_policy="ptr").fit(X, y)

  def test_raises_when_red_size_missing_for_performance(self, binary_dataset):
    X, y = binary_dataset
    with pytest.raises(ValueError, match="requires red_size"):
      EoQ(_make_base_quantifier(), selection_policy="performance").fit(X, y)

  def test_raises_when_red_size_exceeds_n_estimators(self, binary_dataset):
    X, y = binary_dataset
    with pytest.raises(ValueError, match="red_size must satisfy"):
      EoQ(_make_base_quantifier(), n_estimators=10, selection_policy="ptr", red_size=20).fit(X, y)

  def test_raises_for_non_positive_n_estimators(self, binary_dataset):
    X, y = binary_dataset
    with pytest.raises(ValueError, match="n_estimators must be a positive integer"):
      EoQ(_make_base_quantifier(), n_estimators=0).fit(X, y)


class TestEoQAveragePolicy:
  def test_predict_sums_to_one(self, binary_dataset):
    X, y = binary_dataset
    ensemble = EoQ(_make_base_quantifier(), n_estimators=10, selection_policy="average", random_state=0)
    ensemble.fit(X, y)
    prevalences = ensemble.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (2,)

  def test_multiclass_supported(self, multiclass_dataset):
    X, y = multiclass_dataset
    ensemble = EoQ(_make_base_quantifier(), n_estimators=10, selection_policy="average", random_state=0)
    ensemble.fit(X, y)
    prevalences = ensemble.predict(X)
    assert prevalences.shape == (3,)
    assert prevalences.sum() == pytest.approx(1.0)

  def test_estimators_and_train_prevalences_have_expected_shapes(self, binary_dataset):
    X, y = binary_dataset
    ensemble = EoQ(_make_base_quantifier(), n_estimators=8, random_state=0)
    ensemble.fit(X, y)
    assert len(ensemble.estimators_) == 8
    assert ensemble.member_train_prevalences_.shape == (8, 2)
    assert ensemble.oob_scores_ is None
    np.testing.assert_array_equal(ensemble.selected_indices_, np.arange(8))

  def test_reproducible_with_same_random_state(self, binary_dataset):
    X, y = binary_dataset
    ens_a = EoQ(_make_base_quantifier(), n_estimators=8, random_state=42).fit(X, y)
    ens_b = EoQ(_make_base_quantifier(), n_estimators=8, random_state=42).fit(X, y)
    np.testing.assert_allclose(ens_a.member_train_prevalences_, ens_b.member_train_prevalences_)
    np.testing.assert_allclose(ens_a.predict(X), ens_b.predict(X))

  def test_parallel_matches_sequential(self, binary_dataset):
    X, y = binary_dataset
    seq_ensemble = EoQ(_make_base_quantifier(), n_estimators=8, random_state=0, n_jobs=None).fit(X, y)
    par_ensemble = EoQ(
      _make_base_quantifier(), n_estimators=8, random_state=0, n_jobs=2, parallel_backend="threading",
    ).fit(X, y)
    np.testing.assert_allclose(seq_ensemble.predict(X), par_ensemble.predict(X), atol=1e-8)

  def test_accepts_custom_bag_generator(self, binary_dataset):
    X, y = binary_dataset
    generator = CovariateShiftBagGenerator(gamma=0.5)
    ensemble = EoQ(_make_base_quantifier(), n_estimators=8, bag_generator=generator, random_state=0)
    ensemble.fit(X, y)
    prevalences = ensemble.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)

  def test_custom_bag_size_is_preserved(self, binary_dataset):
    X, y = binary_dataset
    generator = PriorShiftBagGenerator(bag_size=50, sampling_strategy="dirichlet", dirichlet_alpha=0.5)
    ensemble = EoQ(_make_base_quantifier(), n_estimators=5, bag_generator=generator, random_state=0)
    ensemble.fit(X, y)
    assert len(ensemble.estimators_) == 5  # bag_size=50 didn't break n_bags override


class TestEoQPtrPolicy:
  def test_predict_sums_to_one(self, binary_dataset):
    X, y = binary_dataset
    ensemble = EoQ(
      _make_base_quantifier(), n_estimators=20, selection_policy="ptr", red_size=8, random_state=0,
    )
    ensemble.fit(X, y)
    prevalences = ensemble.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)

  def test_selection_changes_across_different_test_bags(self, binary_dataset):
    X, y = binary_dataset
    ensemble = EoQ(
      _make_base_quantifier(), n_estimators=20, selection_policy="ptr", red_size=5, random_state=0,
    )
    ensemble.fit(X, y)

    # two very differently-shifted test bags should generally not select
    # the exact same subset of members, since 'ptr' picks members whose
    # own training prevalence is closest to each bag's estimated prevalence
    rng = np.random.default_rng(1)
    pos_heavy_idx = rng.choice(np.flatnonzero(y == 1), size=100, replace=True)
    neg_heavy_idx = rng.choice(np.flatnonzero(y == 0), size=100, replace=True)

    p_pos_heavy = ensemble.predict(X[pos_heavy_idx])
    p_neg_heavy = ensemble.predict(X[neg_heavy_idx])
    assert not np.allclose(p_pos_heavy, p_neg_heavy, atol=1e-3)


class TestEoQPerformancePolicy:
  def test_predict_sums_to_one(self, binary_dataset):
    X, y = binary_dataset
    ensemble = EoQ(
      _make_base_quantifier(), n_estimators=15, selection_policy="performance",
      red_size=5, metric="ae", val_split=0.4, random_state=0,
    )
    ensemble.fit(X, y)
    prevalences = ensemble.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)

  def test_oob_scores_and_selected_indices_have_expected_shapes(self, binary_dataset):
    X, y = binary_dataset
    ensemble = EoQ(
      _make_base_quantifier(), n_estimators=15, selection_policy="performance", red_size=5, random_state=0,
    )
    ensemble.fit(X, y)
    assert ensemble.oob_scores_.shape == (15,)
    assert ensemble.selected_indices_.shape == (5,)
    # selected indices must be the ones with the lowest (best) AE score
    expected = np.sort(np.argsort(ensemble.oob_scores_)[:5])
    np.testing.assert_array_equal(ensemble.selected_indices_, expected)

  def test_accepts_metric_instance_directly(self, binary_dataset):
    from quack.metrics import KullbackLeiblerDivergence
    X, y = binary_dataset
    ensemble = EoQ(
      _make_base_quantifier(), n_estimators=10, selection_policy="performance",
      red_size=4, metric=KullbackLeiblerDivergence(), random_state=0,
    )
    ensemble.fit(X, y)
    assert ensemble.predict(X).sum() == pytest.approx(1.0)

  def test_custom_n_val_samples(self, binary_dataset):
    X, y = binary_dataset
    ensemble = EoQ(
      _make_base_quantifier(), n_estimators=10, selection_policy="performance",
      red_size=4, n_val_samples=3, random_state=0,
    )
    ensemble.fit(X, y)
    assert ensemble.oob_scores_.shape == (10,)  # one score per member, averaged over 3 val bags


class TestEoQClassCompleteness:
  def test_every_member_sees_all_classes_multiclass(self, multiclass_dataset):
    X, y = multiclass_dataset
    ensemble = EoQ(_make_base_quantifier(), n_estimators=15, selection_policy="average", random_state=0)
    ensemble.fit(X, y)

    for member in ensemble.estimators_:
      assert len(member.classes_) == 3

    prevalences = ensemble.predict(X)
    assert prevalences.shape == (3,)
    assert prevalences.sum() == pytest.approx(1.0)

  def test_extreme_dirichlet_shift_still_produces_full_class_bags(self, multiclass_dataset):
    # regression test: dirichlet_alpha=0.05 concentrates ~99.9% of the mass
    # on a single class, making "just retry and hope" unreliable (or, for
    # sufficiently extreme configurations, impossible); the deterministic
    # top-up must succeed here every time regardless
    X, y = multiclass_dataset
    generator = PriorShiftBagGenerator(sampling_strategy="dirichlet", dirichlet_alpha=0.05, bag_size=60)
    ensemble = EoQ(
      _make_base_quantifier(), n_estimators=15, bag_generator=generator,
      selection_policy="average", random_state=0,
    )
    ensemble.fit(X, y)
    for member in ensemble.estimators_:
      assert len(member.classes_) == 3
    assert ensemble.predict(X).sum() == pytest.approx(1.0)

  def test_covariate_shift_generator_does_not_crash_on_degenerate_pivot(self, binary_dataset):
    X, y = binary_dataset
    generator = CovariateShiftBagGenerator(gamma=0.5)
    ensemble = EoQ(_make_base_quantifier(), n_estimators=8, bag_generator=generator, random_state=0)
    ensemble.fit(X, y)
    for member in ensemble.estimators_:
      assert len(member.classes_) == 2
    assert ensemble.predict(X).sum() == pytest.approx(1.0)

  def test_raises_value_error_when_bag_size_smaller_than_n_classes(self, multiclass_dataset):
    X, y = multiclass_dataset
    generator = PriorShiftBagGenerator(bag_size=2)  # can never fit 3 classes into 2 slots
    ensemble = EoQ(_make_base_quantifier(), n_estimators=3, bag_generator=generator, random_state=0)
    with pytest.raises(ValueError, match="smaller than the number of required classes"):
      ensemble.fit(X, y)

  def test_member_train_prevalences_reflect_top_up(self, multiclass_dataset):
    X, y = multiclass_dataset
    generator = PriorShiftBagGenerator(sampling_strategy="dirichlet", dirichlet_alpha=0.05, bag_size=60)
    ensemble = EoQ(
      _make_base_quantifier(), n_estimators=15, bag_generator=generator,
      selection_policy="average", random_state=0,
    )
    ensemble.fit(X, y)
    # every reported prevalence must sum to 1 and have strictly positive
    # mass on all 3 classes (that's exactly what top-up guarantees)
    np.testing.assert_allclose(ensemble.member_train_prevalences_.sum(axis=1), 1.0)
    assert np.all(ensemble.member_train_prevalences_ > 0)


class TestTopUpMissingClasses:
  def test_returns_bag_unchanged_when_all_classes_present(self):
    from quack.ensembles._eoq import _top_up_missing_classes
    rng = np.random.default_rng(0)
    X_bag = np.zeros((4, 2))
    y_bag = np.array([0, 0, 1, 1])
    class_pools = {0: np.array([0, 1]), 1: np.array([2, 3])}
    X_result, y_result = _top_up_missing_classes(X_bag, y_bag, X_bag, class_pools, np.array([0, 1]), rng)
    np.testing.assert_array_equal(y_result, y_bag)

  def test_injects_missing_class_replacing_largest_class(self):
    from quack.ensembles._eoq import _top_up_missing_classes
    rng = np.random.default_rng(0)
    X_pool = np.array([[0.0], [1.0], [2.0], [3.0], [9.0]])
    y_bag = np.array([0, 0, 0, 0])  # class 1 entirely missing
    class_pools = {0: np.array([0, 1, 2, 3]), 1: np.array([4])}
    X_bag = X_pool[:4].copy()

    X_result, y_result = _top_up_missing_classes(X_bag, y_bag, X_pool, class_pools, np.array([0, 1]), rng)

    assert np.count_nonzero(y_result == 1) == 1
    assert np.count_nonzero(y_result == 0) == 3
    # the injected row must come from class 1's pool (value 9.0)
    injected_pos = np.flatnonzero(y_result == 1)[0]
    assert X_result[injected_pos, 0] == pytest.approx(9.0)

  def test_handles_multiple_missing_classes(self):
    from quack.ensembles._eoq import _top_up_missing_classes
    rng = np.random.default_rng(0)
    X_pool = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y_bag = np.array([0, 0, 0, 0])  # classes 1 and 2 entirely missing
    class_pools = {0: np.array([0, 1, 2, 3]), 1: np.array([4]), 2: np.array([5])}
    X_bag = X_pool[:4].copy()

    _, y_result = _top_up_missing_classes(X_bag, y_bag, X_pool, class_pools, np.array([0, 1, 2]), rng)

    assert set(np.unique(y_result).tolist()) == {0, 1, 2}
    assert len(y_result) == 4  # bag size preserved


class TestAlignPrediction:
  def test_returns_prediction_unchanged_when_classes_match(self):
    from quack.ensembles._eoq import _align_prediction
    full_classes = np.array([0, 1, 2])
    prediction = np.array([0.2, 0.3, 0.5])
    result = _align_prediction(prediction, np.array([0, 1, 2]), full_classes)
    np.testing.assert_array_equal(result, prediction)

  def test_scatters_prediction_into_full_length_vector(self):
    from quack.ensembles._eoq import _align_prediction
    full_classes = np.array([0, 1, 2])
    member_classes = np.array([0, 2])  # class 1 missing
    prediction = np.array([0.4, 0.6])
    result = _align_prediction(prediction, member_classes, full_classes)
    np.testing.assert_allclose(result, [0.4, 0.0, 0.6])

  def test_returns_prediction_unchanged_when_member_classes_is_none(self):
    from quack.ensembles._eoq import _align_prediction
    full_classes = np.array([0, 1])
    prediction = np.array([0.5, 0.5])
    result = _align_prediction(prediction, None, full_classes)
    np.testing.assert_array_equal(result, prediction)
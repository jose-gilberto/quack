import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from quack.ensembles import MCSQ, MCMQ
from quack.ensembles._mcq import _merge_predictions
from quack.quantifiers import CC, PCC, ACC


def _default_classifiers():
  return [LogisticRegression(max_iter=1000), RandomForestClassifier(n_estimators=30, random_state=0), GaussianNB()]


@pytest.fixture
def binary_dataset():
  return make_classification(n_samples=600, n_classes=2, weights=[0.6, 0.4], random_state=0)


@pytest.fixture
def multiclass_dataset():
  return make_classification(n_samples=600, n_classes=3, n_informative=6, random_state=0)


class TestMergePredictions:
  def test_mean_of_unit_sum_vectors_needs_no_renormalization(self):
    stack = np.array([[0.3, 0.7], [0.5, 0.5]])
    result = _merge_predictions(stack, "mean", n_classes=2)
    np.testing.assert_allclose(result, [0.4, 0.6])

  def test_median_is_renormalized(self):
    stack = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
    result = _merge_predictions(stack, "median", n_classes=2)
    assert result.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(result, [0.5, 0.5])

  def test_raises_for_unknown_merge_fun(self):
    with pytest.raises(ValueError, match="Unknown fusion"):
      _merge_predictions(np.zeros((2, 2)), "not-a-merge", n_classes=2)


class TestMCSQ:
  @pytest.mark.parametrize("fusion", ["median", "mean"])
  def test_predict_sums_to_one(self, binary_dataset, fusion):
    X, y = binary_dataset
    ensemble = MCSQ(classifiers=_default_classifiers(), quantifier=CC(), fusion=fusion)
    ensemble.fit(X, y)
    prevalences = ensemble.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (2,)

  def test_multiclass_supported(self, multiclass_dataset):
    X, y = multiclass_dataset
    ensemble = MCSQ(classifiers=_default_classifiers(), quantifier=PCC())
    ensemble.fit(X, y)
    prevalences = ensemble.predict(X)
    assert prevalences.shape == (3,)
    assert prevalences.sum() == pytest.approx(1.0)

  def test_one_member_per_classifier(self, binary_dataset):
    X, y = binary_dataset
    classifiers = _default_classifiers()
    ensemble = MCSQ(classifiers=classifiers, quantifier=ACC(cv=5))
    ensemble.fit(X, y)
    assert len(ensemble.quantifiers_) == len(classifiers)

  def test_members_get_independent_classifier_clones(self, binary_dataset):
    # regression test: reusing the same classifier instance across
    # `classifiers` must not leak fitted state between members
    X, y = binary_dataset
    shared_clf = LogisticRegression(max_iter=1000)
    ensemble = MCSQ(classifiers=[shared_clf, shared_clf], quantifier=CC())
    ensemble.fit(X, y)
    assert ensemble.quantifiers_[0].classifier_ is not ensemble.quantifiers_[1].classifier_

  def test_raises_for_empty_classifiers(self, binary_dataset):
    X, y = binary_dataset
    with pytest.raises(ValueError, match="non-empty list"):
      MCSQ(classifiers=[], quantifier=CC()).fit(X, y)

  def test_raises_for_unknown_merge_fun(self, binary_dataset):
    X, y = binary_dataset
    with pytest.raises(ValueError, match="Unknown fusion"):
      MCSQ(classifiers=_default_classifiers(), quantifier=CC(), fusion="not-a-merge").fit(X, y)

  def test_raises_when_quantifier_lacks_classifier_param(self, binary_dataset):
    from quack.quantifiers import HDx
    X, y = binary_dataset
    with pytest.raises(ValueError, match="classifier"):
      MCSQ(classifiers=_default_classifiers(), quantifier=HDx()).fit(X, y)

  def test_accepts_n_jobs_and_parallel_backend(self, binary_dataset):
    X, y = binary_dataset
    ensemble = MCSQ(
      classifiers=_default_classifiers(), quantifier=CC(), n_jobs=2, parallel_backend="threading",
    )
    ensemble.fit(X, y)
    assert ensemble.predict(X).sum() == pytest.approx(1.0)

  def test_reproducible_across_repeated_fits(self, binary_dataset):
    X, y = binary_dataset
    ens_a = MCSQ(classifiers=_default_classifiers(), quantifier=CC()).fit(X, y)
    ens_b = MCSQ(classifiers=_default_classifiers(), quantifier=CC()).fit(X, y)
    np.testing.assert_allclose(ens_a.predict(X), ens_b.predict(X))


class TestMCMQ:
  @pytest.mark.parametrize("fusion", ["median", "mean"])
  def test_predict_sums_to_one(self, binary_dataset, fusion):
    X, y = binary_dataset
    ensemble = MCMQ(classifiers=_default_classifiers(), quantifiers=[CC(), PCC(), ACC(cv=5)], fusion=fusion)
    ensemble.fit(X, y)
    prevalences = ensemble.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (2,)

  def test_multiclass_supported(self, multiclass_dataset):
    X, y = multiclass_dataset
    ensemble = MCMQ(classifiers=_default_classifiers(), quantifiers=[CC(), PCC()])
    ensemble.fit(X, y)
    prevalences = ensemble.predict(X)
    assert prevalences.shape == (3,)
    assert prevalences.sum() == pytest.approx(1.0)

  def test_member_groups_shape(self, binary_dataset):
    X, y = binary_dataset
    classifiers = _default_classifiers()
    quantifiers = [CC(), PCC()]
    ensemble = MCMQ(classifiers=classifiers, quantifiers=quantifiers)
    ensemble.fit(X, y)
    assert len(ensemble.member_groups_) == len(classifiers)
    for group in ensemble.member_groups_:
      assert len(group) == len(quantifiers)

  def test_raises_for_empty_quantifiers(self, binary_dataset):
    X, y = binary_dataset
    with pytest.raises(ValueError, match="non-empty list"):
      MCMQ(classifiers=_default_classifiers(), quantifiers=[]).fit(X, y)

  def test_raises_when_a_quantifier_lacks_classifier_param(self, binary_dataset):
    from quack.quantifiers import HDx
    X, y = binary_dataset
    with pytest.raises(ValueError, match="classifier"):
      MCMQ(classifiers=_default_classifiers(), quantifiers=[CC(), HDx()]).fit(X, y)

  def test_accepts_n_jobs_and_parallel_backend(self, binary_dataset):
    X, y = binary_dataset
    ensemble = MCMQ(
      classifiers=_default_classifiers(), quantifiers=[CC(), PCC()], n_jobs=2, parallel_backend="threading",
    )
    ensemble.fit(X, y)
    assert ensemble.predict(X).sum() == pytest.approx(1.0)

  def test_reproducible_across_repeated_fits(self, binary_dataset):
    X, y = binary_dataset
    ens_a = MCMQ(classifiers=_default_classifiers(), quantifiers=[CC(), PCC()]).fit(X, y)
    ens_b = MCMQ(classifiers=_default_classifiers(), quantifiers=[CC(), PCC()]).fit(X, y)
    np.testing.assert_allclose(ens_a.predict(X), ens_b.predict(X))
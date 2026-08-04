import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator, ClassifierMixin

from quack.ensembles import FMCSQ, FMCMQ
from quack.ensembles._fmc import (
  _IdentityScorer, _ALGEBRAIC_FUSIONS, _TEMPLATE_FUSIONS,
  _VALID_OUTPUT_FUSIONS, _compute_decision_templates,
)
from quack.quantifiers import CC, ACC, PCC, GAC
from quack.bag_generator import PriorShiftBagGenerator


class _NoProbaClassifier(BaseEstimator, ClassifierMixin):
  def fit(self, X, y):
    self.classes_ = np.unique(y)
    return self

  def predict(self, X):
    return np.full(X.shape[0], self.classes_[0])


@pytest.fixture
def binary_dataset():
  return make_classification(n_samples=800, n_classes=2, weights=[0.6, 0.4], random_state=0)


@pytest.fixture
def multiclass_dataset():
  return make_classification(n_samples=800, n_classes=3, n_informative=6, random_state=0)


def _default_classifiers():
  return [
    RandomForestClassifier(n_estimators=30, random_state=0),
    HistGradientBoostingClassifier(random_state=0),
    LogisticRegression(max_iter=1000),
    GaussianNB(),
  ]


class TestIdentityScorer:
  def test_predict_proba_is_pass_through(self):
    scorer = _IdentityScorer().fit(np.zeros((3, 1)), np.array([0, 1, 0]))
    X = np.array([[0.2, 0.8], [0.6, 0.4]])
    np.testing.assert_array_equal(scorer.predict_proba(X), X)

  def test_predict_is_argmax_mapped_to_classes(self):
    scorer = _IdentityScorer().fit(np.zeros((3, 1)), np.array([0, 1, 0]))
    X = np.array([[0.2, 0.8], [0.9, 0.1]])
    np.testing.assert_array_equal(scorer.predict(X), [1, 0])

  def test_has_no_decision_function(self):
    assert not hasattr(_IdentityScorer(), "decision_function")


class TestFMCSQMemberSubstitution:
  def test_member_classifier_is_replaced_with_identity(self, binary_dataset):
    X, y = binary_dataset
    ensemble = FMCSQ(classifiers=_default_classifiers(), quantifier=ACC(cv=5), cv=5, random_state=0)
    ensemble.fit(X, y)
    assert isinstance(ensemble.quantifier_.classifier, _IdentityScorer)
    assert not isinstance(ensemble.quantifier.classifier, _IdentityScorer)


class TestFMCSQAccuracy:
  def test_predict_recovers_shifted_test_prevalence_reasonably(self, binary_dataset):
    X, y = binary_dataset
    X_train, y_train = X[:500], y[:500]
    X_pool, y_pool = X[500:], y[500:]

    ensemble = FMCSQ(classifiers=_default_classifiers(), quantifier=ACC(cv=5), cv=5, random_state=0)
    ensemble.fit(X_train, y_train)

    generator = PriorShiftBagGenerator(n_bags=1, bag_size=200, random_state=1)
    (X_bag, y_bag), = generator.to_list(X_pool, y_pool)
    true_prevalence = generator.sampled_prevalences_[0]

    predicted = ensemble.predict(X_bag)
    ae = np.mean(np.abs(predicted - true_prevalence))
    assert ae < 0.15


class TestFMCSQAlgebraicFusions:
  @pytest.mark.parametrize("fusion", _ALGEBRAIC_FUSIONS)
  def test_returns_valid_distribution(self, binary_dataset, fusion):
    X, y = binary_dataset
    ensemble = FMCSQ(classifiers=_default_classifiers(), quantifier=CC(), fusion=fusion, cv=5, random_state=0)
    ensemble.fit(X, y)
    prevalences = ensemble.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert np.all(prevalences >= 0.0)

  def test_templates_is_none_for_algebraic_fusion(self, binary_dataset):
    X, y = binary_dataset
    ensemble = FMCSQ(classifiers=_default_classifiers(), quantifier=CC(), fusion="mean", cv=5, random_state=0)
    ensemble.fit(X, y)
    assert ensemble.templates_ is None

  def test_cos_fusion_normalizes_each_classifier_before_averaging(self):
    from quack.ensembles._fmc import _BaseFMC

    class _Dummy(_BaseFMC):
      def fit(self, X, y): pass
      def predict(self, X): pass

    fuser = _Dummy(classifiers=[LogisticRegression()], fusion="cos")
    stack = np.array([[[3.0, 4.0]], [[0.0, 5.0]]])  # (2, 1, 2), norms 5 and 5
    result = fuser._algebraic_fuse(stack, "cos")
    # each vector normalized to unit norm: [0.6,0.8] and [0.0,1.0]; mean=[0.3,0.9]; renormalized to sum 1
    np.testing.assert_allclose(result, [[0.25, 0.75]])


class TestFMCSQTemplateFusions:
  @pytest.mark.parametrize("fusion", _TEMPLATE_FUSIONS)
  def test_returns_valid_one_hot_derived_distribution(self, multiclass_dataset, fusion):
    X, y = multiclass_dataset
    ensemble = FMCSQ(classifiers=_default_classifiers(), quantifier=CC(), fusion=fusion, cv=5, random_state=0)
    ensemble.fit(X, y)
    assert ensemble.templates_.shape == (3, len(_default_classifiers()), 3)
    prevalences = ensemble.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert np.all(prevalences >= 0.0)

  def test_decision_templates_match_manual_per_class_mean(self, multiclass_dataset):
    X, y = multiclass_dataset
    classes = np.unique(y)
    rng = np.random.default_rng(0)
    score_tensor = rng.dirichlet(np.ones(3), size=(2, len(y))).transpose(0, 1, 2)  # fake (c=2, n, l=3)

    templates = _compute_decision_templates(score_tensor, y, classes)
    for idx, cls in enumerate(classes):
      expected = score_tensor[:, y == cls, :].mean(axis=1)
      np.testing.assert_allclose(templates[idx], expected)

  def test_dt_fuse_picks_nearest_template(self):
    from quack.ensembles._fmc import _dt_fuse
    templates = np.array([
      [[1.0, 0.0]],  # class 0 template, classifier profile (c=1, l=2)
      [[0.0, 1.0]],  # class 1 template
    ])
    score_tensor = np.array([[[0.9, 0.1]], [[0.1, 0.9]]])  # (c=1, n=2, l=2)... reshape needed
    score_tensor = np.array([[[0.9, 0.1], [0.1, 0.9]]])  # (c=1, n=2, l=2)
    result = _dt_fuse(score_tensor, templates)
    np.testing.assert_array_equal(result, [[1, 0], [0, 1]])

  def test_ds_and_ml_produce_one_hot_rows(self, multiclass_dataset):
    X, y = multiclass_dataset
    for fusion in ("ds", "ml"):
      ensemble = FMCSQ(classifiers=_default_classifiers(), quantifier=CC(), fusion=fusion, cv=5, random_state=0)
      ensemble.fit(X, y)
      fused = ensemble._fuse_test_scores(X[:20])
      row_sums = fused.sum(axis=1)
      np.testing.assert_allclose(row_sums, 1.0)
      # each row must be exactly one-hot (a single 1.0, rest 0.0)
      assert np.all(np.isclose(fused, 0.0) | np.isclose(fused, 1.0))


class TestFMCSQ:
  def test_predict_sums_to_one_binary(self, binary_dataset):
    X, y = binary_dataset
    ensemble = FMCSQ(classifiers=_default_classifiers(), quantifier=CC(), cv=5, random_state=0)
    ensemble.fit(X, y)
    prevalences = ensemble.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (2,)

  def test_multiclass_supported(self, multiclass_dataset):
    X, y = multiclass_dataset
    ensemble = FMCSQ(classifiers=_default_classifiers(), quantifier=GAC(cv=5), cv=5, random_state=0)
    ensemble.fit(X, y)
    prevalences = ensemble.predict(X)
    assert prevalences.shape == (3,)
    assert prevalences.sum() == pytest.approx(1.0)

  @pytest.mark.parametrize("quantifier", [CC(), PCC(), ACC(cv=5)])
  def test_works_with_different_quantifier_types(self, binary_dataset, quantifier):
    X, y = binary_dataset
    ensemble = FMCSQ(classifiers=_default_classifiers()[:2], quantifier=quantifier, cv=5, random_state=0)
    ensemble.fit(X, y)
    assert ensemble.predict(X).sum() == pytest.approx(1.0)

  def test_raises_for_unknown_fusion(self, binary_dataset):
    X, y = binary_dataset
    ensemble = FMCSQ(classifiers=_default_classifiers(), quantifier=CC(), fusion="not-a-fusion")
    with pytest.raises(ValueError, match="Unknown fusion"):
      ensemble.fit(X, y)

  def test_raises_for_empty_classifiers(self, binary_dataset):
    X, y = binary_dataset
    ensemble = FMCSQ(classifiers=[], quantifier=CC())
    with pytest.raises(ValueError, match="non-empty list"):
      ensemble.fit(X, y)

  def test_raises_when_classifier_lacks_predict_proba(self, binary_dataset):
    X, y = binary_dataset
    ensemble = FMCSQ(classifiers=[_NoProbaClassifier()], quantifier=CC())
    with pytest.raises(TypeError, match="predict_proba"):
      ensemble.fit(X, y)

  def test_reproducible_with_same_random_state(self, binary_dataset):
    X, y = binary_dataset
    ens_a = FMCSQ(classifiers=_default_classifiers(), quantifier=CC(), cv=5, random_state=0).fit(X, y)
    ens_b = FMCSQ(classifiers=_default_classifiers(), quantifier=CC(), cv=5, random_state=0).fit(X, y)
    np.testing.assert_allclose(ens_a.predict(X), ens_b.predict(X))

  def test_accepts_n_jobs_and_parallel_backend(self, binary_dataset):
    X, y = binary_dataset
    ensemble = FMCSQ(
      classifiers=_default_classifiers(), quantifier=CC(), cv=5,
      n_jobs=2, parallel_backend="threading", random_state=0,
    )
    ensemble.fit(X, y)
    assert ensemble.predict(X).sum() == pytest.approx(1.0)


class TestFMCMQ:
  def test_predict_sums_to_one_binary(self, binary_dataset):
    X, y = binary_dataset
    ensemble = FMCMQ(classifiers=_default_classifiers(), quantifiers=[CC(), ACC(cv=5), PCC()], cv=5, random_state=0)
    ensemble.fit(X, y)
    prevalences = ensemble.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (2,)

  def test_multiclass_supported(self, multiclass_dataset):
    X, y = multiclass_dataset
    ensemble = FMCMQ(classifiers=_default_classifiers(), quantifiers=[CC(), GAC(cv=5)], cv=5, random_state=0)
    ensemble.fit(X, y)
    prevalences = ensemble.predict(X)
    assert prevalences.shape == (3,)
    assert prevalences.sum() == pytest.approx(1.0)

  def test_every_member_uses_identity_scorer(self, multiclass_dataset):
    X, y = multiclass_dataset
    ensemble = FMCMQ(classifiers=_default_classifiers(), quantifiers=[CC(), PCC()], cv=5, random_state=0)
    ensemble.fit(X, y)
    for member in ensemble.quantifiers_:
      assert isinstance(member.classifier, _IdentityScorer)
      assert len(member.classes_) == 3

  @pytest.mark.parametrize("output_fusion", _VALID_OUTPUT_FUSIONS)
  def test_output_fusion_options_all_valid(self, binary_dataset, output_fusion):
    X, y = binary_dataset
    ensemble = FMCMQ(
      classifiers=_default_classifiers(), quantifiers=[CC(), PCC()],
      fusion="mean", output_fusion=output_fusion, cv=5, random_state=0,
    )
    ensemble.fit(X, y)
    assert ensemble.predict(X).sum() == pytest.approx(1.0)

  def test_score_level_fusion_can_use_template_operator_independent_of_output_fusion(self, multiclass_dataset):
    X, y = multiclass_dataset
    ensemble = FMCMQ(
      classifiers=_default_classifiers(), quantifiers=[CC(), GAC(cv=5)],
      fusion="dt", output_fusion="mean", cv=5, random_state=0,
    )
    ensemble.fit(X, y)
    assert ensemble.predict(X).sum() == pytest.approx(1.0)

  def test_raises_for_unknown_output_fusion(self, binary_dataset):
    X, y = binary_dataset
    ensemble = FMCMQ(classifiers=_default_classifiers(), quantifiers=[CC()], output_fusion="not-a-fusion")
    with pytest.raises(ValueError, match="Unknown output_fusion"):
      ensemble.fit(X, y)

  def test_raises_when_default_output_fusion_is_template_based(self, binary_dataset):
    # fusion='dt' at score level is fine, but if output_fusion is left
    # None it would default to 'dt', which isn't valid at the output level
    X, y = binary_dataset
    ensemble = FMCMQ(classifiers=_default_classifiers(), quantifiers=[CC()], fusion="dt")
    with pytest.raises(ValueError, match="not one of the algebraic operators"):
      ensemble.fit(X, y)

  def test_raises_for_empty_quantifiers(self, binary_dataset):
    X, y = binary_dataset
    ensemble = FMCMQ(classifiers=_default_classifiers(), quantifiers=[])
    with pytest.raises(ValueError, match="non-empty list"):
      ensemble.fit(X, y)

  def test_accepts_n_jobs_and_parallel_backend(self, binary_dataset):
    X, y = binary_dataset
    ensemble = FMCMQ(
      classifiers=_default_classifiers(), quantifiers=[CC(), PCC()], cv=5,
      n_jobs=2, parallel_backend="threading", random_state=0,
    )
    ensemble.fit(X, y)
    assert ensemble.predict(X).sum() == pytest.approx(1.0)
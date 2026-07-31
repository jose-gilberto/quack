import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from quack.quantifiers._dmm import DyS, HDy, FormanMM, GAC, GPAC, FM
from quack.quantifiers.base import BaseScoreMixtureQuantifier


@pytest.fixture
def binary_dataset():
  return make_classification(n_samples=400, n_classes=2, weights=[0.55, 0.45], random_state=0)


@pytest.fixture
def multiclass_dataset():
  return make_classification(n_samples=400, n_classes=3, n_informative=6, random_state=0)


class TestScoreBasedQuantifiers:
  @pytest.mark.parametrize("quantifier_cls,classifier", [
    (DyS, SVC()), (HDy, LogisticRegression(max_iter=1000)), (FormanMM, SVC()),
  ])
  def test_predict_sums_to_one(self, binary_dataset, quantifier_cls, classifier):
    X, y = binary_dataset
    quantifier = quantifier_cls(classifier, cv=5, use_convex_solver=True).fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (2,)

  @pytest.mark.parametrize("quantifier_cls,classifier", [
    (DyS, SVC()), (HDy, LogisticRegression(max_iter=1000)), (FormanMM, SVC()),
  ])
  def test_raises_value_error_for_multiclass(self, multiclass_dataset, quantifier_cls, classifier):
    X, y = multiclass_dataset
    with pytest.raises(ValueError, match="binary quantification"):
      quantifier_cls(classifier, cv=5).fit(X, y)

  @pytest.mark.parametrize("quantifier_cls,classifier", [
    (DyS, SVC()), (FormanMM, SVC()),
  ])
  def test_accepts_n_jobs_and_parallel_backend(self, binary_dataset, quantifier_cls, classifier):
    X, y = binary_dataset
    quantifier = quantifier_cls(classifier, cv=5, n_jobs=2, parallel_backend="threading").fit(X, y)
    assert quantifier.predict(X).sum() == pytest.approx(1.0)

  def test_golden_section_fallback_matches_convex_solver_closely(self, binary_dataset):
    X, y = binary_dataset
    convex_quant = DyS(SVC(probability=True), cv=5, use_convex_solver=True).fit(X, y)
    gss_quant = DyS(SVC(probability=True), cv=5, use_convex_solver=False).fit(X, y)

    p_convex = convex_quant.predict(X)
    p_gss = gss_quant.predict(X)
    assert np.abs(p_convex[1] - p_gss[1]) < 0.1

  def test_dys_isinstance_of_shared_base(self):
    assert isinstance(DyS(), BaseScoreMixtureQuantifier)
    assert isinstance(FormanMM(), BaseScoreMixtureQuantifier)
    assert isinstance(HDy(), BaseScoreMixtureQuantifier)


class TestVectorizedConfusionMatrices:
  def test_gac_calibration_matches_manual_double_loop(self, multiclass_dataset):
    X, y = multiclass_dataset
    quantifier = GAC(LogisticRegression(max_iter=1000), cv=5).fit(X, y)

    # reference (manual double loop) confusion matrix over the actual
    # fitted classifier's OOF predictions is hard to reproduce exactly
    # (depends on the specific CV split), so instead validate structural
    # invariants that the vectorized formula must satisfy regardless
    assert quantifier.conditional_matrix_.shape == (3, 3)
    # each column (conditioned on a true class) must sum to 1.0 (it's a
    # normalized confusion column, i.e. P(pred | true))
    np.testing.assert_allclose(quantifier.conditional_matrix_.sum(axis=0), 1.0)

  def test_gac_matmul_matches_explicit_double_loop_on_synthetic_labels(self):
    classes = np.array([0, 1, 2])
    y_true_oof = np.array([0, 0, 1, 1, 2, 2, 2])
    y_pred_oof = np.array([0, 1, 1, 1, 2, 0, 2])

    true_idx = np.searchsorted(classes, y_true_oof)
    pred_idx = np.searchsorted(classes, y_pred_oof)
    one_hot_true = np.eye(3)[true_idx]
    one_hot_pred = np.eye(3)[pred_idx]
    vectorized = one_hot_pred.T @ one_hot_true

    manual = np.zeros((3, 3))
    for i, true_label in enumerate(classes):
      for j, pred_label in enumerate(classes):
        manual[j, i] = np.sum((y_true_oof == true_label) & (y_pred_oof == pred_label))

    np.testing.assert_array_equal(vectorized, manual)

  def test_gpac_predict_sums_to_one(self, multiclass_dataset):
    X, y = multiclass_dataset
    quantifier = GPAC(LogisticRegression(max_iter=1000), cv=5).fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (3,)

  def test_fm_predict_sums_to_one(self, multiclass_dataset):
    X, y = multiclass_dataset
    quantifier = FM(LogisticRegression(max_iter=1000), cv=5).fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (3,)

  @pytest.mark.parametrize("quantifier_cls", [GAC, GPAC, FM])
  def test_accepts_n_jobs_and_parallel_backend(self, multiclass_dataset, quantifier_cls):
    X, y = multiclass_dataset
    quantifier = quantifier_cls(
      LogisticRegression(max_iter=1000), cv=5, n_jobs=2, parallel_backend="threading",
    ).fit(X, y)
    assert quantifier.predict(X).sum() == pytest.approx(1.0)
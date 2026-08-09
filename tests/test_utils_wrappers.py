import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.estimator_checks import check_estimator

from quack.utils.wrappers import SklearnClassifierWrapper
from quack.quantifiers import CC


class _DummyModel:
  """Minimal non-sklearn-flavored model with standard method names."""

  def __init__(self, alpha: float = 1.0):
    self.alpha = alpha

  def fit(self, X, y):
    self.classes_, counts = np.unique(y, return_counts=True)
    self._majority = self.classes_[np.argmax(counts)]
    return self

  def predict(self, X):
    return np.full(len(X), self._majority)

  def predict_proba(self, X):
    proba = np.zeros((len(X), len(self.classes_)))
    proba[:, np.argmax(self.classes_ == self._majority)] = 1.0
    return proba


class _NoProbaModel:
  def fit(self, X, y):
    self.classes_ = np.unique(y)
    return self

  def predict(self, X):
    return np.full(len(X), self.classes_[0])


class _LLMLikeClassifier:
  """Model with entirely non-standard method names, exercising the
  fit_fn/predict_fn adapters."""

  def __init__(self, prompt_template: str = "Classify: {text}"):
    self.prompt_template = prompt_template

  def train(self, X, y):
    self.labels_seen_ = sorted(set(y))
    return self

  def classify(self, X):
    return [self.labels_seen_[0] for _ in X]


@pytest.fixture
def text_dataset():
  X = np.array([
    "I loved this movie", "terrible film, would not recommend",
    "what a great show", "worst movie ever made",
    "fantastic acting all around", "awful plot and boring",
  ], dtype=object).reshape(-1, 1)
  y = np.array([1, 0, 1, 0, 1, 0])
  return X, y


class TestBasicWrapping:
  def test_fit_predict_roundtrip(self):
    wrapper = SklearnClassifierWrapper(model_factory=_DummyModel)
    X, y = np.zeros((6, 2)), np.array([0, 1, 0, 1, 0, 1])
    wrapper.fit(X, y)
    predictions = wrapper.predict(X)
    assert predictions.shape == (6,)

  def test_classes_recovered_from_model(self):
    wrapper = SklearnClassifierWrapper(model_factory=_DummyModel)
    wrapper.fit(np.zeros((4, 2)), np.array([0, 1, 1, 0]))
    np.testing.assert_array_equal(wrapper.classes_, [0, 1])

  def test_classes_fn_overrides_model_classes(self):
    wrapper = SklearnClassifierWrapper(
      model_factory=_DummyModel, classes_fn=lambda model: ["neg", "pos"],
    )
    wrapper.fit(np.zeros((4, 2)), np.array([0, 1, 1, 0]))
    np.testing.assert_array_equal(wrapper.classes_, ["neg", "pos"])

  def test_predict_before_fit_raises_not_fitted(self):
    from sklearn.exceptions import NotFittedError
    wrapper = SklearnClassifierWrapper(model_factory=_DummyModel)
    with pytest.raises(NotFittedError):
      wrapper.predict(np.zeros((2, 2)))

  def test_model_factory_returning_none_raises(self):
    wrapper = SklearnClassifierWrapper(model_factory=lambda: None)
    with pytest.raises(TypeError, match="fresh model instance"):
      wrapper.fit(np.zeros((2, 2)), np.array([0, 1]))

  def test_missing_fit_method_raises_without_fit_fn(self):
    class _NoFitModel:
      def predict(self, X):
        return np.zeros(len(X))
    wrapper = SklearnClassifierWrapper(model_factory=_NoFitModel)
    with pytest.raises(TypeError, match="no 'fit' method"):
      wrapper.fit(np.zeros((2, 2)), np.array([0, 1]))


class TestPredictProbaCapabilityDetection:
  def test_auto_detects_predict_proba_from_model_class(self):
    wrapper = SklearnClassifierWrapper(model_factory=_DummyModel)
    assert hasattr(wrapper, "predict_proba")

  def test_auto_detects_missing_predict_proba(self):
    wrapper = SklearnClassifierWrapper(model_factory=_NoProbaModel)
    assert not hasattr(wrapper, "predict_proba")

  def test_predict_proba_fn_forces_availability(self):
    wrapper = SklearnClassifierWrapper(
      model_factory=_NoProbaModel,
      predict_proba_fn=lambda model, X: np.tile([1.0, 0.0], (len(X), 1)),
    )
    assert hasattr(wrapper, "predict_proba")
    wrapper.fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))
    proba = wrapper.predict_proba(np.zeros((2, 2)))
    assert proba.shape == (2, 2)

  def test_explicit_supports_predict_proba_false_overrides_auto(self):
    wrapper = SklearnClassifierWrapper(model_factory=_DummyModel, supports_predict_proba=False)
    assert not hasattr(wrapper, "predict_proba")

  def test_explicit_supports_predict_proba_true_for_non_class_factory(self):
    # a lambda factory can't be introspected, so 'auto' would default to False;
    # forcing True should still expose predict_proba (may fail at call time if
    # the model genuinely lacks it, which is a user configuration error)
    wrapper = SklearnClassifierWrapper(
      model_factory=lambda: _DummyModel(alpha=2.0), supports_predict_proba=True,
    )
    assert hasattr(wrapper, "predict_proba")

  def test_lambda_factory_defaults_to_no_predict_proba_without_override(self):
    wrapper = SklearnClassifierWrapper(model_factory=lambda: _DummyModel())
    assert not hasattr(wrapper, "predict_proba")


class TestCustomAdapters:
  def test_fit_fn_and_predict_fn_for_non_standard_api(self):
    wrapper = SklearnClassifierWrapper(
      model_factory=_LLMLikeClassifier,
      fit_fn=lambda model, X, y: model.train(X, y),
      predict_fn=lambda model, X: model.classify(X),
    )
    wrapper.fit(["good", "bad", "great"], [1, 0, 1])
    predictions = wrapper.predict(["ok", "meh"])
    assert len(predictions) == 2

  def test_fit_params_forwarded(self):
    calls = []

    class _RecordingModel:
      def fit(self, X, y, sample_weight=None):
        calls.append(sample_weight)
        self.classes_ = np.unique(y)
        return self
      def predict(self, X):
        return np.full(len(X), self.classes_[0])

    wrapper = SklearnClassifierWrapper(
      model_factory=_RecordingModel, fit_params={"sample_weight": [1.0, 2.0, 1.0, 1.0]},
    )
    wrapper.fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))
    assert calls == [[1.0, 2.0, 1.0, 1.0]]


class TestSklearnCloneCompatibility:
  def test_clone_produces_unfitted_wrapper(self):
    wrapper = SklearnClassifierWrapper(model_factory=_DummyModel)
    wrapper.fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))
    cloned = clone(wrapper)
    assert not hasattr(cloned, "model_")

  def test_clone_preserves_predict_proba_capability(self):
    wrapper = SklearnClassifierWrapper(model_factory=_DummyModel)
    cloned = clone(wrapper)
    assert hasattr(cloned, "predict_proba")

  def test_cloned_wrapper_fits_and_predicts_independently(self):
    wrapper = SklearnClassifierWrapper(model_factory=_DummyModel)
    wrapper.fit(np.zeros((4, 2)), np.array([1, 1, 0, 1]))
    cloned = clone(wrapper)
    cloned.fit(np.zeros((4, 2)), np.array([0, 0, 1, 0]))
    # each keeps its own fitted model (majority class differs)
    assert wrapper.predict(np.zeros((1, 2)))[0] == 1
    assert cloned.predict(np.zeros((1, 2)))[0] == 0

  def test_clone_with_class_factory_reuses_same_class(self):
    wrapper = SklearnClassifierWrapper(model_factory=RandomForestClassifier)
    cloned = clone(wrapper)
    assert cloned.model_factory is RandomForestClassifier

  def test_wrapper_survives_pickling(self):
    import pickle
    wrapper = SklearnClassifierWrapper(model_factory=_DummyModel)
    wrapper.fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))
    restored = pickle.loads(pickle.dumps(wrapper))
    np.testing.assert_array_equal(restored.predict(np.zeros((2, 2))), wrapper.predict(np.zeros((2, 2))))


class TestIntegrationWithQuackQuantifiers:
  def test_wrapped_model_works_as_cc_classifier(self):
    wrapper = SklearnClassifierWrapper(model_factory=_DummyModel)
    quantifier = CC(classifier=wrapper)
    X, y = np.zeros((10, 2)), np.array([0, 1] * 5)
    quantifier.fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)

  def test_wrapped_model_works_with_acc_and_cross_validation(self):
    from quack.quantifiers import ACC
    wrapper_factory = lambda: SklearnClassifierWrapper(model_factory=_DummyModel)  # noqa: E731
    quantifier = ACC(classifier=wrapper_factory(), cv=3)
    X = np.random.default_rng(0).normal(size=(60, 3))
    y = np.array([0, 1] * 30)
    quantifier.fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)

  def test_text_input_via_wrapped_pipeline_and_dtype_none(self, text_dataset):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    X, y = text_dataset
    text_pipeline = Pipeline([
      ("ravel", FunctionTransformer(lambda x: np.asarray(x).ravel())),
      ("tfidf", TfidfVectorizer()),
      ("clf", LogisticRegression(max_iter=1000)),
    ])
    quantifier = CC(classifier=text_pipeline)
    quantifier.fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)
    assert prevalences.shape == (2,)

  def test_llm_like_classifier_via_wrapper_with_cc(self, text_dataset):
    X, y = text_dataset
    wrapper = SklearnClassifierWrapper(
      model_factory=_LLMLikeClassifier,
      fit_fn=lambda model, X, y: model.train(X.ravel().tolist(), y),
      predict_fn=lambda model, X: model.classify(X.ravel().tolist()),
    )
    quantifier = CC(classifier=wrapper)
    quantifier.fit(X, y)
    prevalences = quantifier.predict(X)
    assert prevalences.sum() == pytest.approx(1.0)

  def test_predict_proba_gate_prevents_wasted_cv_in_pacc(self, text_dataset):
    from quack.quantifiers import PACC
    X, y = text_dataset
    wrapper = SklearnClassifierWrapper(model_factory=_NoProbaModel)
    with pytest.raises(TypeError, match="predict_proba"):
      PACC(classifier=wrapper, cv=3).fit(X, y)
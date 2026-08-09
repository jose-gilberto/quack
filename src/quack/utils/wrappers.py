import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class SklearnClassifierWrapper(BaseEstimator, ClassifierMixin):
  """Adapts a non-scikit-learn classifier so it can be used as the base
  `classifier` of any `quack` quantifier.

  Every `quack` quantifier that fits an internal classifier relies on
  `sklearn.base.clone()` — once per CV fold in `BaseCalibratedQuantifier`,
  once per ensemble member in `EoQ`/`FMCSQ`/`FMCMQ`/`MCSQ`/`MCMQ` - which
  in turn requires `get_params()`/`set_params()` over the estimator's
  *constructor* arguments (see `sklearn.base.BaseEstimator`). Third-party
  models (PyTorch/TensorFlow modules, remote/API-backed classifiers,
  LLM-based classifiers, or any object that simply doesn't follow that
  convention) break this contract. `SklearnClassifierWrapper` fixes this
  by storing only a *recipe* for building the model - never a live model
  instance - as its constructor arguments, so `clone()` always produces a
  brand-new, unfitted wrapper (exactly what every quantifier's per-fold /
  per-member fitting logic assumes) instead of deep-copying a
  potentially large, stateful, or altogether unpicklable object.

  Parameters
  ----------
  model_factory : callable
    Zero-argument callable returning a *fresh, unfit* instance of the
    underlying model every time it's called — e.g. the model's own class
    (`MyClassifier`), `functools.partial(MyClassifier, lr=0.01)`, or a
    plain `lambda: MyClassifier(...)`. Called exactly once per `.fit()`;
    the previously fitted model (if any) is discarded rather than
    reused/mutated, matching how every other `quack` classifier is
    refit from scratch on each call.
  fit_params : dict, default = None
    Extra keyword arguments forwarded to the underlying model's training
    call on every `.fit()`.
  fit_fn : callable(model, X, y, **fit_params), default = None
    Adapter for models whose training method isn't named `fit` (e.g.
    `.train(...)`), or that need custom pre/post-processing of `X`/`y`
    before training. Defaults to `model.fit(X, y, **fit_params)`.
  predict_fn : callable(model, X) -> array-like, default = None
    Adapter for models whose inference method isn't named `predict`.
    Defaults to `model.predict(X)`.
  predict_proba_fn : callable(model, X) -> array-like, default = None
    Adapter for models whose probability method isn't named
    `predict_proba`, or that need probabilities composed from raw model
    outputs (e.g. a softmax over logits). Defaults to
    `model.predict_proba(X)`.
  classes_fn : callable(model) -> array-like, default = None
    How to recover the fitted class labels from `model` after training,
    for models that don't expose their own `classes_` attribute.
    Defaults to `np.unique(y)` over the training labels seen in `.fit()`.
  supports_predict_proba : bool | "auto", default = "auto"
    Whether `.predict_proba()` should be exposed at all. `"auto"`
    resolves to `True` when `predict_proba_fn` is given, or when
    `model_factory` is itself the model's class and that class already
    defines `predict_proba`; `False` otherwise. Set this explicitly
    (`True`/`False`) whenever `model_factory` is a factory function/
    `functools.partial`/lambda rather than the model's own class, since
    then it can't be introspected without being called. Getting this
    right matters: `PCC`, `PACC`, `HDy`/`DyS`/`FormanMM` (unless
    `predict_proba=True`... wait `predict_proba` there means score
    source) and `BaseCalibratedQuantifier.fit`'s fail-fast check all
    branch on `hasattr(classifier, "predict_proba")` *before* running
    (or in `BaseCalibratedQuantifier`'s case, before even starting) any
    cross-validation.

  Attributes
  ----------
  model_ : object
    The fitted underlying model, freshly built by `model_factory` on
    every `.fit()` call.
  classes_ : ndarray of shape (n_classes,)
    Sorted unique class labels observed during `.fit()` (or produced by
    `classes_fn`, if given).

  Notes
  -----
  `predict_proba` is bound as an instance attribute in `__init__` only
  when `supports_predict_proba` resolves to `True` — never defined as a
  plain class method. This is what makes `hasattr(wrapper, "predict_proba")`
  correctly reflect real availability throughout `quack` (rather than
  always returning `True` and only failing lazily, deep inside a CV
  fold, the way a always-defined method that internally raises would).

  Because only `__init__`-declared arguments are ever considered by
  `get_params()`/`clone()`, cloning a *fitted* wrapper never touches
  `model_`/`classes_`: the clone is unfitted, with a fresh `model_`
  built from `model_factory` the next time `.fit()` runs — exactly the
  semantics `quack.quantifiers.base._clone_fit_predict_fold` (and every
  other per-fold/per-member cloning path in `quack`) already assumes.

  Examples
  --------
  Wrapping a plain scikit-learn-incompatible object with non-standard
  method names (e.g. a hypothetical LLM-backed text classifier):

  >>> class LLMClassifier:
  ...   def __init__(self, prompt_template="Classify: {text}"):
  ...     self.prompt_template = prompt_template
  ...   def train(self, X, y):
  ...     self.labels_seen_ = sorted(set(y))
  ...     return self
  ...   def classify(self, X):
  ...     return [self.labels_seen_[0] for _ in X]
  >>> from quack.quantifiers import SklearnClassifierWrapper, CC
  >>> classifier = SklearnClassifierWrapper(
  ...   model_factory=LLMClassifier,
  ...   fit_fn=lambda model, X, y: model.train(X.ravel().tolist(), y),
  ...   predict_fn=lambda model, X: model.classify(X.ravel().tolist()),
  ... )
  >>> quantifier = CC(classifier=classifier)
  >>> import numpy as np
  >>> X = np.array(["good movie", "bad film", "great show"], dtype=object).reshape(-1, 1)
  >>> quantifier.fit(X, [1, 0, 1])
  >>> quantifier.predict(X)

  A model that *does* support probabilities, auto-detected because
  `model_factory` is the model's own class:

  >>> from sklearn.ensemble import RandomForestClassifier
  >>> classifier = SklearnClassifierWrapper(model_factory=RandomForestClassifier)
  >>> hasattr(classifier, "predict_proba")
  True
  """

  def __init__(self,
               model_factory,
               fit_params: dict = None,
               fit_fn=None,
               predict_fn=None,
               predict_proba_fn=None,
               classes_fn=None,
               supports_predict_proba="auto"):
    self.model_factory = model_factory
    self.fit_params = fit_params
    self.fit_fn = fit_fn
    self.predict_fn = predict_fn
    self.predict_proba_fn = predict_proba_fn
    self.classes_fn = classes_fn
    self.supports_predict_proba = supports_predict_proba

    # bound only when actually usable, so hasattr(...)-based capability
    # checks across quack (BaseCalibratedQuantifier.fit, PCC.fit, ...)
    # detect its absence up front instead of via a lazily raised error
    if self._resolve_supports_predict_proba():
      self.predict_proba = self._predict_proba_impl

  def _resolve_supports_predict_proba(self) -> bool:
    if self.supports_predict_proba != "auto":
      return bool(self.supports_predict_proba)
    if self.predict_proba_fn is not None:
      return True
    return isinstance(self.model_factory, type) and hasattr(self.model_factory, "predict_proba")

  def _build_model(self):
    model = self.model_factory()
    if model is None:
      raise TypeError(
        "model_factory must return a fresh model instance on every call, got None. "
        "Pass the model's class directly (e.g. MyClassifier), or a zero-argument "
        "callable such as functools.partial(MyClassifier, **kwargs) or a lambda."
      )
    return model

  def fit(self, X, y) -> 'SklearnClassifierWrapper':
    """Builds a fresh underlying model via `model_factory` and trains it.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
      Training data, in whatever form the wrapped model (or `fit_fn`)
      expects — including an object-dtype array of raw text when the
      quantifier calling this was itself fit with `dtype=None` validation.
    y : array-like of shape (n_samples,)
      Training labels.

    Returns
    -------
    self : object
      Returns the fitted wrapper instance itself.
    """
    self.model_ = self._build_model()
    fit_params = self.fit_params or {}

    if self.fit_fn is not None:
      self.fit_fn(self.model_, X, y, **fit_params)
    else:
      if not hasattr(self.model_, "fit"):
        raise TypeError(
          f"{self.model_.__class__.__name__} has no 'fit' method and no fit_fn "
          "adapter was provided to SklearnClassifierWrapper."
        )
      self.model_.fit(X, y, **fit_params)

    if self.classes_fn is not None:
      self.classes_ = np.asarray(self.classes_fn(self.model_))
    elif hasattr(self.model_, "classes_"):
      self.classes_ = np.asarray(self.model_.classes_)
    else:
      self.classes_ = np.unique(y)

    return self

  def predict(self, X) -> np.ndarray:
    """Delegates prediction to the underlying model (or `predict_fn`).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
      Data to predict on.

    Returns
    -------
    predictions : ndarray of shape (n_samples,)
      Hard class predictions.
    """
    check_is_fitted(self, "model_")
    if self.predict_fn is not None:
      predictions = self.predict_fn(self.model_, X)
    else:
      if not hasattr(self.model_, "predict"):
        raise TypeError(
          f"{self.model_.__class__.__name__} has no 'predict' method and no predict_fn "
          "adapter was provided to SklearnClassifierWrapper."
        )
      predictions = self.model_.predict(X)
    return np.asarray(predictions)

  def _predict_proba_impl(self, X) -> np.ndarray:
    """Delegates probability estimation to the underlying model (or
    `predict_proba_fn`); only bound as `self.predict_proba` when
    `_resolve_supports_predict_proba()` is True (see `__init__`)."""
    check_is_fitted(self, "model_")
    if self.predict_proba_fn is not None:
      probabilities = self.predict_proba_fn(self.model_, X)
    else:
      probabilities = self.model_.predict_proba(X)
    return np.asarray(probabilities)
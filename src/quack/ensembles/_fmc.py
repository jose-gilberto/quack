import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.parallel import Parallel, delayed

from quack.quantifiers.base import BaseQuantifier, normalize_prevalence

_ALGEBRAIC_FUSIONS = ("mean", "median", "min", "max", "prod", "cos")
_TEMPLATE_FUSIONS = ("dt", "ds", "ml")
_VALID_SCORE_FUSIONS = _ALGEBRAIC_FUSIONS + _TEMPLATE_FUSIONS
_VALID_OUTPUT_FUSIONS = ("mean", "median", "min", "max", "prod")


class _IdentityScorer(BaseEstimator, ClassifierMixin):
  """Pass-through pseudo-classifier feeding an already-computed, per-class
  probability matrix directly into a `quack` quantifier's internal
  calibration/prediction machinery, without fitting a second classifier
  on top of it.

  This is the crux of the FMC-SQ/FMC-MQ architecture (Serapião, Donyavi &
  Batista, 2023): the fused classifier-score matrix `S̄ = N(F(S))` is
  already an unbiased (out-of-fold) posterior-probability estimate,
  computed once by the ensemble's own `c` classifiers. A downstream
  score-consuming quantifier `Q` (e.g. `CC`, `ACC`, `GAC`, `GPAC`, `EM`)
  must treat it as its classifier's own output, rather than as raw
  feature space to fit yet another classifier on.

  `.fit(X, y)` only records `classes_` (sorted, matching every other
  `quack` estimator's convention); `.predict_proba(X)` returns `X`
  unchanged; `.predict(X)` returns the argmax class per row. Deliberately
  does *not* implement `decision_function`, so quantifiers that prefer it
  when available (`DyS`/`FormanMM`) correctly fall back to
  `predict_proba`'s pass-through semantics instead.
  """

  def fit(self, X: np.ndarray, y: np.ndarray) -> '_IdentityScorer':
    self.classes_ = np.unique(y)
    return self

  def predict_proba(self, X: np.ndarray) -> np.ndarray:
    return np.asarray(X)

  def predict(self, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    return self.classes_[np.argmax(X, axis=1)]


def _oof_and_fit_job(classifier: BaseEstimator, X: np.ndarray, y: np.ndarray, cv) -> tuple:
  """Computes one classifier's Out-of-Fold `predict_proba` matrix and
  refits it on the full training data.

  Defined at module level (rather than as a closure/method) so it can be
  pickled and dispatched to worker processes by `joblib`/`Parallel`, the
  same pattern used for the per-fold jobs in `quack.quantifiers.base`.
  """
  oof_proba = cross_val_predict(classifier, X, y, cv=cv, method="predict_proba")
  fitted_classifier = clone(classifier).fit(X, y)
  return oof_proba, fitted_classifier


def _predict_proba_job(classifier: BaseEstimator, X: np.ndarray) -> np.ndarray:
  """Scores one test bag against a single fitted classifier."""
  return classifier.predict_proba(X)


def _fit_quantifier_job(quantifier: BaseQuantifier, X: np.ndarray, y: np.ndarray) -> BaseQuantifier:
  """Fits one already-prepared (identity-substituted, where applicable)
  member quantifier on the shared fused score matrix."""
  quantifier.fit(X, y)
  return quantifier


def _predict_quantifier_job(quantifier: BaseQuantifier, X: np.ndarray) -> np.ndarray:
  """Scores the fused test score matrix against a single fitted member quantifier."""
  return quantifier.predict(X)


def _compute_decision_templates(score_tensor: np.ndarray, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
  """Computes per-class decision templates `DT_j` (Kuncheva, 2004, Ch. 5).

  `DT_j` is the average `(c, l)` score profile observed across every
  training instance truly belonging to class `j`, over the `c`
  classifiers and `l` classes — the reference profile a "typical" class
  `j` instance produces from the classifier committee.

  Args:
    score_tensor (np.ndarray): Out-of-Fold classifier scores, shape `(c, n, l)`.
    y (np.ndarray): True training labels, shape `(n,)`.
    classes (np.ndarray): Sorted unique class labels, shape `(l,)`.

  Returns:
    np.ndarray: Decision templates of shape `(l, c, l)`.
  """
  n_classifiers, _, n_classes_score = score_tensor.shape
  templates = np.zeros((len(classes), n_classifiers, n_classes_score))
  for idx, cls in enumerate(classes):
    mask = (y == cls)
    templates[idx] = score_tensor[:, mask, :].mean(axis=1)
  return templates


def _template_proximity(score_tensor: np.ndarray, templates: np.ndarray) -> np.ndarray:
  """Per-classifier normalized proximity `Phi_{k,j}(x)` between each test
  instance's per-classifier score vector and each class's decision
  template row (Kuncheva, 2004, eq. 5.16-ish): an inverse-squared-distance
  similarity, normalized to sum to 1 over classes for each classifier.

  Shared by the `'ds'` and `'ml'` template-based fusion operators.

  Args:
    score_tensor (np.ndarray): Classifier scores, shape `(c, n, l)`.
    templates (np.ndarray): Decision templates, shape `(l, c, l)`.

  Returns:
    np.ndarray: `Phi_{k,j}(x)` of shape `(n_classes, c, n_samples)`.
  """
  diff = templates[:, :, None, :] - score_tensor[None, :, :, :]  # (n_classes, c, n, l)
  sq_dist = np.sum(diff ** 2, axis=-1)  # (n_classes, c, n)
  inv_dist = 1.0 / (1.0 + sq_dist)
  return inv_dist / inv_dist.sum(axis=0, keepdims=True)  # normalize over classes


def _dt_fuse(score_tensor: np.ndarray, templates: np.ndarray) -> np.ndarray:
  """Decision Templates (DT) combiner (Kuncheva, 2004): assigns each test
  instance to the class whose template minimizes the squared Euclidean
  distance to the observed `(c, l)` decision profile, as explicitly
  specified in the paper ("For DT, we use Euclidean distance as a
  similarity measure"). Returns a one-hot `(n, l)` decision matrix.
  """
  diff = templates[:, :, None, :] - score_tensor[None, :, :, :]  # (n_classes, c, n, l)
  sq_dist = np.sum(diff ** 2, axis=(1, 3))  # (n_classes, n)
  decision = np.argmin(sq_dist, axis=0)
  return np.eye(templates.shape[0])[decision]


def _ds_fuse(score_tensor: np.ndarray, templates: np.ndarray) -> np.ndarray:
  """Dempster-Shafer (DS) combiner (Kuncheva, 2004, §5.4): treats each
  classifier's proximity to the decision templates as evidence, combines
  the per-classifier beliefs `b_k(j) = Phi_{k,j} * prod_{i != j}(1 - Phi_{k,i})`
  across classifiers via Dempster's product rule, and assigns each
  instance to the class with the highest combined belief.

  Notes
  -----
  This implements the standard proximity-and-belief construction from
  Kuncheva's textbook, but omits its final per-classifier renormalization
  term (the `1 - Phi_{k,j} * (1 - prod_{i!=j}(1-Phi_{k,i}))` denominator),
  relying instead on a single normalization after combining across
  classifiers. This monotonicity-preserving simplification does not
  change the resulting argmax decision (and therefore not the one-hot
  output), only unnormalized intermediate magnitudes that are never
  otherwise consumed.
  """
  phi = _template_proximity(score_tensor, templates)  # (n_classes, c, n)
  total_complement = np.prod(1.0 - phi, axis=0, keepdims=True)  # (1, c, n)
  with np.errstate(divide='ignore', invalid='ignore'):
    prod_excl_self = np.where((1.0 - phi) > 1e-12, total_complement / (1.0 - phi), 0.0)
  belief = phi * prod_excl_self  # (n_classes, c, n)
  combined = np.prod(belief, axis=1)  # product across classifiers -> (n_classes, n)
  decision = np.argmax(combined, axis=0)
  return np.eye(templates.shape[0])[decision]


def _ml_fuse(score_tensor: np.ndarray, templates: np.ndarray, prior: np.ndarray) -> np.ndarray:
  """Maximum Likelihood (ML) combiner (Kuncheva, 2004, §5.5): a naive-Bayes-
  style combination assuming per-classifier proximities are conditionally
  independent given the true class, `mu(j) = prior(j) * prod_k Phi_{k,j}(x)`,
  assigning each instance to the class maximizing `mu`.
  """
  phi = _template_proximity(score_tensor, templates)  # (n_classes, c, n)
  combined = np.prod(phi, axis=1) * prior[:, None]  # (n_classes, n)
  decision = np.argmax(combined, axis=0)
  return np.eye(templates.shape[0])[decision]


class _BaseFMC(BaseQuantifier, ABC):
  """Shared classifier-fusion machinery for `FMCSQ` and `FMCMQ`.

  Implements the common "C" stage from Serapião, Donyavi & Batista (2023):
  fitting `c` independent classifiers, computing their Out-of-Fold
  `predict_proba` matrices, and fusing the `c` score matrices into one via
  a fusion operator `F` followed by row-normalization `N`, `S̄ = N(F(S))`.

  Parameters
  ----------
  classifiers : list of estimator objects
    The `c` classifiers to fuse. Each must implement `predict_proba`.
  fusion : {'mean', 'median', 'min', 'max', 'prod', 'cos', 'dt', 'ds', 'ml'}, default = 'mean'
    Fusion operator `F` combining the `c` classifiers' score matrices:

    - `'mean'`/`'median'`/`'min'`/`'max'`/`'prod'`: elementwise algebraic
      combination of the `c` score vectors, followed by `N`.
    - `'cos'`: L2-normalizes each classifier's score vector before
      averaging (a cosine-geometry combiner), followed by `N`. The paper
      does not spell out its exact formula; this is the standard
      interpretation of a "cosine similarity" score combiner.
    - `'dt'`/`'ds'`/`'ml'`: one-hot decision combiners based on Kuncheva
      (2004)'s decision templates, each returning a `1`/`0` vector per
      instance rather than a smooth probability (see `_dt_fuse`,
      `_ds_fuse`, `_ml_fuse`).
  cv : int, cross-validation generator or an iterable, default = 10
    Cross-validation strategy used to generate each classifier's
    Out-of-Fold `predict_proba` matrix (and, for `'dt'`/`'ds'`/`'ml'`, the
    decision templates).
  n_jobs : int, default = None
    Number of jobs to run in parallel across the `c` independent
    classifiers. See `joblib.Parallel`.
  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the classifier jobs.
  random_state : int, RandomState instance or None, default = None
    Reserved for subclasses/future fusion strategies that require
    randomness; unused by every fusion operator currently implemented.
  """

  def __init__(self,
               classifiers: list,
               fusion: str = "mean",
               cv: int = 10,
               n_jobs: int = None,
               parallel_backend: str = "loky",
               random_state=None):
    super().__init__(classifier=None)
    self.classifiers = classifiers
    self.fusion = fusion
    self.cv = cv
    self.n_jobs = n_jobs
    self.parallel_backend = parallel_backend
    self.random_state = random_state

  def _validate_fusion(self):
    if self.fusion not in _VALID_SCORE_FUSIONS:
      raise ValueError(f"Unknown fusion '{self.fusion}'. Supported options are {_VALID_SCORE_FUSIONS}.")
    if not self.classifiers:
      raise ValueError("classifiers must be a non-empty list of estimators.")
    for clf in self.classifiers:
      if not hasattr(clf, "predict_proba"):
        raise TypeError(
          f"Classifier {clf.__class__.__name__} does not support probability "
          "estimation; every classifier passed to FMC-based ensembles must "
          "implement 'predict_proba'."
        )

  @staticmethod
  def _prepare_member(quantifier: BaseQuantifier) -> BaseQuantifier:
    """Clones `quantifier` and, if it exposes a `classifier` parameter,
    substitutes it with `_IdentityScorer` so it consumes the fused score
    matrix directly instead of fitting a second classifier on top of it.
    Feature-based quantifiers without a `classifier` parameter (`HDx`,
    `ReadMe`, `ED`) are left untouched, since they operate on `X` as raw
    features by design.
    """
    member = clone(quantifier)
    if hasattr(member, "classifier"):
      member.set_params(classifier=_IdentityScorer())
    return member

  @staticmethod
  def _algebraic_fuse(score_stack: np.ndarray, fusion: str) -> np.ndarray:
    """Applies an elementwise algebraic fusion operator along
    `score_stack`'s leading axis, followed by the normalization function
    `N` (clip negative noise, divide by row/vector sum; rows that
    collapse entirely to zero mass fall back to a uniform distribution).

    Works uniformly for the `(c, n_samples, n_classes)` classifier-score
    tensor (`FMCSQ`/`FMCMQ`'s "C" stage, algebraic operators only) and the
    `(q, n_classes)` member prediction stack (`FMCMQ`'s output-fusion
    stage), since both simply collapse axis 0.
    """
    if fusion == "mean":
      fused = score_stack.mean(axis=0)
    elif fusion == "median":
      fused = np.median(score_stack, axis=0)
    elif fusion == "min":
      fused = score_stack.min(axis=0)
    elif fusion == "max":
      fused = score_stack.max(axis=0)
    elif fusion == "prod":
      fused = score_stack.prod(axis=0)
    elif fusion == "cos":
      norms = np.linalg.norm(score_stack, axis=-1, keepdims=True)
      safe_norms = np.where(norms > 0, norms, 1.0)
      fused = (score_stack / safe_norms).mean(axis=0)
    else:
      raise ValueError(f"Unknown algebraic fusion '{fusion}'. Supported options are {_ALGEBRAIC_FUSIONS}.")

    fused = np.clip(fused, 0.0, None)
    row_sums = fused.sum(axis=-1, keepdims=True)
    safe_row_sums = np.where(row_sums > 0, row_sums, 1.0)
    normalized = fused / safe_row_sums
    zero_rows = (row_sums.squeeze(-1) == 0)
    if np.any(zero_rows):
      normalized[zero_rows] = 1.0 / fused.shape[-1]
    return normalized

  def _fuse_classifier_scores(self, score_tensor: np.ndarray) -> np.ndarray:
    """Dispatches to the algebraic or template-based fusion path
    according to `self.fusion`."""
    if self.fusion in _TEMPLATE_FUSIONS:
      if self.fusion == "dt":
        return _dt_fuse(score_tensor, self.templates_)
      if self.fusion == "ds":
        return _ds_fuse(score_tensor, self.templates_)
      return _ml_fuse(score_tensor, self.templates_, self.train_prevalence_)
    return self._algebraic_fuse(score_tensor, self.fusion)

  def _fit_classifier_ensemble(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fits every classifier and returns the fused Out-of-Fold score matrix.

    Populates `self.classifiers_` (fitted on the full training data) and,
    for template-based fusions, `self.templates_` (else `None`), as a
    side effect, ready for `_fuse_test_scores`.
    """
    jobs = [delayed(_oof_and_fit_job)(clf, X, y, self.cv) for clf in self.classifiers]
    results = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(jobs)
    oof_probas, fitted_classifiers = zip(*results)

    self.classifiers_ = list(fitted_classifiers)
    oof_tensor = np.stack(oof_probas, axis=0)  # (c, n_samples, n_classes)

    self.templates_ = (
      _compute_decision_templates(oof_tensor, y, self.classes_) if self.fusion in _TEMPLATE_FUSIONS else None
    )
    return self._fuse_classifier_scores(oof_tensor)

  def _fuse_test_scores(self, X: np.ndarray) -> np.ndarray:
    """Computes and fuses every fitted classifier's `predict_proba` for a test bag."""
    jobs = [delayed(_predict_proba_job)(clf, X) for clf in self.classifiers_]
    scores = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(jobs)
    score_tensor = np.stack(scores, axis=0)  # (c, n_samples, n_classes)
    return self._fuse_classifier_scores(score_tensor)

  @abstractmethod
  def fit(self, X: np.ndarray, y: np.ndarray):
    pass

  @abstractmethod
  def predict(self, X: np.ndarray) -> np.ndarray:
    pass


class FMCSQ(_BaseFMC):
  """Fusioned Multiple Classifiers with Single Quantifier (FMC-SQ).

  Fuses the probability scores of `c` independently trained classifiers
  into a single, row-normalized score matrix (`S̄ = N(F(S))`), then fits a
  single `quack` quantifier directly on that fused matrix, treating it as
  the quantifier's own classifier output (see `_IdentityScorer`) rather
  than as raw features.

  Parameters
  ----------
  classifiers : list of estimator objects
    The `c` classifiers to fuse. Each must implement `predict_proba`.
  quantifier : BaseQuantifier
    The single quantifier fit on the fused score matrix. Its own
    `classifier` parameter (if any) is ignored/substituted; see
    `_IdentityScorer`.
  fusion : {'mean', 'median', 'min', 'max', 'prod', 'cos', 'dt', 'ds', 'ml'}, default = 'mean'
    Fusion operator combining the `c` classifiers' score matrices. See
    `_BaseFMC` for a description of each option.
  cv : int, cross-validation generator or an iterable, default = 10
    Cross-validation strategy for each classifier's Out-of-Fold scoring.
  n_jobs : int, default = None
    Number of jobs to run in parallel across the `c` classifiers.
  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the classifier jobs.
  random_state : int, RandomState instance or None, default = None
    Reserved for future randomized fusion strategies.

  Attributes
  ----------
  classes_ : ndarray of shape (n_classes,)
    The distinct class labels found during the training phase.
  n_classes_ : int
    The total number of unique classes.
  train_prevalence_ : ndarray of shape (n_classes,)
    The baseline prevalence proportion of each class observed in training.
  classifiers_ : list of estimator objects
    The `c` fitted classifiers, refit on the full training data.
  templates_ : ndarray of shape (n_classes, c, n_classes) or None
    Decision templates (Kuncheva, 2004), populated only when
    `fusion` is `'dt'`, `'ds'`, or `'ml'`; `None` otherwise.
  quantifier_ : BaseQuantifier
    The fitted quantifier, trained on the fused Out-of-Fold score matrix.

  Notes
  -----
  The paper's evaluation used 7 classifiers (Random Forest, Naive Bayes,
  Gradient Boosting, SVM, LDA, LightGBM, Logistic Regression). `quack`
  intentionally does not depend on `lightgbm`; `sklearn.ensemble.
  HistGradientBoostingClassifier` is a drop-in, dependency-free
  substitute if you want to reproduce a similar committee.

  References
  ----------
  Serapião, A. B. S., Donyavi, Z., & Batista, G. (2023). Ensembles of
  Classifiers and Quantifiers with Data Fusion for Quantification
  Learning. In Discovery Science (DS 2023), LNAI 14276, pp. 3-17.
  Springer. https://doi.org/10.1007/978-3-031-45275-8_1

  Examples
  --------
  >>> from sklearn.datasets import make_classification
  >>> from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
  >>> from sklearn.linear_model import LogisticRegression
  >>> from quack.quantifiers import ACC
  >>> from quack.ensembles import FMCSQ
  >>> X, y = make_classification(n_samples=1000, n_classes=2, random_state=0)
  >>> ensemble = FMCSQ(
  ...   classifiers=[RandomForestClassifier(), HistGradientBoostingClassifier(), LogisticRegression()],
  ...   quantifier=ACC(cv=5), fusion="mean", random_state=0,
  ... )
  >>> ensemble.fit(X, y)
  >>> X_test, _ = make_classification(n_samples=200, n_classes=2, random_state=7)
  >>> prevalences = ensemble.predict(X_test)
  """

  def __init__(self,
               classifiers: list,
               quantifier: BaseQuantifier,
               fusion: str = "mean",
               cv: int = 10,
               n_jobs: int = None,
               parallel_backend: str = "loky",
               random_state=None):
    super().__init__(classifiers=classifiers, fusion=fusion, cv=cv,
                     n_jobs=n_jobs, parallel_backend=parallel_backend, random_state=random_state)
    self.quantifier = quantifier

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'FMCSQ':
    """Fits the classifier committee and the single downstream quantifier.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
      Training data.
    y : array-like of shape (n_samples,)
      Labels for the corresponding classes.

    Returns
    -------
    self : object
      Returns the fitted estimator instance itself.
    """
    X, y = check_X_y(X, y, accept_sparse=True)
    self._validate_fusion()

    self.classes_, counts = np.unique(y, return_counts=True)
    self.n_classes_ = len(self.classes_)
    self.train_prevalence_ = counts / len(y)

    fused_oof = self._fit_classifier_ensemble(X, y)

    self.quantifier_ = self._prepare_member(self.quantifier)
    self.quantifier_.fit(fused_oof, y)

    return self

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Estimates class prevalences for the test bag X.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
      The test bag with unlabelled instances.

    Returns
    -------
    prevalences : ndarray of shape (n_classes,)
      The estimated, normalized prevalence vector.
    """
    check_is_fitted(self)
    X = check_array(X, accept_sparse=True)

    fused_test = self._fuse_test_scores(X)
    prevalences = self.quantifier_.predict(fused_test)
    return normalize_prevalence(prevalences, self.n_classes_)


class FMCMQ(_BaseFMC):
  """Fusioned Multiple Classifiers with Multiple Quantifiers (FMC-MQ).

  Extends `FMCSQ` by fitting `q` independent quantifiers (of potentially
  different types) on the same fused, row-normalized classifier score
  matrix, then fuses their individual prevalence predictions (`N(F(P))`)
  into the final estimate — the paper's best-performing configuration.

  Parameters
  ----------
  classifiers : list of estimator objects
    The `c` classifiers to fuse. Each must implement `predict_proba`.
  quantifiers : list of BaseQuantifier
    The `q` quantifiers independently fit on the fused score matrix.
    Can mix different quantifier types (e.g. `[CC(), ACC(), GAC()]`).
  fusion : {'mean', 'median', 'min', 'max', 'prod', 'cos', 'dt', 'ds', 'ml'}, default = 'mean'
    Fusion operator combining the `c` classifiers' score matrices. See
    `_BaseFMC` for a description of each option.
  output_fusion : {'mean', 'median', 'min', 'max', 'prod'}, default = None
    Fusion operator combining the `q` quantifiers' individual prevalence
    predictions into the final estimate. Restricted to the 5 algebraic
    operators (the paper does not evaluate template-based combiners at
    the quantifier-output level). If None, reuses `fusion` — which must
    then itself be one of these 5 algebraic options.
  cv : int, cross-validation generator or an iterable, default = 10
    Cross-validation strategy for each classifier's Out-of-Fold scoring.
  n_jobs : int, default = None
    Number of jobs to run in parallel across the `c` classifiers and,
    separately, the `q` quantifiers.
  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the classifier and quantifier jobs.
  random_state : int, RandomState instance or None, default = None
    Reserved for future randomized fusion strategies.

  Attributes
  ----------
  classes_ : ndarray of shape (n_classes,)
    The distinct class labels found during the training phase.
  n_classes_ : int
    The total number of unique classes.
  train_prevalence_ : ndarray of shape (n_classes,)
    The baseline prevalence proportion of each class observed in training.
  classifiers_ : list of estimator objects
    The `c` fitted classifiers, refit on the full training data.
  templates_ : ndarray of shape (n_classes, c, n_classes) or None
    Decision templates, populated only when `fusion` is `'dt'`/`'ds'`/`'ml'`.
  quantifiers_ : list of BaseQuantifier
    The `q` fitted quantifiers, all trained on the same fused
    Out-of-Fold score matrix.

  Notes
  -----
  Every member quantifier is fit on (and predicts from) the exact same
  fused score matrix, so unlike `quack.ensembles.EoQ` there is no risk of
  a member missing a class.

  References
  ----------
  Serapião, A. B. S., Donyavi, Z., & Batista, G. (2023). Ensembles of
  Classifiers and Quantifiers with Data Fusion for Quantification
  Learning. In Discovery Science (DS 2023), LNAI 14276, pp. 3-17.
  Springer. https://doi.org/10.1007/978-3-031-45275-8_1

  Examples
  --------
  >>> from sklearn.datasets import make_classification
  >>> from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
  >>> from sklearn.linear_model import LogisticRegression
  >>> from quack.quantifiers import CC, ACC, GAC
  >>> from quack.ensembles import FMCMQ
  >>> X, y = make_classification(n_samples=1000, n_classes=2, random_state=0)
  >>> ensemble = FMCMQ(
  ...   classifiers=[RandomForestClassifier(), HistGradientBoostingClassifier(), LogisticRegression()],
  ...   quantifiers=[CC(), ACC(cv=5), GAC(cv=5)],
  ...   fusion="median", output_fusion="mean", random_state=0,
  ... )
  >>> ensemble.fit(X, y)
  >>> X_test, _ = make_classification(n_samples=200, n_classes=2, random_state=7)
  >>> prevalences = ensemble.predict(X_test)
  """

  def __init__(self,
               classifiers: list,
               quantifiers: list,
               fusion: str = "mean",
               output_fusion: str = None,
               cv: int = 10,
               n_jobs: int = None,
               parallel_backend: str = "loky",
               random_state=None):
    super().__init__(classifiers=classifiers, fusion=fusion, cv=cv,
                     n_jobs=n_jobs, parallel_backend=parallel_backend, random_state=random_state)
    self.quantifiers = quantifiers
    self.output_fusion = output_fusion

  def _validate_params(self):
    self._validate_fusion()
    if not self.quantifiers:
      raise ValueError("quantifiers must be a non-empty list of BaseQuantifier instances.")

    effective_output_fusion = self.output_fusion if self.output_fusion is not None else self.fusion
    if effective_output_fusion not in _VALID_OUTPUT_FUSIONS:
      if self.output_fusion is not None:
        raise ValueError(
          f"Unknown output_fusion '{self.output_fusion}'. Supported options are {_VALID_OUTPUT_FUSIONS}."
        )
      raise ValueError(
        f"output_fusion is None, so it defaults to fusion='{self.fusion}', but that is not one "
        f"of the algebraic operators supported at the quantifier-output level: {_VALID_OUTPUT_FUSIONS}. "
        "Set output_fusion explicitly to one of these."
      )

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'FMCMQ':
    """Fits the classifier committee and every member quantifier.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
      Training data.
    y : array-like of shape (n_samples,)
      Labels for the corresponding classes.

    Returns
    -------
    self : object
      Returns the fitted estimator instance itself.
    """
    X, y = check_X_y(X, y, accept_sparse=True)
    self._validate_params()

    self.classes_, counts = np.unique(y, return_counts=True)
    self.n_classes_ = len(self.classes_)
    self.train_prevalence_ = counts / len(y)

    fused_oof = self._fit_classifier_ensemble(X, y)

    prepared_members = [self._prepare_member(q) for q in self.quantifiers]
    fit_jobs = [delayed(_fit_quantifier_job)(member, fused_oof, y) for member in prepared_members]
    self.quantifiers_ = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(fit_jobs)

    return self

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Estimates class prevalences for the test bag X.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
      The test bag with unlabelled instances.

    Returns
    -------
    prevalences : ndarray of shape (n_classes,)
      The estimated, normalized prevalence vector.
    """
    check_is_fitted(self)
    X = check_array(X, accept_sparse=True)

    fused_test = self._fuse_test_scores(X)

    predict_jobs = [delayed(_predict_quantifier_job)(q, fused_test) for q in self.quantifiers_]
    member_predictions = np.array(Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(predict_jobs))

    output_fusion_op = self.output_fusion if self.output_fusion is not None else self.fusion
    aggregated = self._algebraic_fuse(member_predictions, output_fusion_op)

    return normalize_prevalence(aggregated, self.n_classes_)
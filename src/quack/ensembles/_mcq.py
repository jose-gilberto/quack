import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.parallel import Parallel, delayed

from quack.quantifiers.base import BaseQuantifier, normalize_prevalence

_VALID_FUSION_OUTPUTS = ("median", "mean", "prod", "min", "max")


def _merge_predictions(prediction_stack: np.ndarray, fusion: str, n_classes: int) -> np.ndarray:
  """Combines a stack of individual prevalence-vector predictions into one.

  `'median'` is renormalized afterward — the median of several unit-sum
  vectors is not itself guaranteed to sum to 1 (only guaranteed for
  n_classes == 2, per Pérez-Gállego et al.). `'mean'` needs no
  renormalization since averaging unit-sum vectors is linear.
  """
  if fusion == "median":
    return normalize_prevalence(np.median(prediction_stack, axis=0), n_classes)
  if fusion == "mean":
    return prediction_stack.mean(axis=0)
  if fusion == "min":
    return prediction_stack.min(axis=0)
  if fusion == "max":
    return prediction_stack.max(axis=0)
  if fusion == "prod":
    return prediction_stack.prod(axis=0)

  raise ValueError(f"Unknown fusion '{fusion}'. Supported options are {_VALID_FUSION_OUTPUTS}.")


def _clone_member(quantifier: BaseQuantifier, classifier) -> BaseQuantifier:
  """Clones a quantifier template and substitutes its own independent
  clone of `classifier`, so fitting one member never mutates another's
  (or the caller's) classifier instance — even when the same classifier
  object is reused across several entries in `classifiers`."""
  member = clone(quantifier)
  member.set_params(classifier=clone(classifier))
  return member


def _fit_mcsq_member_job(quantifier: BaseQuantifier, classifier, X: np.ndarray, y: np.ndarray) -> BaseQuantifier:
  """Fits one independent MCSQ member (one classifier x the shared quantifier template).

  Defined at module level (rather than as a closure/method) so it can be
  pickled and dispatched to worker processes by `joblib`/`Parallel`, the
  same pattern used throughout `quack.ensembles`/`quack.quantifiers.base`.
  """
  member = _clone_member(quantifier, classifier)
  member.fit(X, y)
  return member


def _predict_member_job(member: BaseQuantifier, X: np.ndarray) -> np.ndarray:
  """Scores one test bag against a single fitted quantifier member."""
  return member.predict(X)


def _fit_scmq_group_job(quantifiers: list, classifier, X: np.ndarray, y: np.ndarray) -> list:
  """Fits the `q` quantifier templates for a single classifier (one MCMQ group).

  The `q` members within a group are fit sequentially — only the `c`
  groups themselves are dispatched as parallel jobs (see `MCMQ.fit`), to
  avoid nested `joblib.Parallel` pools.
  """
  members = []
  for template in quantifiers:
    member = _clone_member(template, classifier)
    member.fit(X, y)
    members.append(member)
  return members


def _predict_scmq_group_job(members: list, X: np.ndarray, fusion: str, n_classes: int) -> np.ndarray:
  """Scores a test bag against every member of one MCMQ group and merges
  their `q` predictions into a single per-classifier prevalence estimate."""
  predictions = np.array([member.predict(X) for member in members])
  return _merge_predictions(predictions, fusion, n_classes)


class MCSQ(BaseQuantifier):
  """Multiple Classifiers with a Single Quantifier template (MCSQ).

  Fits one independent copy of `quantifier` per classifier in
  `classifiers` (each with its own classifier substituted in), then
  merges the `c` independent prevalence predictions at the output level
  (median or mean) — as opposed to `quack.ensembles.FMCSQ`, which fuses
  classifier *scores* into one matrix before a single quantifier ever
  sees them. Every member is trained on the exact same `(X, y)`, so
  (unlike `quack.ensembles.EoQ`) there is no risk of a member missing a
  class and no prediction-alignment step is needed.

  Parameters
  ----------
  classifiers : list of estimator objects
    The `c` classifiers; one independent quantifier member is fitted per
    classifier.
  quantifier : BaseQuantifier
    The quantifier template cloned once per classifier. Must expose a
    `classifier` parameter (e.g. `CC`, `PCC`, `ACC`, `PACC`, `HDy`,
    `DyS`, `GAC`, `GPAC`, `FM`, `EM`, `CDE`, any of the threshold
    selectors) so its base classifier can be substituted per member.
  fusion : {'median', 'mean'}, default = 'median'
    How the `c` members' individual prevalence predictions are combined.
  n_jobs : int, default = None
    Number of jobs to run in parallel while fitting/predicting the `c`
    independent members, since none of them depend on each other. `None`
    means sequential; `-1` uses all available processors.
  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the member jobs.

  Attributes
  ----------
  classes_ : ndarray of shape (n_classes,)
    The distinct class labels found during the training phase.
  n_classes_ : int
    The total number of unique classes.
  train_prevalence_ : ndarray of shape (n_classes,)
    The prevalence of each class in the full training dataset.
  quantifiers_ : list of BaseQuantifier
    The `c` fitted quantifier members, one per classifier in `classifiers`.

  References
  ----------
  Z. Donyavi, A. B. S. Serapião and G. Batista, "MC-SQ and MC-MQ: Ensembles for
  Multi-Class Quantification," in IEEE Transactions on Knowledge and Data
  Engineering, vol. 36, no. 8, pp. 4007-4019, Aug. 2024.

  Examples
  --------
  >>> from sklearn.datasets import make_classification
  >>> from sklearn.ensemble import RandomForestClassifier
  >>> from sklearn.linear_model import LogisticRegression
  >>> from quack.quantifiers import ACC
  >>> from quack.ensembles import MCSQ
  >>> X, y = make_classification(n_samples=1000, n_classes=2, random_state=0)
  >>> ensemble = MCSQ(
  ...   classifiers=[RandomForestClassifier(), LogisticRegression()],
  ...   quantifier=ACC(cv=5), fusion="median",
  ... )
  >>> ensemble.fit(X, y)
  >>> X_test, _ = make_classification(n_samples=200, n_classes=2, random_state=7)
  >>> prevalences = ensemble.predict(X_test)
  """

  def __init__(self,
               classifiers: list,
               quantifier: BaseQuantifier,
               fusion: str = "median",
               n_jobs: int = None,
               parallel_backend: str = "loky"):
    super().__init__(classifier=None)
    self.classifiers = classifiers
    self.quantifier = quantifier
    self.fusion = fusion
    self.n_jobs = n_jobs
    self.parallel_backend = parallel_backend

  def _validate_params(self):
    if self.fusion not in _VALID_FUSION_OUTPUTS:
      raise ValueError(f"Unknown fusion '{self.fusion}'. Supported options are {_VALID_FUSION_OUTPUTS}.")
    if not self.classifiers:
      raise ValueError("classifiers must be a non-empty list of estimators.")
    if not hasattr(self.quantifier, "classifier"):
      raise TypeError(
        f"{self.quantifier.__class__.__name__} does not expose a 'classifier' parameter; "
        "MCSQ requires a quantifier template whose base classifier can be substituted per member."
      )

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'MCSQ':
    """Fits one independent quantifier member per classifier in `classifiers`.

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

    fit_jobs = [delayed(_fit_mcsq_member_job)(self.quantifier, clf, X, y) for clf in self.classifiers]
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
      The merged, normalized prevalence estimate.
    """
    check_is_fitted(self)
    X = check_array(X, accept_sparse=True)

    predict_jobs = [delayed(_predict_member_job)(member, X) for member in self.quantifiers_]
    prediction_stack = np.array(Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(predict_jobs))

    merged = _merge_predictions(prediction_stack, self.fusion, self.n_classes_)
    return normalize_prevalence(merged, self.n_classes_)


class MCMQ(BaseQuantifier):
  """Multiple Classifiers with Multiple Quantifiers (MCMQ).

  Extends `MCSQ` to a full `c x q` grid: for each of the `c` classifiers,
  every one of the `q` quantifier templates is independently fitted with
  that classifier substituted in, and the `q` resulting predictions are
  first merged (within-classifier, via `fusion`) into one prevalence
  estimate per classifier; those `c` per-classifier estimates are then
  merged again (across classifiers, via the same `fusion`) into the
  final prediction. Unlike `quack.ensembles.FMCMQ`, no classifier-score
  fusion happens anywhere in the pipeline — every merge is done purely at
  the prevalence-output level.

  Parameters
  ----------
  classifiers : list of estimator objects
    The `c` classifiers.
  quantifiers : list of BaseQuantifier
    The `q` quantifier templates, each cloned and fit once per
    classifier. Every template must expose a `classifier` parameter.
  fusion : {'median', 'mean'}, default = 'median'
    How predictions are combined, both within a classifier's `q`
    quantifiers and across the `c` classifiers.
  n_jobs : int, default = None
    Number of jobs to run in parallel across the `c` independent
    classifier groups (fitting/predicting), since none of them depend on
    each other. The `q` quantifiers within a single group are always fit
    sequentially, to avoid nested `joblib.Parallel` pools. `None` means
    sequential; `-1` uses all available processors.
  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the per-classifier group jobs.

  Attributes
  ----------
  classes_ : ndarray of shape (n_classes,)
    The distinct class labels found during the training phase.
  n_classes_ : int
    The total number of unique classes.
  train_prevalence_ : ndarray of shape (n_classes,)
    The prevalence of each class in the full training dataset.
  member_groups_ : list of list of BaseQuantifier
    `member_groups_[i]` holds the `q` fitted quantifiers built on top of
    `classifiers[i]`.

  References
  ----------
  Z. Donyavi, A. B. S. Serapião and G. Batista, "MC-SQ and MC-MQ: Ensembles for
  Multi-Class Quantification," in IEEE Transactions on Knowledge and Data
  Engineering, vol. 36, no. 8, pp. 4007-4019, Aug. 2024.

  Examples
  --------
  >>> from sklearn.datasets import make_classification
  >>> from sklearn.ensemble import RandomForestClassifier
  >>> from sklearn.linear_model import LogisticRegression
  >>> from quack.quantifiers import CC, ACC, PCC
  >>> from quack.ensembles import MCMQ
  >>> X, y = make_classification(n_samples=1000, n_classes=2, random_state=0)
  >>> ensemble = MCMQ(
  ...   classifiers=[RandomForestClassifier(), LogisticRegression()],
  ...   quantifiers=[CC(), ACC(cv=5), PCC()],
  ...   fusion="median",
  ... )
  >>> ensemble.fit(X, y)
  >>> X_test, _ = make_classification(n_samples=200, n_classes=2, random_state=7)
  >>> prevalences = ensemble.predict(X_test)
  """

  def __init__(self,
               classifiers: list,
               quantifiers: list,
               fusion: str = "median",
               n_jobs: int = None,
               parallel_backend: str = "loky"):
    super().__init__(classifier=None)
    self.classifiers = classifiers
    self.quantifiers = quantifiers
    self.fusion = fusion
    self.n_jobs = n_jobs
    self.parallel_backend = parallel_backend

  def _validate_params(self):
    if self.fusion not in _VALID_FUSION_OUTPUTS:
      raise ValueError(f"Unknown fusion '{self.fusion}'. Supported options are {_VALID_FUSION_OUTPUTS}.")
    if not self.classifiers:
      raise ValueError("classifiers must be a non-empty list of estimators.")
    if not self.quantifiers:
      raise ValueError("quantifiers must be a non-empty list of quantifier templates.")
    for template in self.quantifiers:
      if not hasattr(template, "classifier"):
        raise TypeError(
          f"{template.__class__.__name__} does not expose a 'classifier' parameter; "
          "MCMQ requires every quantifier template's base classifier to be substitutable."
        )

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'MCMQ':
    """Fits `q` quantifier members for each of the `c` classifiers.

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

    fit_jobs = [delayed(_fit_scmq_group_job)(self.quantifiers, clf, X, y) for clf in self.classifiers]
    self.member_groups_ = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(fit_jobs)

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
      The doubly-merged, normalized prevalence estimate.
    """
    check_is_fitted(self)
    X = check_array(X, accept_sparse=True)

    predict_jobs = [
      delayed(_predict_scmq_group_job)(members, X, self.fusion, self.n_classes_)
      for members in self.member_groups_
    ]
    group_predictions = np.array(Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(predict_jobs))

    merged = _merge_predictions(group_predictions, self.fusion, self.n_classes_)
    return normalize_prevalence(merged, self.n_classes_)
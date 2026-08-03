import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.parallel import Parallel, delayed

from quack.quantifiers.base import BaseQuantifier, normalize_prevalence
from quack.bag_generator.base import BaseBagGenerator
from quack.bag_generator._prior_shift import PriorShiftBagGenerator
from quack.metrics import MetricRegistry
from quack.metrics.base import QuantificationMetric

# in the 2019 paper the name of these policies are
# [average, p_tr, acc]
# TODO: include MAX and DS from the original paper, mainly because DS has the best performance
# compared to the other two methods
_VALID_SELECTION_METHODS = ("average", "ptr", "performance")


def _top_up_missing_classes(X_bag: np.ndarray,
                            y_bag: np.ndarray,
                            X_pool: np.ndarray,
                            class_pools: dict,
                            required_classes: np.ndarray,
                            rng) -> tuple[np.ndarray, np.ndarray]:
  """Guarantees a bag contains at least one instance of every class in
  `required_classes`, by construction rather than by chance.

  Extreme prevalence/covariate shift sampling can legitimately draw a bag
  missing one or more classes entirely — that is the whole point of
  sampling near the edges of the prevalence simplex — but most base
  quantifiers cannot be `.fit()` on data missing a class (e.g.
  `sklearn.linear_model.LogisticRegression` requires at least 2 classes).

  Rather than re-drawing the whole bag and hoping for a luckier sample
  (which, for a sufficiently extreme shift configuration, may need an
  arbitrarily large or even unbounded number of attempts to succeed by
  chance), this deterministically swaps in one randomly drawn instance of
  each missing class, each time replacing an instance from the bag's
  current largest class. This always succeeds in a single pass, as long
  as every required class has at least one instance in `X_pool` (which
  `_generate_valid_bags` checks upfront), and only perturbs the handful
  of missing classes — everything else in the bag is left exactly as
  sampled, so the requested shift is preserved as closely as possible.

  Returns
  -------
  X_bag, y_bag : ndarray, ndarray
    The (possibly amended) bag, with the same shape as the input.
  """
  present_classes, counts = np.unique(y_bag, return_counts=True)
  missing_classes = np.setdiff1d(required_classes, present_classes)
  if missing_classes.size == 0:
    return X_bag, y_bag

  X_bag = X_bag.copy()
  y_bag = y_bag.copy()
  count_map = dict(zip(present_classes.tolist(), counts.tolist()))

  for missing_class in missing_classes:
    donor_idx = rng.choice(class_pools[missing_class])
    largest_class = max(count_map, key=count_map.get)
    replace_pos = rng.choice(np.flatnonzero(y_bag == largest_class))

    X_bag[replace_pos] = X_pool[donor_idx]
    y_bag[replace_pos] = missing_class

    count_map[largest_class] -= 1
    count_map[missing_class] = count_map.get(missing_class, 0) + 1

  return X_bag, y_bag


def _generate_valid_bags(bag_generator: BaseBagGenerator,
                         X: np.ndarray,
                         y: np.ndarray,
                         required_classes: np.ndarray,
                         random_state) -> tuple[list, np.ndarray]:
  """Generates `bag_generator.n_bags` bags, topping up any bag that is
  missing a required class so every returned bag is fittable by a
  standard classifier.

  Returns
  -------
  bags : list[tuple[np.ndarray, np.ndarray]]
    The `n_bags` bags, each containing every class in `required_classes`.
  sampled_prevalences : ndarray of shape (n_bags, n_classes)
    The *actual* realized prevalence of each returned bag, recomputed
    after any top-up (so `selection_policy='ptr'`, which relies on this
    array, reflects the true training composition rather than the
    pre-top-up sampled target).

  Raises
  ------
  ValueError
    If any required class has zero instances in `X`/`y` altogether, or
    if the bag size is smaller than the number of required classes
    (physically impossible to satisfy regardless of top-up).
  """
  rng = check_random_state(random_state)

  class_pools = {c: np.flatnonzero(y == c) for c in required_classes}
  empty_classes = [c for c in required_classes if class_pools[c].size == 0]
  if empty_classes:
    raise ValueError(
      f"Classes {empty_classes} have zero instances in the provided data; "
      "cannot guarantee every training bag contains all required classes."
    )

  generator = clone(bag_generator)
  generator.random_state = rng.randint(np.iinfo(np.int32).max)
  bags = generator.to_list(X, y)

  bag_size = bags[0][0].shape[0] if bags else 0
  if bag_size < len(required_classes):
    raise ValueError(
      f"bag_size ({bag_size}) is smaller than the number of required classes "
      f"({len(required_classes)}); it is impossible for a bag that small to "
      "contain every class. Increase bag_size on the bag generator."
    )

  sampled_prevalences = np.zeros((len(bags), len(required_classes)))
  for i, (X_bag, y_bag) in enumerate(bags):
    X_bag, y_bag = _top_up_missing_classes(X_bag, y_bag, X, class_pools, required_classes, rng)
    bags[i] = (X_bag, y_bag)
    sampled_prevalences[i] = np.array([np.mean(y_bag == c) for c in required_classes])

  return bags, sampled_prevalences


def _align_prediction(prediction: np.ndarray, member_classes, full_classes: np.ndarray) -> np.ndarray:
  """Scatters a member's prediction (over its own, possibly narrower, set
  of classes) into a full-length vector aligned to `full_classes`,
  filling any class the member never saw during fit with 0.0.

  Kept as a defensive measure even though `_generate_valid_bags` already
  guarantees every training bag contains all classes, in case a member's
  own `.fit()` drops a class for some other reason.
  """
  if member_classes is None:
    return prediction
  member_classes = np.asarray(member_classes)
  if member_classes.shape == full_classes.shape and np.array_equal(member_classes, full_classes):
    return prediction
  aligned = np.zeros(full_classes.shape[0])
  positions = np.searchsorted(full_classes, member_classes)
  aligned[positions] = prediction
  return aligned


def _fit_member_job(base_quantifier: BaseQuantifier, X_bag: np.ndarray, y_bag: np.ndarray) -> BaseQuantifier:
  """Fits one independent ensemble member on its own resampled bag.

  Defined at module level (rather than as a closure/method) so it can be
  pickled and dispatched to worker processes by `joblib`/`Parallel`, the
  same pattern used for the per-fold jobs in `quack.quantifiers.base` and
  the per-subspace jobs in `quack.quantifiers._features.ReadMe`.
  """
  member = clone(base_quantifier)
  member.fit(X_bag, y_bag)
  return member


def _predict_member_job(member: BaseQuantifier, X: np.ndarray, full_classes: np.ndarray) -> np.ndarray:
  """Scores one test bag against a single fitted ensemble member,
  aligning the result to the full ensemble class space.

  Defined at module level for the same pickling reasons as `_fit_member_job`.
  """
  prediction = member.predict(X)
  return _align_prediction(prediction, getattr(member, "classes_", None), full_classes)


def _mean_error_job(member: BaseQuantifier,
                    metric: QuantificationMetric,
                    val_bags: list,
                    val_true_prevalences: np.ndarray,
                    full_classes: np.ndarray) -> float:
  """Evaluates one ensemble member's mean quantification error across
  every validation bag, used by the `'performance'` selection policy."""
  errors = []
  for i, (X_val_bag, _) in enumerate(val_bags):
    prediction = _align_prediction(member.predict(X_val_bag), getattr(member, "classes_", None), full_classes)
    errors.append(metric(val_true_prevalences[i], prediction))
  return float(np.mean(errors))


class EoQ(BaseQuantifier):
  """Ensemble of Quantifiers (EoQ).

  Trains `n_estimators` independent copies of `base_quantifier`, each on
  a training bag with an artificially shifted class prevalence (drawn via
  `bag_generator`, defaulting to `PriorShiftBagGenerator`), and aggregates
  their individual predictions into a single, typically more robust,
  prevalence estimate. This directly reuses `quack.bag_generator` for the
  Artificial Prevalence Protocol (APP) resampling described in the
  reference papers, rather than reimplementing bootstrap/prevalence
  sampling from scratch.

  Three aggregation strategies are supported via `selection_policy`:

  - `'average'`: simple average of every member's prediction (Pérez-Gállego
    et al., 2017's baseline ensemble).
  - `'performance'`: static selection. A held-out validation split
    (`val_split`) is used to generate validation bags with known
    prevalence; each member's mean quantification error on them (via
    `metric`) is computed once during `fit`, and only the `red_size`
    lowest-error members are kept for every subsequent `predict` call.
  - `'ptr'` (Training Prevalence): dynamic selection (Pérez-Gállego et al.,
    2019). For each test bag, a preliminary prevalence estimate is formed
    by averaging every member's prediction; only the `red_size` members
    whose *own training bag* prevalence is closest (Euclidean distance) to
    that estimate are then re-averaged into the final prediction. Because
    the selection depends on the specific test bag, it is recomputed on
    every `predict` call.

  Parameters
  ----------
  base_quantifier : BaseQuantifier
    The quantifier prototype cloned and independently fitted for each
    ensemble member.
  n_estimators : int, default = 30
    Number of ensemble members to train.
  bag_generator : BaseBagGenerator, default = None
    Bag generator used to resample each member's training bag (and, for
    `selection_policy='performance'`, the validation bags). If None,
    defaults to `PriorShiftBagGenerator(sampling_strategy='uniform')`.
    Its `n_bags` is overridden internally; any other parameter (e.g.
    `bag_size`) set on the instance you pass in is preserved. Every
    resulting training bag is guaranteed to contain all training classes
    (see Notes).
  selection_policy : {'average', 'ptr', 'performance'}, default = 'average'
    Aggregation/selection strategy, as described above.
  red_size : int, default = None
    Number of members retained by `'ptr'`/`'performance'`. Required
    (and must not exceed `n_estimators`) for those two policies; unused
    for `'average'`.
  metric : str | QuantificationMetric, default = 'ae'
    Quantification error metric used only by `selection_policy='performance'`
    to rank members. Accepts any key registered in
    `quack.metrics.MetricRegistry` (e.g. `'ae'`, `'kld'`) or a
    `QuantificationMetric` instance directly.
  val_split : float, default = 0.4
    Fraction of the training data held out (stratified) to build
    validation bags for `selection_policy='performance'`. Unused by the
    other two policies, which train every member on the full dataset.
  n_val_samples : int, default = None
    Number of validation bags generated for `selection_policy='performance'`.
    If None, defaults to `n_estimators`.
  n_jobs : int, default = None
    Number of jobs to run in parallel while fitting/predicting/scoring
    the `n_estimators` independent members, since none of them depend on
    each other. `None` means sequential; `-1` uses all available
    processors. See `joblib.Parallel`.
  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the member jobs.
  random_state : int, RandomState instance or None, default = None
    Controls the randomness of every bag resampling step (training and,
    when applicable, validation bags, plus the held-out split for
    `selection_policy='performance'`, plus the top-up mechanism below).

  Attributes
  ----------
  classes_ : ndarray of shape (n_classes,)
    The distinct class labels found during the training phase.
  n_classes_ : int
    The total number of unique classes.
  train_prevalence_ : ndarray of shape (n_classes,)
    The prevalence of each class in the full training dataset.
  estimators_ : list of BaseQuantifier
    The `n_estimators` fitted ensemble members. Every member is
    guaranteed to have been fit on a bag containing all `n_classes_`
    classes.
  member_train_prevalences_ : ndarray of shape (n_estimators, n_classes)
    The *actual* realized training-bag prevalence of each member (after
    any class top-up, see Notes); used by `selection_policy='ptr'`.
  oob_scores_ : ndarray of shape (n_estimators,) or None
    Mean validation error of each member, only populated when
    `selection_policy='performance'`; None otherwise.
  selected_indices_ : ndarray of int
    Indices into `estimators_` kept for `selection_policy='performance'`
    (static, decided in `fit`); for `'average'`, this is every member's
    index; unused for `'ptr'` (recomputed per test bag in `predict`).

  Extreme prevalence/covariate shift configurations can legitimately draw
  a training bag missing one or more classes entirely — that is precisely
  the point of sampling near the edges of the prevalence simplex — but
  most base quantifiers cannot be `.fit()` on data missing a class. Rather
  than re-drawing the whole bag and hoping for a luckier sample (which may
  need arbitrarily many attempts, or never succeed, for a sufficiently
  extreme configuration), every training bag is deterministically
  "topped up": any missing class has one instance swapped in from the
  bag's current largest class. This always succeeds in a single pass and
  only perturbs the handful of missing classes, so `member_train_prevalences_`
  reflects the bag's true final composition (used as-is by
  `selection_policy='ptr'`), which may differ marginally from the
  originally sampled target for very extreme shift configurations.

  References
  ----------
  Pérez-Gállego, P., Quevedo, J. R., & del Coz, J. J. (2017). Using
  ensembles for problems with characterizable changes in data
  distribution: A case study on quantification. Information Fusion, 34,
  87-100.

  Pérez-Gállego, P., Castaño, A., Quevedo, J. R., & del Coz, J. J. (2019).
  Dynamic ensemble selection for quantification tasks. Information
  Fusion, 45, 1-15.

  Examples
  --------
  >>> from sklearn.datasets import make_classification
  >>> from sklearn.linear_model import LogisticRegression
  >>> from quack.quantifiers import CC
  >>> from quack.ensembles import EoQ
  >>> X, y = make_classification(n_samples=1000, n_classes=2, random_state=0)
  >>> ensemble = EoQ(CC(LogisticRegression(max_iter=1000)), n_estimators=30,
  ...                selection_policy="average", random_state=0)
  >>> ensemble.fit(X, y)
  >>> X_test, _ = make_classification(n_samples=200, n_classes=2, random_state=7)
  >>> prevalences = ensemble.predict(X_test)
  """

  def __init__(self,
               base_quantifier: BaseQuantifier,
               n_estimators: int = 30,
               bag_generator: BaseBagGenerator = None,
               selection_policy: str = "average",
               red_size: int = None,
               metric: str = "ae",
               val_split: float = 0.4,
               n_val_samples: int = None,
               n_jobs: int = None,
               parallel_backend: str = "loky",
               random_state=None):
    super().__init__(classifier=None)
    self.base_quantifier = base_quantifier
    self.n_estimators = n_estimators
    self.bag_generator = bag_generator
    self.selection_policy = selection_policy
    self.red_size = red_size
    self.metric = metric
    self.val_split = val_split
    self.n_val_samples = n_val_samples
    self.n_jobs = n_jobs
    self.parallel_backend = parallel_backend
    self.random_state = random_state

  def _validate_params(self):
    if self.selection_policy not in _VALID_SELECTION_METHODS:
      raise ValueError(
        f"Unknown selection_policy '{self.selection_policy}'. "
        f"Supported options are {_VALID_SELECTION_METHODS}."
      )
    if self.n_estimators <= 0:
      raise ValueError(f"n_estimators must be a positive integer, got {self.n_estimators}.")
    if self.selection_policy in ("ptr", "performance"):
      if self.red_size is None:
        raise ValueError(
          f"selection_policy='{self.selection_policy}' requires red_size to be set "
          "(number of ensemble members to retain after selection)."
        )
      if not (0 < self.red_size <= self.n_estimators):
        raise ValueError(
          f"red_size must satisfy 0 < red_size <= n_estimators ({self.n_estimators}), "
          f"got {self.red_size}."
        )

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'EoQ':
    """Fits every ensemble member on an independently resampled training bag.

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

    base_bag_generator = self.bag_generator if self.bag_generator is not None else PriorShiftBagGenerator(
      sampling_strategy="uniform",
    )

    if self.selection_policy == "performance":
      X_train_pool, X_val_pool, y_train_pool, y_val_pool = train_test_split(
        X, y, test_size=self.val_split, stratify=y, random_state=self.random_state,
      )
    else:
      X_train_pool, y_train_pool = X, y
      X_val_pool = y_val_pool = None

    train_generator = clone(base_bag_generator)
    train_generator.n_bags = self.n_estimators

    train_bags, self.member_train_prevalences_ = _generate_valid_bags(
      train_generator, X_train_pool, y_train_pool, self.classes_, self.random_state,
    )

    fit_jobs = [delayed(_fit_member_job)(self.base_quantifier, X_bag, y_bag) for X_bag, y_bag in train_bags]
    self.estimators_ = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(fit_jobs)

    if self.selection_policy == "performance":
      metric = self.metric if isinstance(self.metric, QuantificationMetric) else MetricRegistry.get(self.metric)

      val_generator = clone(base_bag_generator)
      val_generator.n_bags = self.n_val_samples if self.n_val_samples is not None else self.n_estimators
      val_generator.random_state = self.random_state

      val_bags = val_generator.to_list(X_val_pool, y_val_pool)
      val_true_prevalences = val_generator.sampled_prevalences_

      score_jobs = [
        delayed(_mean_error_job)(member, metric, val_bags, val_true_prevalences, self.classes_)
        for member in self.estimators_
      ]
      self.oob_scores_ = np.array(Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(score_jobs))

      order = np.argsort(self.oob_scores_)
      if not metric.lower_is_better:
        order = order[::-1]
      self.selected_indices_ = np.sort(order[:self.red_size])
    else:
      self.oob_scores_ = None
      self.selected_indices_ = np.arange(self.n_estimators)

    return self

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Aggregates every ensemble member's prediction for the test bag X.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
      The test bag with unlabelled instances.

    Returns
    -------
    p_adjusted : ndarray of shape (n_classes,)
      The aggregated, normalized prevalence estimate.
    """
    check_is_fitted(self)
    X = check_array(X, accept_sparse=True)

    predict_jobs = [delayed(_predict_member_job)(member, X, self.classes_) for member in self.estimators_]
    member_predictions = np.array(Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(predict_jobs))

    if self.selection_policy == "average":
      aggregated = member_predictions.mean(axis=0)

    elif self.selection_policy == "performance":
      aggregated = member_predictions[self.selected_indices_].mean(axis=0)

    else:  # "ptr": dynamic selection, recomputed for this specific test bag
      p_test_estimate = member_predictions.mean(axis=0)
      distances = np.linalg.norm(self.member_train_prevalences_ - p_test_estimate, axis=1)
      selected = np.argsort(distances)[:self.red_size]
      aggregated = member_predictions[selected].mean(axis=0)

    return normalize_prevalence(aggregated, self.n_classes_)
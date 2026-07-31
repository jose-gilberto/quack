import numpy as np
import math
import warnings
import cvxpy as cvx
from abc import ABC, abstractmethod
from typing import TypeVar
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import check_cv
from sklearn.utils.parallel import Parallel, delayed
from sklearn.linear_model import LogisticRegression


def normalize_prevalence(p: np.ndarray, n_classes: int) -> np.ndarray:
  """Clips negative noise from a raw prevalence-like vector and renormalizes it to sum to 1.0.

  Shared by every quantifier's `.predict()` so the exact same geometric
  post-processing rule is applied everywhere (`CC`, `PCC`,
  `BaseCalibratedQuantifier`, ...) instead of being copy-pasted with a
  risk of silently drifting between implementations.

  Only the lower bound (`0.0`) is enforced here, as a guard against
  floating-point noise producing tiny negative values (e.g. from ACC's
  subtraction-based adjustment formula). No upper bound is applied to
  individual components before normalizing: inputs are not guaranteed to
  be per-component probabilities in `[0, 1]` (e.g. `CC` passes raw class
  counts, which can be arbitrarily larger than 1) — the division by the
  total sum is what enforces the `[0, 1]` range and the sum-to-1
  constraint on the *output*, not a per-component clip on the input.

  Parameters
  ----------
  p: np.ndarray
    Raw prevalence-like vector of shape `(n_classes,)`
    (e.g. class counts, averaged probabilities, or an optimizer's
    output), not necessarily non-negative or summing to 1.0.
  n_classes: int
    Number of classes, used for the uniform fallback
    when `p` collapses entirely to zero/negative mass.

  Returns
  -------
  probs: np.ndarray
    A valid probability vector of shape `(n_classes,)`.
  """
  p = np.clip(p, 0.0, None)
  p_sum = np.sum(p)
  if p_sum > 0:
    return p / p_sum
  return np.ones(n_classes) / n_classes


def _clone_fit_predict_fold(base_classifier: BaseEstimator,
                            X: np.ndarray,
                            y: np.ndarray,
                            train_idx: np.ndarray,
                            test_idx: np.ndarray,
                            oof_method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Fits a cloned classifier on one CV fold and predicts on its held-out split.

  Defined at module level (rather than as a closure/method) so it can be
  pickled and dispatched to worker processes by `joblib`/`Parallel`.

  Returns
  -------
  test_idx : ndarray
    The held-out indices this fold predicted for (echoed back so results
    can be scattered into the right positions after parallel execution).
  y_test : ndarray
    True labels for `test_idx`.
  y_pred : ndarray
    Out-of-fold predictions (`predict` or `predict_proba` output) for `test_idx`.
  """
  fold_classifier = clone(base_classifier)
  fold_classifier.fit(X[train_idx], y[train_idx])
  predict_fn = getattr(fold_classifier, oof_method)
  return test_idx, y[test_idx], predict_fn(X[test_idx])


def _clone_fit_full(base_classifier: BaseEstimator, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
  """Fits a cloned classifier on the entire dataset.

  Defined at module level so it can run as an independent `joblib` task
  alongside the per-fold jobs (it does not depend on their results).
  """
  classifier = clone(base_classifier)
  classifier.fit(X, y)
  return classifier


class QuantifierMixin:
  """Mixin to identify that this class belongs to quantifiers family."""
  _estimator_type = "quantifier"


class BaseQuantifier(BaseEstimator, QuantifierMixin, ABC):
  """
  Abstract class for all quantifiers. Inherits from BaseEstimator and QuantifierMixin
  to guarantee compatibility with scikit-learn, as clone of estimators, get or set
  parameters (get_parameter/set_parameter) and the integration with GridSeach (GridSearchCV).

  Parameters
  ----------
  classifier: estimator object, default = None
    The classifier that will be used as a base for the quantifier.
    If its None, an instance of `LogisticRegression()` will be created.
  """

  def __init__(self, classifier: BaseEstimator = None):
    self.classifier = classifier

  @abstractmethod
  def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseQuantifier':
    """ Adjusts the quantifier based on the training data.

    Parameters
    ----------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
      Training data.
    y: array-like of shape (n_samples,)
      Labels for the corresponding classes.

    Returns
    -------
    self: object
      Returns the fitted estimator instance itself.
    """
    pass

  @abstractmethod
  def predict(self, X: np.ndarray) -> np.ndarray:
    """ Estimate the prevalences (distribution) for the test bag X.

    Parameters
    ----------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
      The test bag with unlabelled instances.

    Returns
    -------
    prevalences: ndarray of shape (n_classes,)
      An array with the estimated prevalences for each class.
      The sum of the elements in this array must be equals to 1.0.
    """
    pass


class BaseCalibratedQuantifier(BaseQuantifier, ABC):
  """
  Base class to quantifiers that requires cross-validation Out-of-Fold.

  This class manages automatically the pipeline for training with cross-validation
  needed to create the calibration matrices (or confusion matrix) avoiding the
  overfitting.

  Parameters
  ----------
  classifiers: estimator object, default = None
    The classifier to be internally adjusted. If its None, adopts the `LogisticRegressor()`.
  cv: int, cross-validation generator or an iterable, default = 10
    Determine the strategy for cross validation to generate the predictions Out-of-Fold.
    Options include int (number of folds for `StratifiedKFold`), a generator for CV or an interable
    with custom splits.
  n_jobs: int, default = None
    Number of jobs to run in parallel while fitting the `cv` folds (plus the final
    classifier refit on the whole training set, dispatched as one extra independent
    job). `None` means 1 (sequential, matching the previous behavior); `-1` means
    using all available processors. See `joblib.Parallel` for details.
  parallel_backend: str, default = "loky"
    Backend passed through to `joblib.Parallel` (e.g. `"loky"` for process-based
    parallelism, `"threading"` for thread-based parallelism, the latter is
    preferable when the base classifier releases the GIL during `fit`/`predict`,
    such as most scikit-learn estimators backed by Cython/BLAS).
  """

  def __init__(self, classifier: BaseEstimator = None, cv: int = 10,
               n_jobs: int = None, parallel_backend: str = "loky"):
    super().__init__(classifier=classifier)
    self.cv = cv
    self.n_jobs = n_jobs
    self.parallel_backend = parallel_backend

  @abstractmethod
  def _get_oof_method(self) -> str:
    """ Determine which method from the classifier will be used to generate the
    predictions.

    Subclasses must return valid strings like `"predict"` or `"predict_proba"`.

    Returns
    -------
    method_name: str
      Name of the method of the base estimator.
    """
    pass
  
  @abstractmethod
  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    """ Internal method to build structures for calibration (matrix CM).

    Parameters
    ----------
    y_true_oof: ndarray of shape (n_samples,)
      True labels colected out-of-fold.
    y_pred_oof: ndarray of shape (n_samples, n_classes) or (n_samples,)
      Predictions (continuous or crisp) generated out-of-fold.
    """
    pass
  
  @abstractmethod
  def _quantify(self, X: np.ndarray) -> np.ndarray:
    """ End-to-end algorithm for prevalence estimation specific for each subclass.

    Parameters
    ----------
    X: ndarray of shape (n_samples, n_features)
      Raw data from test bags

    Returns
    -------
    prevalences: ndarray of shape (n_classes,)
      Prevalences estimated by the method.
    """
    pass

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseCalibratedQuantifier':
    """ Adjust the complete quantification pipeline using predictions out-of-fold.
    
    Executes the cross-validation for internal calibration and, following, adjust
    the model with all.

    The `n_splits` per-fold fits and the final full-data classifier refit are all
    mutually independent, so they are dispatched together as `n_splits + 1`
    `joblib` jobs (see `n_jobs`/`parallel_backend`), rather than running the CV
    loop sequentially and only starting the final fit afterwards.

    Parameters
    ----------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
      Training data features.
    y: array-like of shape (n_samples,)
      True labels for the training data.

    Returns
    -------
    self: object
      Returns the fitted estimator instance itself.
    """
    X, y = check_X_y(X, y, accept_sparse=True)

    # save the classes and number of classes metadata
    self.classes_, counts = np.unique(y, return_counts=True)
    self.n_classes_ = len(self.classes_)
    self.train_prevalence_ = counts / len(y)
    self.y_prevs_ = self.train_prevalence_ # compatibility purposes  

    base_classifier = self.classifier if self.classifier is not None else LogisticRegression()

    oof_method = self._get_oof_method()
    if oof_method == "predict_proba" and not hasattr(base_classifier, "predict_proba"):
      # fail fast, before spending time on cross-validation
      raise TypeError(
        f"{self.__class__.__name__} requires a classifier implementing "
        f"`predict_proba`, but {base_classifier.__class__.__name__} does not "
        "provide one."
      )

    # check validation and instanciate the cross val strategy
    cv = check_cv(self.cv, y, classifier=True) # Returns a StratifiedKFold object
    n_samples = X.shape[0]
    splits = list(cv.split(X, y))

    if oof_method == "predict_proba":
      y_pred_oof = np.zeros((n_samples, self.n_classes_)) # 2d
    else:
      y_pred_oof = np.zeros(n_samples) # 1d
    y_true_oof = np.zeros(n_samples, dtype=y.dtype)

    # dispatch every fold fit/predict AND the final full-data refit as
    # independent parallel jobs, since none of them depend on each other
    fold_jobs = [
      delayed(_clone_fit_predict_fold)(base_classifier, X, y, train_idx, test_idx, oof_method)
      for train_idx, test_idx in splits
    ]
    final_fit_job = delayed(_clone_fit_full)(base_classifier, X, y)

    *fold_results, self.classifier_ = Parallel(
      n_jobs=self.n_jobs, backend=self.parallel_backend,
    )(fold_jobs + [final_fit_job])

    # cv.split guarantees test_idx never repeats across folds (mutually exclusive),
    # so scattering results back is race-free even though jobs ran out of order
    for test_idx, y_test, y_pred in fold_results:
      y_true_oof[test_idx] = y_test
      y_pred_oof[test_idx] = y_pred

    # trigger the calibration method
    self._calibrate(y_true_oof, y_pred_oof)

    return self

  def predict(self, X: np.ndarray) -> np.ndarray:
    """ Maps the test bag and estimate the class prevalences.

    Ensures that the post-optimization output adopts consistent geometric properties.

    Parameters
    ----------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
      Test bag of unlabelled test samples.

    Returns
    -------
    p_adjusted: ndarray of shape (n_classes,)
      Array normalized and with all valid predictions.
    """
    check_is_fitted(self)

    X = check_array(X, accept_sparse=True)
    p_adjusted = self._quantify(X) # runs the specific math calculus for the subclass

    p_adjusted = np.clip(p_adjusted, 0.0, 1.0) # avoid values outside the valid range
    p_sum = np.sum(p_adjusted)

    if p_sum > 0:
      p_adjusted /= p_sum
    else:
      p_adjusted = np.ones(self.n_classes_) / self.n_classes_

    return p_adjusted


class BaseMixtureQuantifier(BaseQuantifier, ABC):
  """Base class for Distance-based Mixture Models (DMM) for quantification.

  This abstract class manages the core optimization engine (via cvxpy) and 
  the Golden Section Search (GSS) binary fallback mechanism for quantifiers 
  that estimate class prevalences by minimizing statistical distances between 
  training and test feature/score distributions.

  Parameters
  ----------
  classifier : estimator object, default=None
    The underlying base classifier. Can be None for feature-based mixture 
    models (e.g., HDx, ReadMe) or an instance of a classifier for prediction-based 
    mixture models (e.g., HDy, EDy).

  distance_metric : str, default='L1'
    The mathematical distance metric to minimize. Supported metrics:
    - 'L1': Manhattan Distance (Sum of absolute errors).
    - 'L2': Euclidean Distance (Root of the sum of squared errors).
    - 'HD': Hellinger Divergence (Measures overlap between probability distributions).
    - 'TS': Topsoe Distance (Symmetric version of Kullback-Leibler Divergence).

  use_convex_solver : bool, default=True
    If True, attempts to find the exact global minimum using `cvxpy`.
    If False or if the convex solver fails, automatically activates the 
    Golden Section Search numerical fallback mechanism.

  Attributes
  ----------
  classes_ : ndarray of shape (n_classes,)
    The distinct class labels found during the training phase.

  n_classes_ : int
    The total number of unique classes.

  train_prevalence_ : ndarray of shape (n_classes,)
    The baseline prevalence proportion of each class observed in the training data.

  conditional_matrix_ : ndarray of shape (n_components, n_classes)
    The conditional probability matrix built during the `fit` phase.
    Represents the expected distribution profile for each class from the training set.

  Notes
  -----
  When `use_convex_solver=True`, the underlying CVXPY `Problem` (variable,
  parameters and constraints) is built lazily on the first `.predict()` call
  and cached on the instance, keyed by `(n_components, n_classes,
  distance_metric)`. Subsequent `.predict()` calls only update the cached
  `Parameter` values (`conditional_matrix_` and the test frequency vector)
  and re-solve, skipping the relatively expensive problem construction/DCP
  validation step — this matters a lot when scoring many bags (e.g. from
  `quack.bag_generator`) with the same fitted quantifier. The cache is
  automatically rebuilt if `conditional_matrix_`'s shape or
  `distance_metric` change (e.g. after calling `.fit()` again with
  different data).
  """
  def __init__(self,
               classifier: BaseEstimator = None,
               distance_metric: str = 'L1',
               use_convex_solver: bool = True):
    super().__init__(classifier=classifier)
    self.distance_metric = distance_metric
    self.use_convex_solver = use_convex_solver
    self.conditional_matrix_ = None
    self._cvx_cache_key_ = None

  def _compute_distance(self,
                        candidate_prevalence: np.ndarray,
                        test_frequencies: np.ndarray) -> float:
    """Computes the selected distance error for a given prevalence candidate.

    Simulates how the test dataset should look if the true prevalence matched 
    `candidate_prevalence`, comparing the result against the actual observed 
    `test_frequencies`. Used primarily as a cost function for the GSS optimizer.
    """
    # project the candidate prevalence using the training distribution profile (CM * p)
    projected_frequencies = self.conditional_matrix_.dot(candidate_prevalence)

    if self.distance_metric == 'L1':
      return np.linalg.norm(projected_frequencies - test_frequencies, ord=1)
    if self.distance_metric == 'L2':
      return np.linalg.norm(projected_frequencies - test_frequencies)
    if self.distance_metric == 'HD':
      return np.sqrt(np.sum((np.sqrt(projected_frequencies) - np.sqrt(test_frequencies)) ** 2))

    if self.distance_metric == 'TS':
      # vectorized Topsoe distance; terms where a frequency is exactly 0
      # contribute 0 by convention (x*log(x) -> 0 as x -> 0), so they are
      # masked out instead of computed (which would otherwise emit
      # divide-by-zero / invalid-value warnings from log(0) or 0/0)
      denom = projected_frequencies + test_frequencies
      with np.errstate(divide='ignore', invalid='ignore'):
        term_projected = np.where(
          projected_frequencies > 0,
          projected_frequencies * np.log(2 * projected_frequencies / denom),
          0.0,
        )
        term_test = np.where(
          test_frequencies > 0,
          test_frequencies * np.log(2 * test_frequencies / denom),
          0.0,
        )
      return float(np.sum(term_projected) + np.sum(term_test))

    raise ValueError(f"Unknown distance metric: {self.distance_metric}")

  def _get_convex_problem(self, n_components: int, n_classes: int):
    """Returns the cached CVXPY `(problem, matrix_param, test_freq_param, prevalence_var)`
    tuple for the current `(n_components, n_classes, distance_metric)`, building
    (and caching) it on first use or whenever that key changes."""
    cache_key = (n_components, n_classes, self.distance_metric)
    if self._cvx_cache_key_ == cache_key:
      return (self._cvx_problem_, self._cvx_matrix_param_,
              self._cvx_test_freq_param_, self._cvx_prevalence_var_)

    estimated_prevalence = cvx.Variable(n_classes)
    matrix_param = cvx.Parameter((n_components, n_classes))
    test_freq_param = cvx.Parameter(n_components, nonneg=True)

    constraints = [estimated_prevalence >= 0, cvx.sum(estimated_prevalence) == 1.0]
    projected_frequencies = matrix_param @ estimated_prevalence

    if self.distance_metric == 'L1':
      objective_function = cvx.Minimize(cvx.norm1(projected_frequencies - test_freq_param))
    elif self.distance_metric == 'L2':
      objective_function = cvx.Minimize(cvx.norm(projected_frequencies - test_freq_param))
    elif self.distance_metric == 'HD':
      # maximizing affinity is mathematically equivalent to minimizing Hellinger Distance
      objective_function = cvx.Maximize(cvx.sum(cvx.sqrt(cvx.multiply(test_freq_param, projected_frequencies))))
    elif self.distance_metric == 'TS':
      objective_function = cvx.Minimize(cvx.sum(
        cvx.kl_div(2 * projected_frequencies, test_freq_param) +
        cvx.kl_div(2 * test_freq_param, projected_frequencies)
      ))
    else:
      raise ValueError(f"Distance metric not supported by the convex solver: {self.distance_metric}")

    problem = cvx.Problem(objective_function, constraints)

    self._cvx_cache_key_ = cache_key
    self._cvx_problem_ = problem
    self._cvx_matrix_param_ = matrix_param
    self._cvx_test_freq_param_ = test_freq_param
    self._cvx_prevalence_var_ = estimated_prevalence

    return problem, matrix_param, test_freq_param, estimated_prevalence

  def _solve_via_convex_programming(self, test_frequencies: np.ndarray) -> np.ndarray:
    """Solves the constrained mixture problem using exact convex optimization.

    Reuses the cached CVXPY problem for this instance (see
    `_get_convex_problem`), only updating the `conditional_matrix_` and
    `test_frequencies` parameter values before re-solving — avoiding the
    cost of rebuilding the problem graph on every call.
    """
    n_components, n_classes = self.conditional_matrix_.shape
    problem, matrix_param, test_freq_param, estimated_prevalence = self._get_convex_problem(
      n_components, n_classes
    )

    matrix_param.value = self.conditional_matrix_
    test_freq_param.value = test_frequencies
    problem.solve(warm_start=True)

    return estimated_prevalence.value
  
  def _golden_section_search_fallback(self, test_frequencies: np.ndarray, tolerance: float = 1e-04) -> np.ndarray:
    """Golden Section Search (GSS) algorithm for binary optimization fallback.

    Approximates the minimum point of the distance function by narrowing down 
    the search window based on the Golden Ratio (phi). Automatically triggered 
    if the convex solver fails or is disabled.
    """
    # golden Section mathematical constants (1/phi and 1/phi^2)
    inverse_phi = (math.sqrt(5) - 1) / 2
    inverse_phi_squared = (3 - math.sqrt(5)) / 2
    
    # initial search boundaries for the positive class prevalence: between 0% and 100%
    lower_bound, upper_bound = 0.0, 1.0
    interval_width = 1.0
    
    # calculate the required number of steps to satisfy the target tolerance threshold
    total_steps = int(math.ceil(math.log(tolerance / interval_width) / math.log(inverse_phi)))

    # define the two initial internal sampling probe points
    probe_point_1 = lower_bound + inverse_phi_squared * interval_width
    probe_point_2 = lower_bound + inverse_phi * interval_width
    
    # evaluate distance errors at both sample probes (assuming binary scenario: [p, 1-p])
    error_at_probe_1 = self._compute_distance(np.array([probe_point_1, 1.0 - probe_point_1]), test_frequencies)
    error_at_probe_2 = self._compute_distance(np.array([probe_point_2, 1.0 - probe_point_2]), test_frequencies)

    # iteratively shrink the search window
    for _ in range(total_steps - 1):
      if error_at_probe_1 < error_at_probe_2:
        # the minimum lies in the left segment; discard the rightmost region
        upper_bound = probe_point_2
        probe_point_2 = probe_point_1
        error_at_probe_2 = error_at_probe_1
        interval_width = inverse_phi * interval_width
        probe_point_1 = lower_bound + inverse_phi_squared * interval_width
        error_at_probe_1 = self._compute_distance(np.array([probe_point_1, 1.0 - probe_point_1]), test_frequencies)
      else:
        # the minimum lies in the right segment; discard the leftmost region
        lower_bound = probe_point_1
        probe_point_1 = probe_point_2
        error_at_probe_1 = error_at_probe_2
        interval_width = inverse_phi * interval_width
        probe_point_2 = lower_bound + inverse_phi * interval_width
        error_at_probe_2 = self._compute_distance(np.array([probe_point_2, 1.0 - probe_point_2]), test_frequencies)

    # select the absolute best prevalence candidate within the finalized narrow window
    if error_at_probe_1 < error_at_probe_2:
      error_at_lower = self._compute_distance(np.array([lower_bound, 1.0 - lower_bound]), test_frequencies)
      mid_point = (lower_bound + probe_point_2) / 2
      error_at_mid = self._compute_distance(np.array([mid_point, 1.0 - mid_point]), test_frequencies)
      best_positive_prevalence = [lower_bound, mid_point, probe_point_2][int(np.argmin([error_at_lower, error_at_mid, error_at_probe_2]))]
    else:
      error_at_upper = self._compute_distance(np.array([upper_bound, 1.0 - upper_bound]), test_frequencies)
      mid_point = (upper_bound + probe_point_1) / 2
      error_at_mid = self._compute_distance(np.array([mid_point, 1.0 - mid_point]), test_frequencies)
      best_positive_prevalence = [upper_bound, mid_point, probe_point_1][int(np.argmin([error_at_upper, error_at_mid, error_at_probe_1]))]

    return np.array([best_positive_prevalence, 1.0 - best_positive_prevalence])

  @abstractmethod
  def _compute_score(self, X: np.ndarray) -> np.ndarray:
    """Extracts the empirical frequency distribution of the test batch X.

    Must be implemented by specific sub-quantifiers (e.g., HDx, HDy, ReadMe) 
    to map the testing data into the matching distribution space.
    """
    pass

  def _solve_mixture(self, test_frequencies: np.ndarray) -> np.ndarray:
    """Solves for the class prevalences that best explain `test_frequencies`,
    trying the exact convex solver first and falling back to Golden Section
    Search on failure or when the solver is disabled.

    Shared by every DMM-style quantifier's prevalence estimation step
    (`BaseMixtureQuantifier.predict` for feature-based methods like `HDx`/
    `ReadMe`/`ED`, and `_quantify` in score/confusion-matrix-based methods
    like `DyS`/`FormanMM`/`GAC`/`GPAC`/`FM` in `quack.quantifiers._dmm`) so
    the convex-solve-with-fallback control flow lives in exactly one place.

    Parameters
    ----------
    test_frequencies: np.ndarray
      Empirical frequency/score distribution
      of the test bag, in the same space as `conditional_matrix_`.

    Returns
    -------
    raw_prevalences: np.ndarray
      Raw (not yet clipped/renormalized) prevalence solution
      of shape `(n_classes,)`.
    """
    if not self.use_convex_solver:
      return self._golden_section_search_fallback(test_frequencies)

    try:
      prevalence_solution = self._solve_via_convex_programming(test_frequencies)
      if prevalence_solution is None:
        warnings.warn("Convex optimization returned an empty result. Falling back to GSS search.")
        return self._golden_section_search_fallback(test_frequencies)
      return np.array(prevalence_solution).squeeze()
    except cvx.SolverError:
      warnings.warn("CVXPY SolverError encountered. Falling back to GSS search as a safety measure.")
      return self._golden_section_search_fallback(test_frequencies)
  
  def predict(self, X: np.ndarray) -> np.ndarray:
    """Estimates the class prevalences for the given test data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
      The testing data matrix.

    Returns
    -------
    estimated_prevalences : ndarray of shape (n_classes,)
      A normalized probability vector indicating the estimated prevalence 
      proportions for each class.
    """
    check_is_fitted(self)
    X = check_array(X, accept_sparse=False)

    test_frequencies = self._compute_score(X)
    estimated_prevalences = self._solve_mixture(test_frequencies)

    # --- Geometric Post-Processing Pipeline ---
    # Clip edge values into the strict [0.0, 1.0] range to fix floating-point precision noise
    estimated_prevalences = np.clip(estimated_prevalences, 0.0, 1.0)

    # Enforce probability closure (the final vector elements must sum to exactly 1.0)
    total_sum = np.sum(estimated_prevalences)
    if total_sum > 0:
      estimated_prevalences /= total_sum
    else:
      # Safe boundary fallback: apply a uniform distribution if everything collapsed to zero
      estimated_prevalences = np.ones(self.n_classes_) / self.n_classes_

    return estimated_prevalences


class BaseScoreMixtureQuantifier(BaseCalibratedQuantifier, BaseMixtureQuantifier, ABC):
  """Base class for binary mixture-model quantifiers built on 1D classifier scores.

  Factors out the boilerplate shared by score-based binary DMM quantifiers
  (`DyS`/`HDy`, which bin scores into histograms, and `FormanMM`, which
  bins them into a CDF): resolving whether Out-of-Fold scores come from
  `predict_proba` or `decision_function`, extracting the positive-class
  column from either output shape, enforcing the binary-only constraint,
  and delegating prevalence estimation to
  `BaseMixtureQuantifier._solve_mixture` once `_compute_score` has mapped
  a test bag into the same binned space used to build
  `conditional_matrix_`. Subclasses only need to implement `_calibrate`
  (building `conditional_matrix_` from Out-of-Fold scores) and
  `_compute_score` (mapping a test bag into that same space).

  Parameters
  ----------
  classifier : estimator object
    The underlying base classifier.
  distance_metric : str
    The distance metric minimized by the mixture solver.
  cv : int, cross-validation generator or an iterable, default = 10
    Cross-validation strategy used to generate Out-of-Fold scores.
  use_convex_solver : bool, default = True
    If True, solves via CVXPY; falls back to Golden Section Search
    otherwise or on solver failure.
  predict_proba : bool, default = False
    If True, forces `predict_proba` for scoring. If False, uses
    `decision_function` when the classifier provides one, falling back to
    `predict_proba` otherwise.
  n_jobs : int, default = None
    Number of jobs to run in parallel while fitting the `cv` folds. See
    `BaseCalibratedQuantifier`.
  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the CV/final-fit jobs.
  """

  def __init__(self,
               classifier: BaseEstimator,
               distance_metric: str,
               cv: int = 10,
               use_convex_solver: bool = True,
               predict_proba: bool = False,
               n_jobs: int = None,
               parallel_backend: str = "loky"):
    BaseCalibratedQuantifier.__init__(self, classifier=classifier, cv=cv,
                                      n_jobs=n_jobs, parallel_backend=parallel_backend)
    BaseMixtureQuantifier.__init__(self, classifier=classifier, distance_metric=distance_metric,
                                   use_convex_solver=use_convex_solver)
    self.predict_proba = predict_proba

  def _get_oof_method(self) -> str:
    return ("predict_proba" if self.predict_proba
            else ("decision_function" if hasattr(self.classifier, "decision_function") else "predict_proba"))

  @staticmethod
  def _extract_1d_scores(y_predictions: np.ndarray) -> np.ndarray:
    """Extracts positive-class probabilities from a `predict_proba` (2D)
    output; passes a `decision_function` (already 1D) output through."""
    if y_predictions.ndim == 2:
      return y_predictions[:, 1]
    return y_predictions

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseScoreMixtureQuantifier':
    """Fits the quantifier, enforcing the binary-only constraint shared
    by every score-based mixture model in this family.

    Raises
    ------
    ValueError
      If `y` contains more than 2 distinct classes.
    """
    unique_classes = np.unique(y)
    if len(unique_classes) > 2:
      raise ValueError(f"{self.__class__.__name__} only works for binary quantification.")
    return BaseCalibratedQuantifier.fit(self, X, y)

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    test_frequencies = self._compute_score(X)
    return self._solve_mixture(test_frequencies)

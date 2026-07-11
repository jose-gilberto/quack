import numpy as np
import math
import warnings
import cvxpy as cvx
from abc import ABC, abstractmethod
from typing import TypeVar
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import check_cv
from sklearn.linear_model import LogisticRegression

T = TypeVar('T', bound='BaseQuantifier')


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
  def fit(self, X: np.ndarray, y: np.ndarray) -> T:
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
  """

  def __init__(self, classifier: BaseEstimator = None, cv: int = 10):
    super().__init__(classifier=classifier)
    self.cv = cv
    
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

  def fit(self, X: np.ndarray, y: np.ndarray) -> T:
    """ Adjust the complete quantification pipeline using predictions out-of-fold.
    
    Executes the cross-validation for internal calibration and, following, adjust
    the model with all.
    
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
    
    # check validation and instanciate the cross val strategy
    cv = check_cv(self.cv, y, classifier=True) # Returns a StratifiedKFold object
    n_samples = X.shape[0]
    
    oof_method = self._get_oof_method()
    if oof_method == "predict_proba":
      y_pred_oof = np.zeros((n_samples, self.n_classes_)) # 2d
    else:
      y_pred_oof = np.zeros(n_samples) # 1d

    y_true_oof = np.zeros(n_samples, dtype=y.dtype)
    # cross validation loop
    for train_idx, test_idx in cv.split(X, y):
      X_train, X_test = X[train_idx], X[test_idx]
      y_train, y_test = y[train_idx], y[test_idx]
      
      # clone in a clean way to avoid data leak between folds
      fold_classifier = clone(base_classifier)
      fold_classifier.fit(X_train, y_train)
      
      y_true_oof[test_idx] = y_test
      pred_func = getattr(fold_classifier, oof_method) # returns the method callable 
      y_pred_oof[test_idx] = pred_func(X_test) # even if appear that we can have an overhead
      # the cv.split guarantees the a test_idx do not appears as test more than once
      # (disjunt or mutually exclusive)

    # trigger the calibration method
    self._calibrate(y_true_oof, y_pred_oof)
    
    self.classifier_ = clone(base_classifier)
    self.classifier_.fit(X, y) # train the final classifier with all training data     

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
  """
  def __init__(self,
               classifier: BaseEstimator = None,
               distance_metric: str = 'L1',
               use_convex_solver: bool = True):
    super().__init__(classifier=classifier)
    self.distance_metric = distance_metric
    self.use_convex_solver = use_convex_solver
    self.conditional_matrix_ = None

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
      term_projected = sum(projected_frequencies[i] * np.log(2 * projected_frequencies[i] / (projected_frequencies[i] + test_frequencies[i])) 
                           if projected_frequencies[i] != 0 else 0 for i in range(projected_frequencies.shape[0]))
      term_test = sum(test_frequencies[i] * np.log(2 * test_frequencies[i] / (projected_frequencies[i] + test_frequencies[i])) 
                      if test_frequencies[i] != 0 else 0 for i in range(projected_frequencies.shape[0]))
      return term_projected + term_test

    raise ValueError(f"Unknown distance metric: {self.distance_metric}")

  def _solve_via_convex_programming(self, test_frequencies: np.ndarray) -> np.ndarray:
    """Solves the constrained mixture problem using exact convex optimization."""
    # define the target optimization variable: the estimated prevalence vector
    estimated_prevalence = cvx.Variable(self.conditional_matrix_.shape[1])

    # proportions cannot be negative and must sum to exactly 1.0
    constraints = [estimated_prevalence >= 0, cvx.sum(estimated_prevalence) == 1.0]
    
    # map the selected distance metric to CVXPY objective functions
    if self.distance_metric == 'L1':
      objective_function = cvx.Minimize(cvx.norm1(self.conditional_matrix_ @ estimated_prevalence - test_frequencies))
    elif self.distance_metric == 'L2':
      objective_function = cvx.Minimize(cvx.norm(self.conditional_matrix_ @ estimated_prevalence - test_frequencies))
    elif self.distance_metric == 'HD':
      # maximizing affinity is mathematically equivalent to minimizing Hellinger Distance
      objective_function = cvx.Maximize(cvx.sum(cvx.sqrt(cvx.multiply(test_frequencies, self.conditional_matrix_ @ estimated_prevalence))))
    elif self.distance_metric == 'TS':
      objective_function = cvx.Minimize(cvx.sum(cvx.kl_div(2 * self.conditional_matrix_ @ estimated_prevalence, test_frequencies) + 
                                                cvx.kl_div(2 * test_frequencies, self.conditional_matrix_ @ estimated_prevalence)))
    else:
      raise ValueError(f"Distance metric not supported by the convex solver: {self.distance_metric}")

    # construct and solve the mathematical problem globally
    problem = cvx.Problem(objective_function, constraints)
    problem.solve()

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
    # validate that the quantifier has been fitted and parse input data
    check_is_fitted(self)
    X = check_array(X, accept_sparse=False)
    
    # calculate the test score distribution/frequencies
    test_frequencies = self._compute_score(X)

    # bypass the convex solver entirely if use_convex_solver is explicitly turned off
    if not self.use_convex_solver:
      return self._golden_section_search_fallback(test_frequencies)

    # Primary route: Attempt exact programming via convex optimization
    try:
      prevalence_solution = self._solve_via_convex_programming(test_frequencies)
      
      if prevalence_solution is None:
        warnings.warn("Convex optimization returned an empty result. Falling back to GSS search.")
        return self._golden_section_search_fallback(test_frequencies)
          
      estimated_prevalences = np.array(prevalence_solution).squeeze()
        
    except cvx.SolverError:
      # Catch mathematical instabilities or convergence issues triggered by CVXPY
      warnings.warn("CVXPY SolverError encountered. Falling back to GSS search as a safety measure.")
      return self._golden_section_search_fallback(test_frequencies)

    # --- Geometric Post-Processing Pipeline ---
    # Clip edge values into the strict [0.0, 1.0] range to fix floating-point precision noise
    estimated_prevalences = np.clip(estimated_prevalences, 0.0, 1.0)
    
    # Enforce probability closure (The final vector elements must sum to exactly 1.0)
    total_sum = np.sum(estimated_prevalences)
    if total_sum > 0:
      estimated_prevalences /= total_sum
    else:
      # Safe boundary fallback: apply a uniform distribution if everything collapsed to zero
      estimated_prevalences = np.ones(self.n_classes_) / self.n_classes_
        
    return estimated_prevalences

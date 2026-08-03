import warnings
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from quack.quantifiers.base import BaseCalibratedQuantifier


class EM(BaseCalibratedQuantifier):
  """Expectation Maximization (EM) Quantifier.

  An iterative quantification algorithm that adapts a classifier's output probabilities 
  to a target test set by maximizing the likelihood of the test data. It recursively 
  updates sample posteriors and prior estimations until convergence.

  Parameters
  ----------
  classifier : estimator object, default = None
    The underlying base classifier implementing `predict_proba`. If None, 
    defaults to `LogisticRegression()`.

  cv : int, cross-validation generator or an iterable, default = 10
    Determines the cross-validation splitting strategy for the calibration phase.

  epsilon : float, default = 1e-06
    The convergence tolerance threshold. The iterative loop terminates when the 
    Euclidean norm between consecutive steps is smaller than this value.

  max_iter : int, default = 1000
    The maximum allowable optimization iterations.

  n_jobs : int, default = None
    Number of jobs to run in parallel while fitting the `cv` folds (plus
    the final full-data classifier refit). See `BaseCalibratedQuantifier`.

  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the CV/final-fit jobs.

  Attributes
  ----------
  classes_ : ndarray of shape (n_classes,)
    The distinct class labels found during the training phase.

  n_classes_ : int
    The total number of unique classes.

  train_prevalence_ : ndarray of shape (n_classes,)
    The baseline prevalence proportion of each class observed in the training data.

  classifier_ : estimator object
    The final trained base classifier adjusted on the entire training dataset.

  References
  ----------
  Marco Saerens, Patrice Latinne, and Christine Decaestecker.
  Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure.
  Neural Computation, 14(1): 21-41, 2002.
  """

  def __init__(self, classifier: BaseEstimator = None, cv: int = 10,
               epsilon: float = 1e-06, max_iter: int = 1000,
               n_jobs: int = None, parallel_backend: str = "loky"):
    super().__init__(classifier=classifier, cv=cv, n_jobs=n_jobs, parallel_backend=parallel_backend)
    self.epsilon = epsilon
    self.max_iter = max_iter

  def _get_oof_method(self) -> str:
    return "predict_proba"

  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    """EM handles calibration adjustments dynamically inside the prediction loop."""
    pass

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    # extract probabilistic scores
    predicted_probabilities = self.classifier_.predict_proba(X)

    # initialize convergence tracking arrays matching the original calculus state
    prevalence_new = self.train_prevalence_
    prevalence_old = np.ones(self.train_prevalence_.shape)
    iteration_count = 0

    # convergence logic loop
    while (np.linalg.norm(prevalence_old - prevalence_new) > self.epsilon) and iteration_count < self.max_iter:
      prevalence_old = np.array(prevalence_new)

      # vectorized update: row-wise multiply every sample's posterior
      # vector by the (prior_new / prior_train) ratio via broadcasting,
      # then renormalize each row to sum to 1.0 — mathematically identical
      # to the previous per-sample Python loop, just without the loop
      posterior_matrix = predicted_probabilities * (prevalence_old / self.train_prevalence_)
      posterior_matrix /= posterior_matrix.sum(axis=1, keepdims=True)

      # update step: average the adjusted sample posterior profiles
      prevalence_new = posterior_matrix.mean(axis=0)
      iteration_count += 1

    return prevalence_new


class CDE(BaseCalibratedQuantifier):
  """Class Conditional Density Estimation (CDE) Quantifier.

  An iterative binary threshold-adjusting quantifier that modulates prediction 
  cutoffs dynamically by evaluating relative target distribution shifts.

  Parameters
  ----------
  classifier : estimator object, default = None
    The underlying base classifier implementing `predict_proba`. If None, 
    defaults to `LogisticRegression()`.

  cv : int, cross-validation generator or an iterable, default = 10
    Determines the cross-validation splitting strategy for the calibration phase.

  epsilon : float, default = 1e-06
    The convergence tolerance threshold.

  max_iter : int, default = 1000
    The maximum allowable optimization iterations.

  n_jobs : int, default = None
    Number of jobs to run in parallel while fitting the `cv` folds. See
    `BaseCalibratedQuantifier`.

  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the CV/final-fit jobs.

  Attributes
  ----------
  classes_ : ndarray of shape (2,)
    The binary class labels found during the training phase.

  n_classes_ : int
    The total number of unique classes (strictly equals 2).

  train_prevalence_ : ndarray of shape (2,)
    The baseline prevalence proportion of each class observed in the training data.

  classifier_ : estimator object
    The final trained base classifier adjusted on the entire training dataset.
      
  References
  ----------
  Dirk Tasche. Fisher consistency for prior probability shift.
  Journal of Machine Learning Research, 18(95):1-32, 2017.
  """

  def __init__(self, classifier: BaseEstimator = None, cv: int = 10,
               epsilon: float = 1e-06, max_iter: int = 1000,
               n_jobs: int = None, parallel_backend: str = "loky"):
    super().__init__(classifier=classifier, cv=cv, n_jobs=n_jobs, parallel_backend=parallel_backend)
    self.epsilon = epsilon
    self.max_iter = max_iter

  def _get_oof_method(self) -> str:
    return "predict_proba"

  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    """CDE thresholding adjustments are executed entirely during the testing phase."""
    pass

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'CDE':
    # check target dimensionality before triggering the pipeline execution
    unique_classes = np.unique(y)
    if len(unique_classes) > 2:
      raise ValueError(
        "CDE only works for binary quantification. Multiclass is possible via the "
        "OVR strategy, but not recommended due to theoretical issues with that approach."
      )

    return super().fit(X, y)

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    predicted_probabilities = self.classifier_.predict_proba(X)
    pos_probs = predicted_probabilities[:, 1]

    # initialize directional weight arrays matching the original state
    weights = np.ones(2)
    weights_old = np.zeros(2)

    positive_prevalence = 2.0
    iteration_count = 0

    # strict preservation of your original termination criteria (<= max_iter)
    while np.linalg.norm(weights - weights_old) > self.epsilon and iteration_count <= self.max_iter:
      # vectorized hard-label assignment: replaces the previous
      # np.apply_along_axis(lambda, ...) call, which is a thin Python-loop
      # wrapper around each row and not actually vectorized; np.where
      # performs the exact same row-wise comparison in a single C-level pass
      threshold = weights[0] / np.sum(weights)
      threshold_labels = np.where(pos_probs > threshold, self.classes_[1], self.classes_[0])
      weights_old = np.copy(weights)

      # calculate the empirical mean of positive labels
      positive_prevalence = np.mean(threshold_labels == self.classes_[1])

      # re-weight updates based on baseline training rates discrepancies
      weights[0] = (1.0 - positive_prevalence) / self.train_prevalence_[0]
      weights[1] = positive_prevalence / self.train_prevalence_[1]
      iteration_count += 1

    if iteration_count >= self.max_iter:
      warnings.warn("The CDE iteration has not converged.")

    return np.array([1.0 - positive_prevalence, positive_prevalence])
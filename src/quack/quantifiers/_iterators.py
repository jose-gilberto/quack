import warnings
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
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
    defaults to `LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto')`.

  cv : int, cross-validation generator or an iterable, default = 10
    Determines the cross-validation splitting strategy for the calibration phase.

  epsilon : float, default = 1e-06
    The convergence tolerance threshold. The iterative loop terminates when the 
    Euclidean norm between consecutive steps is smaller than this value.

  max_iter : int, default = 1000
    The maximum allowable optimization iterations.

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
               epsilon: float = 1e-06, max_iter: int = 1000):
    super().__init__(classifier=classifier, cv=cv)
    self.epsilon = epsilon
    self.max_iter = max_iter

  def _get_oof_method(self) -> str:
    return "predict_proba"

  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    """EM handles calibration adjustments dynamically inside the prediction loop."""
    pass

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    n_samples = X.shape[0]
    
    # extract probabilistic scores
    predicted_probabilities = self.classifier_.predict_proba(X)

    # initialize convergence tracking arrays matching the original calculus state
    prevalence_new = self.train_prevalence_
    prevalence_old = np.ones(self.train_prevalence_.shape)
    iteration_count = 0

    # convergence logic loop
    while (np.linalg.norm(prevalence_old - prevalence_new) > self.epsilon) and iteration_count < self.max_iter:
      prevalence_old = np.array(prevalence_new)
      
      # compute updated sample posteriors adjusted by current prior weights
      posterior_matrix = np.array([
        (prevalence_old / self.train_prevalence_) * predicted_probabilities[i] 
        for i in range(n_samples)
      ])
      
      # row-wise normalization to enforce true probability distributions
      row_sums = np.sum(posterior_matrix, axis=1)[:, np.newaxis]
      posterior_matrix = posterior_matrix / row_sums

      # update step: average the adjusted sample posterior profiles
      prevalence_new = (1.0 / n_samples) * np.sum(posterior_matrix, axis=0)
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
    defaults to `LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto')`.

  cv : int, cross-validation generator or an iterable, default = 10
    Determines the cross-validation splitting strategy for the calibration phase.

  epsilon : float, default = 1e-06
    The convergence tolerance threshold.

  max_iter : int, default = 1000
    The maximum allowable optimization iterations.

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
               epsilon: float = 1e-06, max_iter: int = 1000):
    super().__init__(classifier=classifier, cv=cv)
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
    
    # initialize directional weight arrays matching the original state
    weights = np.ones(2)
    weights_old = np.zeros(2)

    positive_prevalence = 2.0
    iteration_count = 0

    # strict preservation of your original termination criteria (<= max_iter)
    while np.linalg.norm(weights - weights_old) > self.epsilon and iteration_count <= self.max_iter:
      # row-by-row hard label assignment based on current directional weights
      threshold_labels = np.apply_along_axis(
        lambda prob: self.classes_[1] if prob[1] > weights[0] / np.sum(weights) else self.classes_[0], 
        axis=1, 
        arr=predicted_probabilities
      )
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

import numpy as np
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

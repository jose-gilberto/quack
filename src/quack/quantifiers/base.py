from abc import ABC, abstractmethod
import numpy as np

from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.model_selection import cross_val_predict, StratifiedKFold


class QuantifierMixin:
  """Mixin to identify that this class belongs to quantifiers family."""
  _estimator_type = "quantifier"


class BaseQuantifier(BaseEstimator, QuantifierMixin, ABC):
  """
  Abstract class for all quantifiers.
  Compatible with Scikit-Learn API.
  """

  def __init__(self, classifier: BaseEstimator):
    self.classifier = classifier

  def fit(self, X: np.ndarray, y: np.ndarray):
    """ Fit the quantifier and the internal classifier to the training data. """
    # validation step - scikit compatible
    X, y = check_X_y(X, y, accept_sparse=True)
    # check_classification_targets(y)
    # save the classes and number of classes metadata
    self.classes_ = np.unique(y)
    self.n_classes_ = len(self.classes_)
    _, counts = np.unique(y, return_counts=True) # training prev calculus due to adjusted methods
    self.train_prevalence_ = counts / len(y)

    self.classifier_ = clone(self.classifier) # lazy loading classifier
    self.classifier_.fit(X, y)

    return self

  def predict(self, X: np.ndarray) -> np.ndarray:
    """
    Predict the class prevalence for each test bag 'X'.
    """
    check_is_fitted(self)
    X = check_array(X, accept_sparse=True)
    # default template method
    return self._quantify(X)

  @abstractmethod
  def _quantify(self, X) -> np.ndarray:
    """
    Abstract quantify method that each quantifier must implement.
    """
    pass


class BaseCalibratedQuantifier(BaseQuantifier, ABC):
  """
  Base class to quantifiers that need an calibration step via KFoldCV.
  """
  
  _strictly_binary = False

  def __init__(self, classifier: BaseEstimator, cv: int = 10):
    super().__init__(classifier=classifier)
    self.cv = cv

  def fit(self, X, y):
    X, y = check_X_y(X, y, accept_sparse=True)
    # check_classification_targets(y)

    # save the classes and number of classes metadata
    self.classes_ = np.unique(y)
    self.n_classes_ = len(self.classes_)
    
    if self._strictly_binary and self.n_classes_ > 2:
      raise ValueError(
        f"The method '{self.__class__.__name__}' only works for "
        f"binary problems (maximum of 2 classes). "
        f"The dataset provided has {self.n_classes_} classes."
        f"To use it with more classes, you have to rely on OVR strategies."
      )
    
    _, counts = np.unique(y, return_counts=True)
    self.train_prevalence_ = counts / len(y)

    base_clf = clone(self.classifier) # lazy loading

    # we train in folds in order to calculate the adjustments to perform
    cv_strategy = self.cv
    if isinstance(cv_strategy, int):
      cv_strategy = StratifiedKFold(n_splits=cv_strategy, shuffle=True, random_state=42)
    
    oof_method = self._get_oof_method()
    y_pred_oof = cross_val_predict(
      base_clf, X, y, cv=cv_strategy, method=oof_method, n_jobs=-1
    )
    self._calibrate(y, y_pred_oof)

    # after all adjustments we fit in all training data
    self.classifier_ = clone(base_clf)
    self.classifier_.fit(X, y)

    return self

  @abstractmethod
  def _get_oof_method(self) -> str:
    """
    Defines if cross validation will get 'predict' or 'predict_proba'.
    """
    pass

  @abstractmethod
  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    """
    Method where the subclass will calculate their matrices adjusts.
    """
    pass

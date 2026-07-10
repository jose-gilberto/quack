import numpy as np
from abc import ABC
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
from quack.quantifiers.base import BaseCalibratedQuantifier


class BaseThresholdQuantifier(BaseCalibratedQuantifier, ABC):
  """Base class for Threshold Selection (TS) methods using data-driven grids.

  This abstract class manages the calibration phase for quantifiers by 
  extracting unique thresholds dynamically from the Out-of-Fold (OOF) 
  predicted probabilities, matching the behavior of the QFY framework. 
  It then builds the True Positive Rate (TPR) and False Positive Rate (FPR) 
  curves over this adaptive grid.
  
  Parameters
  ----------
  classifier : estimator object, default = None
    The classifier to be used as the base for quantification.
    Must implement the `predict_proba` method. If None, an instance of 
    `LogisticRegression()` will be created.
  cv : int, cross-validation generator or an iterable, default=10
    Determines the cross-validation splitting strategy to generate the 
    Out-of-Fold predictions used for calibration.
  precision : int, default = 3
    The decimal precision used to round the Out-of-Fold probabilities 
    before extracting unique threshold candidates via `np.unique`.
  
  Attributes
  ----------
  classes_ : ndarray of shape (2,)
    The distinct class labels found during training.
  
  n_classes_ : int
    The number of distinct classes (expected to be 2).
      
  train_prevalence_ : ndarray of shape (2,)
    The prevalence of each class in the training dataset.
      
  thresholds_ : ndarray of shape (n_thresholds,)
    The dynamic data-driven grid of thresholds evaluated during calibration.
      
  tpr_by_thresh_ : ndarray of shape (n_thresholds,)
    The True Positive Rate computed out-of-fold for each dynamic threshold.
      
  fpr_by_thresh_ : ndarray of shape (n_thresholds,)
    The False Positive Rate computed out-of-fold for each dynamic threshold.
      
  classifier_ : estimator object
    The fitted base classifier trained on the entire dataset.
  """
  def __init__(self, classifier: BaseEstimator = None, cv: int = 10, precision: int = 3):
    super().__init__(classifier=classifier, cv=cv)
    self.precision = precision

  def _get_oof_method(self) -> str:
    return "predict_proba"

  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    # builds TPR and FPR validation curves over a dynamic grid of thresholds.
    neg_class = self.classes_[0]
    pos_class = self.classes_[1]

    actual_pos_mask = (y_true_oof == pos_class)
    actual_neg_mask = (y_true_oof == neg_class)

    n_pos = np.sum(actual_pos_mask)
    n_neg = np.sum(actual_neg_mask)
    
    pos_probs_oof = y_pred_oof[:, 1]  # extract probabilities from positive class

    if self.precision is None:
      self.thresholds_ = np.unique(pos_probs_oof)
    else:
      self.thresholds_ = np.unique(np.around(pos_probs_oof, decimals=self.precision))

    self.tpr_by_thresh_ = np.zeros(len(self.thresholds_))
    self.fpr_by_thresh_ = np.zeros(len(self.thresholds_))

    # loop over the dynamic thresholds calculating the calibration rates
    for idx, t in enumerate(self.thresholds_):
      if n_pos > 0:
        self.tpr_by_thresh_[idx] = np.mean(pos_probs_oof[actual_pos_mask] >= t)
      else:
        self.tpr_by_thresh_[idx] = 1.0

      if n_neg > 0:
        self.fpr_by_thresh_[idx] = np.mean(pos_probs_oof[actual_neg_mask] >= t)
      else:
        self.fpr_by_thresh_[idx] = 0.0

  def _apply_acc_formula(self, p_raw_pos: float, tpr: float, fpr: float) -> float:
    """Applies the classical ACC formula with stability checks."""
    denominator = tpr - fpr
    if np.abs(denominator) > 1e-12:
      p_adj_pos = (p_raw_pos - fpr) / denominator
    else:
      p_adj_pos = p_raw_pos
    return np.clip(p_adj_pos, 0.0, 1.0)

  
class X(BaseThresholdQuantifier):
  """Forman's X threshold selection quantifier matching QFY's TSX strategy.

  This method selects the optimal threshold from the dynamic grid that 
  minimizes the absolute distance |TPR - (1 - FPR)|, searching for the 
  intersection point in the ROC curve where TPR + FPR ~= 1.0.

  References
  ----------
  Forman, G. (2008). Quantifying counts and costs via classification. 
  Data Mining and Knowledge Discovery, 17(2), 164-206.
  
  Examples
  --------
  >>> from sklearn.datasets import make_classification
  >>> X_, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
  >>> quantifier = X()
  >>> quantifier.fit(X_, y)
  >>> X_test, _ = make_classification(n_samples=200, n_classes=2, random_state=7)
  >>> prevalences = quantifier.predict(X_test)
  >>> print(prevalences)
  """

  def _quantify(self, X_test: np.ndarray) -> np.ndarray:
    # minimizes the absolute distance |TPR - (1 - FPR)|
    idx = np.argmin(np.abs(self.tpr_by_thresh_ - (1.0 - self.fpr_by_thresh_)))

    t_chosen = self.thresholds_[idx]
    tpr = self.tpr_by_thresh_[idx]
    fpr = self.fpr_by_thresh_[idx]

    pos_probs_test = self.classifier_.predict_proba(X_test)[:, 1]
    p_raw_pos = np.mean(pos_probs_test >= t_chosen)

    p_adj_pos = self._apply_acc_formula(p_raw_pos, tpr, fpr)
    return np.array([1.0 - p_adj_pos, p_adj_pos])


class Max(BaseThresholdQuantifier):
  """Forman's Max threshold selection quantifier matching QFY's TSMax strategy.

  This method selects the threshold that maximizes the separation split 
  between classes, maximizing the absolute difference |TPR - FPR| to achieve 
  the highest denominator stability during shift adjustment.

  References
  ----------
  Forman, G. (2008). Quantifying counts and costs via classification. 
  Data Mining and Knowledge Discovery, 17(2), 164-206.
  
  Examples
  --------
  >>> from sklearn.datasets import make_classification
  >>> X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
  >>> quantifier = Max()
  >>> quantifier.fit(X, y)
  >>> X_test, _ = make_classification(n_samples=200, n_classes=2, random_state=7)
  >>> prevalences = quantifier.predict(X_test)
  >>> print(prevalences)
  """

  def _quantify(self, X_test: np.ndarray) -> np.ndarray:
    # maximizes the separation split |TPR - FPR|
    idx = np.argmax(np.abs(self.tpr_by_thresh_ - self.fpr_by_thresh_))
    
    t_chosen = self.thresholds_[idx]
    tpr = self.tpr_by_thresh_[idx]
    fpr = self.fpr_by_thresh_[idx]

    pos_probs_test = self.classifier_.predict_proba(X_test)[:, 1]
    p_raw_pos = np.mean(pos_probs_test >= t_chosen)

    p_adj_pos = self._apply_acc_formula(p_raw_pos, tpr, fpr)
    return np.array([1.0 - p_adj_pos, p_adj_pos])


class T50(BaseThresholdQuantifier):
  """Forman's Threshold 50 quantifier matching QFY's TS50 strategy.

  This method selects the threshold that is closest to achieving a 
  True Positive Rate (TPR) of exactly 0.5, minimizing the cost function 
  |TPR - 0.5| across the candidate grid.

  References
  ----------
  Forman, G. (2008). Quantifying counts and costs via classification. 
  Data Mining and Knowledge Discovery, 17(2), 164-206.
  
  Examples
  --------
  >>> from sklearn.datasets import make_classification
  >>> X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
  >>> quantifier = T50()
  >>> quantifier.fit(X, y)
  >>> X_test, _ = make_classification(n_samples=200, n_classes=2, random_state=7)
  >>> prevalences = quantifier.predict(X_test)
  >>> print(prevalences)
  """

  def _quantify(self, X_test: np.ndarray) -> np.ndarray:
    # minimizes abs(tpr - 0.5)
    idx = np.argmin(np.abs(self.tpr_by_thresh_ - 0.5))
    
    t_chosen = self.thresholds_[idx]
    tpr = self.tpr_by_thresh_[idx]
    fpr = self.fpr_by_thresh_[idx]

    pos_probs_test = self.classifier_.predict_proba(X_test)[:, 1]
    p_raw_pos = np.mean(pos_probs_test >= t_chosen)

    p_adj_pos = self._apply_acc_formula(p_raw_pos, tpr, fpr)
    return np.array([1.0 - p_adj_pos, p_adj_pos])


class MedianSweep(BaseThresholdQuantifier):
  """Forman's Median Sweep (MS) quantifier faithful to QFY's implementation.

  Median Sweep evaluates individual ACC adjustments across all candidates in the 
  dynamic grid. It enforces a strict filter where thresholds with a denominator
  (TPR - FPR) smaller than `delta_min` are ignored.
  If no thresholds satisfy the filter, it falls back to the estimate with the 
  largest available separation. Otherwise, it returns the robust median of all 
  valid predictions.

  Parameters
  ----------
  classifier : estimator object, default=None
    The classifier to be used as the base for quantification.
  cv : int, default=10
    Determines the cross-validation splitting strategy.
  precision : int, default=3
    The decimal precision used to round the Out-of-Fold probabilities.
  delta_min : float, default=0.25
    The minimum required threshold separation (TPR - FPR) to accept an 
    individual ACC adaptation, avoiding denominator instability.

  References
  ----------
  Forman, G. (2008). Quantifying counts and costs via classification. 
  Data Mining and Knowledge Discovery, 17(2), 164-206.
  
  Examples
  --------
  >>> from sklearn.datasets import make_classification
  >>> X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
  >>> quantifier = MedianSweep()
  >>> quantifier.fit(X, y)
  >>> X_test, _ = make_classification(n_samples=200, n_classes=2, random_state=7)
  >>> prevalences = quantifier.predict(X_test)
  >>> print(prevalences)
  """

  def __init__(self, classifier: BaseEstimator = None, cv: int = 10, precision: int = 3, delta_min: float = 0.25):
    super().__init__(classifier=classifier, cv=cv, precision=precision)
    self.delta_min = delta_min

  def _quantify(self, X_test: np.ndarray) -> np.ndarray:
    pos_probs_test = self.classifier_.predict_proba(X_test)[:, 1]
    
    p_adj_estimates = []
    delta_max = -2.0
    p_max_fallback = 0.5

    # runs the ACC adjust individually for each dynamic threshold
    for idx, t in enumerate(self.thresholds_):
      tpr = self.tpr_by_thresh_[idx]
      fpr = self.fpr_by_thresh_[idx]
      delta = tpr - fpr

      p_raw_pos = np.mean(pos_probs_test >= t)

      # filter out thresholds where denominator <= delta_min
      if delta > self.delta_min:
        p_adj_pos = (p_raw_pos - fpr) / delta
        p_adj_estimates.append(np.clip(p_adj_pos, 0.0, 1.0))
      elif delta > delta_max and len(p_adj_estimates) == 0:
        # track fallback in case no threshold passes the delta_min requirement
        if delta == 0:
          p_max_fallback = tpr
        else:
          p_max_fallback = (p_raw_pos - fpr) / delta
        delta_max = delta

    # if 0 thresholds are valid
    if len(p_adj_estimates) == 0:
      p_adj_pos = np.clip(p_max_fallback, 0.0, 1.0)
    else:
      p_adj_pos = np.median(p_adj_estimates)

    return np.array([1.0 - p_adj_pos, p_adj_pos])
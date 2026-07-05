import numpy as np
from abc import ABC
from quack.quantifiers.base import BaseCalibratedQuantifier


class BaseThresholdQuantifier(BaseCalibratedQuantifier, ABC):
  """
  Base class for all methods that selects thresholds (Threshold Selecion or TS methods).
  Collects the tpr, fpr and a threshold grid [0.01, ..., 0.99] during calibration step.
  """

  def _get_oof_method(self) -> str:
    # all methods that uses thresholds works with predict proba
    return "predict_proba"

  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    # define the grid of thresholds (99 pos from 0.01 until 0.99)
    self.thresholds_ = np.linspace(0.01, 0.99, 99)
    
    neg_class = self.classes_[0]
    pos_class = self.classes_[1]

    actual_pos_mask = (y_true_oof == pos_class)
    actual_neg_mask = (y_true_oof == neg_class)

    n_pos = np.sum(actual_pos_mask)
    n_neg = np.sum(actual_neg_mask)

    # arrays to save the calibration curve
    self.tpr_by_thresh_ = np.zeros(len(self.thresholds_))
    self.fpr_by_thresh_ = np.zeros(len(self.thresholds_))
    
    pos_probs_oof = y_pred_oof[:, 1] # extract only the probs from positive class

    # Loop in the threshold calculating the calibration rates
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
    """
    Apply the classical ACC formula.
    """
    denominator = tpr - fpr
    if np.abs(denominator) > 1e-12:
      p_adj_pos = (p_raw_pos - fpr) / denominator
    else:
      p_adj_pos = p_raw_pos
    return np.clip(p_adj_pos, 0.0, 1.0)
  
  
class X(BaseThresholdQuantifier):
  """X: Choose the point where tpr + fpr are closer to 1.0
  
  Refs  
    [1] Forman, G. Quantifying counts and costs via classification.
        Data Min Knowl Disc 17, 164-206 (2008).
        https://doi.org/10.1007/s10618-008-0097-y
  """
  
  _strictly_binary = True

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    # minimizes the abs distance of |TPR - (1 - FPR)|
    # which is equivalent to search the equivalent point in ROC curve where tpr + fpr = 1 
    idx = np.argmin(np.abs(self.tpr_by_thresh_ - (1.0 - self.fpr_by_thresh_)))

    t_chosen = self.thresholds_[idx]
    tpr = self.tpr_by_thresh_[idx]
    fpr = self.fpr_by_thresh_[idx]

    pos_probs_test = self.classifier_.predict_proba(X)[:, 1]
    p_raw_pos = np.mean(pos_probs_test >= t_chosen)

    p_adj_pos = self._apply_acc_formula(p_raw_pos, tpr, fpr)
    return np.array([1.0 - p_adj_pos, p_adj_pos])


class Max(BaseThresholdQuantifier):
  """Max: Maximize the absolute difference |tpr - fpr| in order of denominator stability.
  
  Refs  
  [1] Forman, G. Quantifying counts and costs via classification.
      Data Min Knowl Disc 17, 164-206 (2008).
      https://doi.org/10.1007/s10618-008-0097-y
  """
  
  _strictly_binary = True

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    # find the index that miximize the split of classifier
    idx = np.argmax(np.abs(self.tpr_by_thresh_ - self.fpr_by_thresh_))
    
    t_chosen = self.thresholds_[idx]
    tpr = self.tpr_by_thresh_[idx]
    fpr = self.fpr_by_thresh_[idx]

    pos_probs_test = self.classifier_.predict_proba(X)[:, 1]
    p_raw_pos = np.mean(pos_probs_test >= t_chosen)

    p_adj_pos = self._apply_acc_formula(p_raw_pos, tpr, fpr)
    return np.array([1.0 - p_adj_pos, p_adj_pos])
  

class T50(BaseThresholdQuantifier):
  """Threshold 50: Choose the closest threshold where tpr >= 0.5
  
  Refs
  [1] Forman, G. Quantifying counts and costs via classification.
      Data Min Knowl Disc 17, 164-206 (2008).
      https://doi.org/10.1007/s10618-008-0097-y
  """
  
  _strictly_binary = True

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    # finds the index where tpr >= 0.5 (or closest to this)
    valid_indices = np.where(self.tpr_by_thresh_ >= 0.5)[0]
    idx = valid_indices[0] if len(valid_indices) > 0 else np.argmax(self.tpr_by_thresh_)
    
    t_chosen = self.thresholds_[idx]
    tpr = self.tpr_by_thresh_[idx]
    fpr = self.fpr_by_thresh_[idx]

    # quantify in test using the chosen threshold
    pos_probs_test = self.classifier_.predict_proba(X)[:, 1]
    p_raw_pos = np.mean(pos_probs_test >= t_chosen)

    p_adj_pos = self._apply_acc_formula(p_raw_pos, tpr, fpr)
    return np.array([1.0 - p_adj_pos, p_adj_pos])


class MedianSweep(BaseThresholdQuantifier):
  """
  Median Sweep (MS): Applies the correction in adjusts for
  all thresholds and returns the mean of all remaining estimatives.
  
  Refs
  [1] Forman, G. Quantifying counts and costs via classification.
      Data Min Knowl Disc 17, 164-206 (2008).
      https://doi.org/10.1007/s10618-008-0097-y
  """
  
  _strictly_binary = True

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    pos_probs_test = self.classifier_.predict_proba(X)[:, 1]
    
    p_adj_estimates = []

    # runs the ACC adjust individually for each threshold
    for idx, t in enumerate(self.thresholds_):
      tpr = self.tpr_by_thresh_[idx]
      fpr = self.fpr_by_thresh_[idx]

      # ignore thresholds where the classifier fails to separate
      # (denominator null or close to 0) to avoid noise in the mean calculus
      if np.abs(tpr - fpr) < 1e-4:
        continue

      p_raw_pos = np.mean(pos_probs_test >= t)
      p_adj_pos = (p_raw_pos - fpr) / (tpr - fpr)
      
      # median sweep perform the clipping individually before
      p_adj_estimates.append(np.clip(p_adj_pos, 0.0, 1.0))

    # if 0 thresholds is valid, run the fallback to the center (0.5) or training prevalence
    if len(p_adj_estimates) == 0:
      p_adj_pos = self.train_prevalence_[1]
    else:
      # the median protects the result against unstable thresholds (or outliers)
      p_adj_pos = np.median(p_adj_estimates)

    return np.array([1.0 - p_adj_pos, p_adj_pos])
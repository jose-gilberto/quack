import numpy as np
from quack.quantifiers.base import BaseQuantifier, BaseCalibratedQuantifier
from sklearn.metrics import confusion_matrix


class CC(BaseQuantifier):
  """Basic Classify & Count method.
  """

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    predictions = self.classifier_.predict(X)
    
    # calculate the prevalence for each known class
    prevalences = np.zeros(self.n_classes_)
    for i, c in enumerate(self.classes_):
      prevalences[i] = np.sum(predictions == c) / len(predictions)
    
    return prevalences


class PCC(BaseQuantifier):
  """Probabilistic Classify & Count method.
  """
  
  def _quantify(self, X) -> np.ndarray:
    if not hasattr(self.classifier_, "predict_proba"):
      raise AttributeError(
        f"Base classifier '{self.classifier_.__class__.__name__}' "
        f"do not support predict_proba."
      )

    # Get the probabilities matrix (n_samples, n_classes)
    probabilities = self.classifier_.predict_proba(X)

    # The prevalence calculus is the mean of each column
    return np.mean(probabilities, axis=0)
  
  
class ACC(BaseCalibratedQuantifier):
  """Adjusted Classify & Count method.
  """
  
  _strictly_binary = True

  def _get_oof_method(self) -> str:
    return "predict"
  
  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    # Maps each class using the scikit-learn pattern (asc format)
    neg_class = self.classes_[0]
    pos_class = self.classes_[1]

    actual_pos_mask = (y_true_oof == pos_class)
    actual_neg_mask = (y_true_oof == neg_class)

    # TPR (sensibility), proportion of true positives predicted as positives
    if np.sum(actual_pos_mask) > 0:
      self.tpr_ = np.mean(y_pred_oof[actual_pos_mask] == pos_class)
    else:
      self.tpr_ = 1.0

    # FPR (1 - specificity), proportion of true negatives predicted as positives
    if np.sum(actual_neg_mask) > 0:
      self.fpr_ = np.mean(y_pred_oof[actual_neg_mask] == pos_class)
    else:
      self.fpr_ = 0.0

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    predictions = self.classifier_.predict(X)
    # Calculate the predicted raw prevalence obtained by the classifier
    pos_class = self.classes_[1]
    p_raw_pos = np.mean(predictions == pos_class)
    
    # Apply the math formula from Forman (2005)
    denominator = self.tpr_ - self.fpr_
    if np.abs(denominator) > 1e-12:
      p_adj_pos = (p_raw_pos - self.fpr_) / denominator
    else:
      # if tpr == fpr, the model isnt able to distinguish between classes.
      # we maintain the raw prevalence as fallback to avoid the division by 0.
      p_adj_pos = p_raw_pos
    
    # guardrails: clipping
    p_adj_pos = np.clip(p_adj_pos, 0.0, 1.0)
    p_adj_neg = 1.0 - p_adj_pos

    return np.array([p_adj_neg, p_adj_pos])


class PACC(BaseCalibratedQuantifier):
  """Probabilistic Adjusted Classify & Count (soft-labels)."""

  _strictly_binary = True

  def _get_oof_method(self) -> str:
    return "predict_proba"

  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    neg_class = self.classes_[0]
    pos_class = self.classes_[1]

    actual_pos_mask = (y_true_oof == pos_class)
    actual_neg_mask = (y_true_oof == neg_class)

    # mu_pos: probabilities mean attributed to the positive class
    if np.sum(actual_pos_mask) > 0:
      self.mu_pos_ = np.mean(y_pred_oof[actual_pos_mask, 1])
    else:
      self.mu_pos_ = 1.0

    # mu_neg: probabilities mean attributed to the negative class
    if np.sum(actual_neg_mask) > 0:
      self.mu_neg_ = np.mean(y_pred_oof[actual_neg_mask, 1])
    else:
      self.mu_neg_ = 0.0

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    if not hasattr(self.classifier_, "predict_proba"):
      raise AttributeError(
        f"The base classifier '{self.classifier_.__class__.__name__}' "
        f"dont have the method 'predict_proba' required for PACC."
      )
    # get probabilities for the batch
    probabilities = self.classifier_.predict_proba(X)

    # probabilities mean for the positive class in test bag
    p_raw_pos = np.mean(probabilities[:, 1])

    # apply the adjust formula
    denominator = self.mu_pos_ - self.mu_neg_
    
    if np.abs(denominator) > 1e-12:
      p_adj_pos = (p_raw_pos - self.mu_neg_) / denominator
    else:
      p_adj_pos = p_raw_pos

    # guardrails
    p_adj_pos = np.clip(p_adj_pos, 0.0, 1.0)
    p_adj_neg = 1.0 - p_adj_pos

    return np.array([p_adj_neg, p_adj_pos])

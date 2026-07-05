import numpy as np
from quack.metrics.base import QuantificationMetric


class KullbackLeiblerDivergence(QuantificationMetric):
  def __init__(self, epsilon: float = 1e-5):
    super().__init__(name="Kullback-Leibler Divergence", lower_is_better=True)
    self.epsilon = epsilon

  def compute(self, p_true: np.ndarray, p_pred: np.ndarray) -> float:
    p_true_smoothed = np.clip(p_true, self.epsilon, 1.0)
    p_pred_smoothed = np.clip(p_pred, self.epsilon, 1.0)
    return float(np.sum(p_true_smoothed * np.log(p_true_smoothed / p_pred_smoothed)))
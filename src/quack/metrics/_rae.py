import numpy as np
from quack.metrics.base import QuantificationMetric


class RelativeAbsoluteError(QuantificationMetric):
  def __init__(self, epsilon: float = 1e-5):
    super().__init__(name="Relative Absolute Error", lower_is_better=True)
    self.epsilon = epsilon

  def compute(self, p_true: np.ndarray, p_pred: np.ndarray) -> float:
    denominator = np.abs(p_true) + self.epsilon
    return float(np.sum(np.abs(p_true - p_pred) / denominator))
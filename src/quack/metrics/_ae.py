import numpy as np
from quack.metrics.base import QuantificationMetric


class AbsoluteError(QuantificationMetric):
  def __init__(self):
    super().__init__(name="Absolute Error", lower_is_better=True)

  def compute(self, p_true: np.ndarray, p_pred: np.ndarray) -> float:
    return float(np.sum(np.abs(p_true - p_pred)))
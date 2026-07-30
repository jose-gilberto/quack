import numpy as np
from quack.metrics.base import QuantificationMetric


class AbsoluteError(QuantificationMetric):
  """Mean Absolute Error (AE) between true and predicted prevalence vectors.

  Averages the absolute per-class deviation across all classes, bounding
  the metric to `[0, 1]` regardless of the number of classes (0 = perfect
  quantification, 1 = maximally wrong).

    AE(p, p_hat) = (1 / n_classes) * sum_c |p(c) - p_hat(c)|

  References
  ----------
  George Forman. Quantifying counts and costs via classification.
  Data Mining and Knowledge Discovery, 17(2):164-206, 2008.
  """
  def __init__(self):
    super().__init__(name="Absolute Error", lower_is_better=True)

  def compute(self, p_true: np.ndarray, p_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(p_true - p_pred)))
import numpy as np
from quack.metrics.base import QuantificationMetric


class RelativeAbsoluteError(QuantificationMetric):
  """Relative Absolute Error (RAE) between true and predicted prevalence vectors.

  Averages the per-class absolute deviation relative to the true
  prevalence. Both `p_true` and `p_pred` are additively smoothed (see
  `QuantificationMetric._smooth`) before the ratio is computed, since
  dividing by a true prevalence of exactly 0 would otherwise make the
  metric undefined for classes absent from the test bag.

    p_s(c)        = (p(c) + eps) / (1 + n_classes * eps)
    RAE(p, p_hat) = (1 / n_classes) * sum_c |p_s(c) - p_hat_s(c)| / p_s(c)

  Parameters
  ----------
  epsilon : float, default = 1e-5
    Smoothing factor applied to both `p_true` and `p_pred` before
    computing the ratio.

  References
  ----------
  George Forman. Quantifying counts and costs via classification.
  Data Mining and Knowledge Discovery, 17(2):164-206, 2008.
  """
  def __init__(self, epsilon: float = 1e-5):
    super().__init__(name="Relative Absolute Error", lower_is_better=True)
    self.epsilon = epsilon

  def compute(self, p_true: np.ndarray, p_pred: np.ndarray) -> float:
    p_true_smoothed = self._smooth(p_true, self.epsilon)
    p_pred_smoothed = self._smooth(p_pred, self.epsilon)
    return float(np.mean(np.abs(p_true_smoothed - p_pred_smoothed) / p_true_smoothed))
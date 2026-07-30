import numpy as np
from quack.metrics.base import QuantificationMetric


class NormalizedAbsoluteError(QuantificationMetric):
  """Normalized Absolute Error (NAE) between true and predicted prevalence vectors.

  Normalizes the (unaveraged) Absolute Error by its theoretical maximum
  given `p_true`, bounding the metric to `[0, 1]` regardless of how
  skewed the true prevalence is. This makes NAE more comparable across
  experiments/datasets with very different training or test prevalences
  than the plain `AbsoluteError`.

    NAE(p, p_hat) = sum_c |p(c) - p_hat(c)| / (2 * (1 - min_c p(c)))

  References
  ----------
  Esuli, A. & Sebastiani, F. (2015). Optimizing text quantifiers for
  multivariate loss functions. ACM Transactions on Knowledge Discovery
  from Data, 9(4), 1-27.
  """
  def __init__(self):
    super().__init__(name="Normalized Absolute Error", lower_is_better=True)

  def compute(self, p_true: np.ndarray, p_pred: np.ndarray) -> float:
    max_ae = 2.0 * (1.0 - np.min(p_true))
    if max_ae <= 0:
      # only possible when n_classes == 1 (a single class holds all the mass
      # across every class simultaneously, i.e. a degenerate 1-class problem)
      return 0.0
    return float(np.sum(np.abs(p_true - p_pred)) / max_ae)
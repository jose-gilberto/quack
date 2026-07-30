import numpy as np
from quack.metrics.base import QuantificationMetric


class KullbackLeiblerDivergence(QuantificationMetric):
  """Kullback-Leibler Divergence (KLD) between true and predicted prevalence vectors.

  Both `p_true` and `p_pred` are additively smoothed and renormalized
  (see `QuantificationMetric._smooth`) so they remain valid probability
  distributions before computing the divergence, avoiding `log(0)` /
  division issues for classes with zero true or estimated prevalence.

    p_s(c)          = (p(c) + eps) / (1 + n_classes * eps)
    KLD(p, p_hat)   = sum_c p_s(c) * log( p_s(c) / p_hat_s(c) )

  Parameters
  ----------
  epsilon : float, default = 1e-5
    Smoothing factor applied to both `p_true` and `p_pred`.

  References
  ----------
  Esuli, A. & Sebastiani, F. (2015). Optimizing text quantifiers for
  multivariate loss functions. ACM Transactions on Knowledge Discovery
  from Data, 9(4), 1-27.
  """
  def __init__(self, epsilon: float = 1e-5):
    super().__init__(name="Kullback-Leibler Divergence", lower_is_better=True)
    self.epsilon = epsilon

  def compute(self, p_true: np.ndarray, p_pred: np.ndarray) -> float:
    p_true_smoothed = self._smooth(p_true, self.epsilon)
    p_pred_smoothed = self._smooth(p_pred, self.epsilon)
    return float(np.sum(p_true_smoothed * np.log(p_true_smoothed / p_pred_smoothed)))
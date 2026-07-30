import math
import numpy as np
from quack.metrics.base import QuantificationMetric
from quack.metrics._kld import KullbackLeiblerDivergence


class NormalizedKullbackLeiblerDivergence(QuantificationMetric):
  """Normalized Kullback-Leibler Divergence (NKLD).

  Squashes the unbounded KLD into the `[0, 1)` range via a logistic-style
  transform, making it comparable across experiments/datasets:

    NKLD(p, p_hat) = max(0, 2 * exp(KLD(p, p_hat)) / (1 + exp(KLD(p, p_hat))) - 1)

  Parameters
  ----------
  epsilon : float, default = 1e-5
    Smoothing factor forwarded to the internal `KullbackLeiblerDivergence`
    instance (`self.kld`).

  References
  ----------
  Esuli, A. & Sebastiani, F. (2015). Optimizing text quantifiers for
  multivariate loss functions. ACM Transactions on Knowledge Discovery
  from Data, 9(4), 1-27.
  """
  def __init__(self, epsilon: float = 1e-5):
    super().__init__(name="Normalized Kullback-Leibler Divergence", lower_is_better=True)
    self.epsilon = epsilon
    self.kld = KullbackLeiblerDivergence(epsilon=epsilon)

  def compute(self, p_true: np.ndarray, p_pred: np.ndarray) -> float:
    # inputs are already validated by __call__, so we go straight to
    # .compute() on the internal KLD instance instead of re-validating
    # through its own __call__ (which does not accept an `eps` kwarg).
    exp_kld = math.exp(self.kld.compute(p_true, p_pred))
    return max(0.0, 2.0 * exp_kld / (1.0 + exp_kld) - 1.0)
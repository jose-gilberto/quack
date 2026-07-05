import numpy as np
import math
from quack.metrics.base import QuantificationMetric
from quack.metrics._kld import KullbackLeiblerDivergence


class NormalizedKullbackLeiblerDivergence(QuantificationMetric):
  def __init__(self, epsilon: float = 1e-5):
    super().__init__(name="Normalized Kullback-Leibler Divergence", lower_is_better=True)
    self.epsilon = epsilon
    self.kld = KullbackLeiblerDivergence()

  def compute(self, p_true: np.ndarray, p_pred: np.ndarray) -> float:
    exp_kld = math.exp(self.kld(p_true, p_pred, eps=self.epsilon))
    return max(0., 2 * exp_kld / (1 + exp_kld) - 1)

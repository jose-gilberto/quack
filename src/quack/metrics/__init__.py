from quack.metrics.base import QuantificationMetric

from quack.metrics._ae import AbsoluteError
from quack.metrics._rae import RelativeAbsoluteError
from quack.metrics._kld import KullbackLeiblerDivergence
from quack.metrics._nkld import NormalizedKullbackLeiblerDivergence

# Sigleton-instances for use
ae = AbsoluteError()
rae = RelativeAbsoluteError()
kld = KullbackLeiblerDivergence()
nkld = NormalizedKullbackLeiblerDivergence()

# Registry or factory of metrics
class MetricRegistry:
  """ Center the map of strings to metrics."""

  _registry = {
    "ae": AbsoluteError,
    "mae": AbsoluteError, # alias
    "rae": RelativeAbsoluteError,
    "kld": KullbackLeiblerDivergence,
    "nkld": NormalizedKullbackLeiblerDivergence
  }

  @classmethod
  def get(cls, name: str, **kwargs) -> QuantificationMetric:
    metric_cls = cls._registry.get(name.lower())
    if metric_cls is None:
      raise KeyError(
        f"Metric '{name}' not supported. "
        f"Available options: {list(cls._registry.keys())}"
      )
    return metric_cls(**kwargs)

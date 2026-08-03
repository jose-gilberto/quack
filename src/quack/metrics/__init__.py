# src/quack/metrics/__init__.py
from quack.metrics.base import QuantificationMetric

from quack.metrics._ae import AbsoluteError
from quack.metrics._rae import RelativeAbsoluteError
from quack.metrics._nae import NormalizedAbsoluteError
from quack.metrics._kld import KullbackLeiblerDivergence
from quack.metrics._nkld import NormalizedKullbackLeiblerDivergence

# Singleton instances for direct use
ae = AbsoluteError()
rae = RelativeAbsoluteError()
nae = NormalizedAbsoluteError()
kld = KullbackLeiblerDivergence()
nkld = NormalizedKullbackLeiblerDivergence()

# Registry / factory of metrics
class MetricRegistry:
  """ Centralizes the mapping from string keys to metric instances/classes."""

  _registry = {
    "ae": AbsoluteError,
    "mae": AbsoluteError,  # alias
    "rae": RelativeAbsoluteError,
    "nae": NormalizedAbsoluteError,
    "kld": KullbackLeiblerDivergence,
    "nkld": NormalizedKullbackLeiblerDivergence,
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

  @classmethod
  def available_metrics(cls) -> list[str]:
    return list(cls._registry.keys())
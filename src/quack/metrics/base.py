# src/quack/metrics/base.py
import warnings
from abc import ABC, abstractmethod
import numpy as np


class QuantificationMetric(ABC):
  """
  Abstract class to all quantification metrics. It uses the strategy design pattern.

  Parameters
  ----------
  name : str
    Human-readable name of the metric (used in reports/plots).
  lower_is_better : bool, default = True
    Whether lower values of this metric indicate better quantification
    performance. All metrics currently shipped with `quack` are error/
    divergence measures, so this defaults to True.
  """
  def __init__(self, name: str, lower_is_better: bool = True):
    self.name = name
    self.lower_is_better = lower_is_better

  def _validate_inputs(self, p_true: np.ndarray, p_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Ensure that all inputs are in the same format, shape and type.

    Parameters
    ----------
    p_true: np.ndarray
      Array-like with all true prevalences.
    p_pred: np.ndarray
      Array-like with all predicted prevalences.

    Raises
    ------
    ValueError
      thrown when shapes are incompatible or inputs aren't 1D.

    Returns
    -------
    (p_true, p_pred): tuple[np.ndarray, np.ndarray]
      Returns the two arrays in a numpy compatible format.
    """
    p_true = np.asarray(p_true, dtype=np.float64)
    p_pred = np.asarray(p_pred, dtype=np.float64)

    if p_true.shape != p_pred.shape:
      raise ValueError(
        f"Shape Error: p_true has a shape {p_true.shape} "
        f"and p_pred have shape {p_pred.shape}."
      )
    if p_true.ndim != 1:
      raise ValueError(
        f"Quantification metrics expect 1D prevalence vectors (one value "
        f"per class), got arrays with shape {p_true.shape}."
      )

    for label, p in (("p_true", p_true), ("p_pred", p_pred)):
      if not np.isclose(p.sum(), 1.0, atol=1e-6):
        warnings.warn(
          f"'{label}' does not sum to 1.0 (got {p.sum():.6f}); it may not "
          "represent a valid prevalence distribution and the metric result "
          "may be misleading.",
          stacklevel=3,
        )

    return p_true, p_pred

  @staticmethod
  def _smooth(p: np.ndarray, epsilon: float) -> np.ndarray:
    """Additive smoothing that keeps `p` a valid probability distribution.

    Following the convention adopted in the quantification literature
    (Forman, 2008; Esuli & Sebastiani, 2015), this both avoids division-
    by-zero / `log(0)` issues for classes absent from a bag and, unlike a
    naive `p + epsilon`, renormalizes so the smoothed vector still sums
    to exactly 1.0.

      p_s(c) = (p(c) + epsilon) / (1 + n_classes * epsilon)

    Parameters
    ----------
    p: np.ndarray
      Prevalence vector of shape `(n_classes,)`.
    epsilon: float
      Smoothing factor.

    Returns
    -------
      p_smoothed: np.ndarray
        Smoothed prevalence vector, still summing to 1.0.
    """
    n_classes = p.shape[0]
    return (p + epsilon) / (1.0 + n_classes * epsilon)

  @abstractmethod
  def compute(self, p_true: np.ndarray, p_pred: np.ndarray) -> float:
    """ Each metric implements its own logic and mathematics."""
    pass

  def __call__(self, p_true: np.ndarray, p_pred: np.ndarray) -> float:
    """ Call and perform the input validation and metric computation.

    Parameters
    ----------
    p_true: np.ndarray
      Array-like with all true prevalences.
    p_pred: np.ndarray
      Array-like with all predicted prevalences.

    Returns
    -------
    result: float
      The computed metric value.
    """
    p_true_clean, p_pred_clean = self._validate_inputs(p_true, p_pred)
    return self.compute(p_true_clean, p_pred_clean)

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(name={self.name!r}, lower_is_better={self.lower_is_better})"
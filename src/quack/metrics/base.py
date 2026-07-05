from abc import ABC, abstractmethod
import numpy as np

class QuantificationMetric(ABC):
  """
  Abstract class to all quantification metrics. It uses the strategy design pattern.
  """
  def __init__(self, name: str, lower_is_better: bool = True):
    self.name = name
    self.lower_is_better = lower_is_better

  def _validate_inputs(self, p_true: np.ndarray, p_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Secute that all inputs are in the same format, shape and type.

    Args:
      p_true (np.ndarray): Array-like with all true prevalences.
      p_pred (np.ndarray): Array-like with all predicted prevalences.

    Raises:
      ValueError: throws it when the shapes are incompatible.

    Returns:
        tuple[np.ndarray, np.ndarray]: Returns the two arrays in a numpy compatible format.
    """
    p_true = np.asarray(p_true, dtype=np.float64)
    p_pred = np.asarray(p_pred, dtype=np.float64)
    
    if p_true.shape != p_pred.shape:
      raise ValueError(
        f"Shape Error: p_true has a shape {p_true.shape} "
        f"and p_pred have shape {p_pred.shape}."
      )
    return p_true, p_pred

  @abstractmethod
  def compute(self, p_true: np.ndarray, p_pred: np.ndarray) -> float:
    """ Each metric implements its own logic and mathematics."""
    pass

  def __call__(self, p_true: np.ndarray, p_pred: np.ndarray) -> float:
    """ Call and perform the input validation and metric computation.

    Args:
      p_true (np.ndarray): Array-like with all true prevalences.
      p_pred (np.ndarray): Array-like with all predicted prevalences.

    Returns:
      float: Array-like containing each metric calculated among the two inputs.
    """
    p_true_clean, p_pred_clean = self._validate_inputs(p_true, p_pred)
    return self.compute(p_true_clean, p_pred_clean)

from abc import ABC, abstractmethod
import numpy as np


class BaseDatasetLoader(ABC):
  """ Abstract class for dataset loaders.
  """
  
  def load_dataset(self) -> tuple[np.ndarray, np.ndarray]:
    """Template method to dataset loaders.

    Returns:
      tuple[np.ndarray, np.ndarray]: features array and labels array.
    """
    raw_data = self._download_and_load()
    cleanned_data = self._preprocess(raw_data)
    X_raw, y_raw = self._split_features_and_target(cleanned_data)

    X = np.asarray(X_raw, dtype=np.float32)
    y = np.asarray(y_raw)

    return X, y
  
  @abstractmethod
  def _download_and_load(self):
    """For each dataset we customize where to download.
    """
    pass
  
  @abstractmethod
  def _preprocess(self, raw_data):
    """Each dataset implement its own specific preprocessing methods."""
    pass
  
  @abstractmethod
  def _split_features_and_target(self, data) -> tuple:
    """Each dataset known which column is the target to split.
    """
    pass

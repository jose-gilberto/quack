import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from quack.datasets._hierarchy import Hierarchy
from quack.datasets._licenses import DatasetLicense
from quack.utils import download_file, get_quack_home


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


@dataclass(frozen=True)
class DatasetMetadata:
  """Information returned by the `load_*` functions when `return_metadata=True`.

  Attributes
  ----------
  name : str
    Human-readable dataset name.
  label_names : list of str or None
    Column names of `Y` (multi-label and hierarchical loaders).
  groups : ndarray or None
    Natural bag of each instance (year, state, month...), when the source
    provides one. Instances sharing a value form a bag with a naturally
    shifted prevalence, a complement to the artificial bags of
    `quack.bag_generator`.
  split : ndarray or None
    Official split of each instance (`"train"`, `"val"`, `"test"`), if any.
  license : DatasetLicense or None
    Terms of the dataset.
  extras : dict
    Source-specific information (vote counts, bin edges, ids...).
  """
  name: str
  label_names: list[str] | None
  groups: np.ndarray | None
  split: np.ndarray | None
  license: DatasetLicense | None
  extras: dict


class BaseStructuredDatasetLoader(BaseDatasetLoader):
  """Common parent of the multi-label, hierarchical and regression loaders.

  Adds what those datasets need on top of `BaseDatasetLoader`: a cache folder
  under `data_home` (default `~/.quack_data/`), checksum-verified downloads,
  license terms (`license` class attribute) and the optional per-instance
  metadata hooks `_groups` and `_split`.

  Parameters
  ----------
  data_home : str or Path, default = None
    Root folder of the cached files. Defaults to `quack.utils.get_quack_home()`.
  force_load : bool, default = False
    Download the files again even if they are cached.
  """
  dataset_name: str = ""
  cache_dir: str = ""
  license: DatasetLicense | None = None

  def __init__(self, data_home: str | Path | None = None, force_load: bool = False):
    self.data_home = Path(data_home) if data_home is not None else get_quack_home()
    self.force_load = force_load
    self.label_names_: list[str] | None = None
    self.groups_: np.ndarray | None = None
    self.split_: np.ndarray | None = None
    self.extras_: dict = {}

  @property
  def dataset_dir(self) -> Path:
    """Folder where this dataset's files are cached."""
    path = self.data_home / (self.cache_dir or type(self).__name__.lower())
    os.makedirs(path, exist_ok=True)
    return path

  def _fetch(self, url: str, filename: str, checksum: str | None = None) -> Path:
    """Download `url` to `dataset_dir/filename` (once) and verify its checksum."""
    path = self.dataset_dir / filename
    download_file(url, str(path), exist_ok=self.force_load, checksum=checksum)
    return path

  def _require_file(self, filename: str, instructions: str) -> Path:
    """Path of a file the user must download by hand; raises with instructions if missing."""
    path = self.dataset_dir / filename
    if not path.exists():
      raise FileNotFoundError(
        f"{self.dataset_name}: {filename} not found in {self.dataset_dir}.\n{instructions}")
    return path

  def _groups(self, data) -> np.ndarray | None:
    """Natural bag of each instance. Override when the source has one."""
    return None

  def _split(self, data) -> np.ndarray | None:
    """Official split of each instance. Override when the source has one."""
    return None

  def _store_metadata(self, data, n_samples: int) -> None:
    for attr, hook in (("groups_", self._groups), ("split_", self._split)):
      value = hook(data)
      if value is not None:
        value = np.asarray(value)
        if len(value) != n_samples:
          raise ValueError(f"{attr[:-1]} has {len(value)} entries for {n_samples} instances")
      setattr(self, attr, value)

  def metadata(self) -> DatasetMetadata:
    """Metadata of the last `load_dataset` call."""
    return DatasetMetadata(self.dataset_name, self.label_names_, self.groups_, self.split_,
                           self.license, dict(self.extras_))


class BaseMultilabelDatasetLoader(BaseStructuredDatasetLoader):
  """Template for multi-label loaders.

  `_split_features_and_target` returns `(X, Y)` with `Y` a binary indicator
  matrix of shape `(n_samples, n_labels)`; `_label_names` names its columns.
  Text datasets set `is_text = True` and return the documents as `X`.
  """
  is_text: bool = False

  def load_dataset(self) -> tuple[np.ndarray, np.ndarray]:
    """Template method for multi-label loaders.

    Returns
    -------
    X : ndarray
      Features as float32, or an object array of strings for text datasets.
    Y : ndarray of shape (n_samples, n_labels)
      Binary indicator matrix (int8).
    """
    raw_data = self._download_and_load()
    clean_data = self._preprocess(raw_data)
    X_raw, Y_raw = self._split_features_and_target(clean_data)
    X = _as_features(X_raw, self.is_text)
    Y = np.asarray(Y_raw.toarray() if sp.issparse(Y_raw) else Y_raw, dtype=np.int8)
    if Y.ndim != 2 or Y.shape[0] != X.shape[0]:
      raise ValueError(f"Y must be (n_samples, n_labels) with {X.shape[0]} rows, got {Y.shape}")
    self.label_names_ = list(self._label_names(clean_data))
    if len(self.label_names_) != Y.shape[1]:
      raise ValueError(f"{len(self.label_names_)} label names for {Y.shape[1]} label columns")
    self._store_metadata(clean_data, X.shape[0])
    return X, Y

  @abstractmethod
  def _label_names(self, data) -> list[str]:
    """Names of the label columns, in the order of `Y`."""
    pass


class BaseHierarchicalDatasetLoader(BaseStructuredDatasetLoader):
  """Template for hierarchical loaders.

  `_split_features_and_target` returns `(X, label_sets)`, where each label set
  lists the classes of one instance (any depth); `_build_hierarchy` returns
  the taxonomy. The template turns the label sets into an indicator matrix
  over every class of the hierarchy, closed upward.
  """
  is_text: bool = False

  def load_dataset(self) -> tuple[np.ndarray, np.ndarray, Hierarchy]:
    """Template method for hierarchical loaders.

    Returns
    -------
    X : ndarray or sparse matrix
      Features as float32 (sparse for very high-dimensional data), or an
      object array of strings for text datasets.
    Y : ndarray of shape (n_samples, n_classes)
      Indicator matrix over `hierarchy.nodes`, closed upward (int8).
    hierarchy : Hierarchy
      The taxonomy.
    """
    raw_data = self._download_and_load()
    clean_data = self._preprocess(raw_data)
    X_raw, label_sets = self._split_features_and_target(clean_data)
    X = _as_features(X_raw, self.is_text)
    hierarchy = self._build_hierarchy(clean_data)
    Y = hierarchy.label_matrix(label_sets)
    if Y.shape[0] != X.shape[0]:
      raise ValueError(f"{Y.shape[0]} label sets for {X.shape[0]} instances")
    self.label_names_ = list(hierarchy.nodes)
    self._store_metadata(clean_data, X.shape[0])
    return X, Y, hierarchy

  @abstractmethod
  def _build_hierarchy(self, data) -> Hierarchy:
    """Taxonomy of the dataset's classes."""
    pass


class BaseRegressionDatasetLoader(BaseStructuredDatasetLoader):
  """Template for loaders with a continuous target (quantification for regression).

  Same contract as `BaseDatasetLoader`, but `y` is returned as float64. Use
  `make_ordinal` to bin the target into ordinal classes.
  """

  def load_dataset(self) -> tuple[np.ndarray, np.ndarray]:
    """Template method for regression loaders.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
      Features as float32.
    y : ndarray of shape (n_samples,)
      Continuous target as float64.
    """
    raw_data = self._download_and_load()
    clean_data = self._preprocess(raw_data)
    X_raw, y_raw = self._split_features_and_target(clean_data)
    X = np.asarray(X_raw, dtype=np.float32)
    y = np.asarray(y_raw, dtype=np.float64)
    self._store_metadata(clean_data, X.shape[0])
    return X, y


def _as_features(X_raw, is_text: bool):
  if is_text:
    return np.asarray(list(X_raw), dtype=object)
  if sp.issparse(X_raw):
    return sp.csr_matrix(X_raw, dtype=np.float32)
  return np.asarray(X_raw, dtype=np.float32)


def make_ordinal(y: np.ndarray, n_bins: int = 10, min_count: int = 1000,
                 grid: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Bin a continuous target into `n_bins` equal-width ordinal classes.

  Follows the recipe of [1] for turning regression datasets into ordinal
  quantification ones: equal-width bins, each with at least `min_count`
  items, dropping the items that fall outside the bins. The exact range used
  in [1] is not published; this function searches a grid of quantiles for
  the range that keeps the most items while meeting `min_count`.

  Parameters
  ----------
  y : array-like of shape (n_samples,)
    Continuous target.
  n_bins : int, default = 10
    Number of ordinal classes.
  min_count : int, default = 1000
    Minimum number of items per class.
  grid : int, default = 60
    Number of candidate quantiles on each side of the median.

  Returns
  -------
  classes : ndarray of shape (n_kept,)
    Ordinal class (0 to `n_bins - 1`) of each kept item.
  keep : ndarray of shape (n_samples,)
    Boolean mask of the kept items.
  edges : ndarray of shape (n_bins + 1,)
    Bin edges.

  Raises
  ------
  ValueError
    If no range gives every bin at least `min_count` items.

  References
  ----------
  [1] Bunse, M., Moreo, A., Sebastiani, F., & Senz, M. (2024).
      Regularization-based methods for ordinal quantification.
      Data Mining and Knowledge Discovery, 38, 4076-4121.

  Examples
  --------
  >>> from quack.datasets import load_regression, make_ordinal
  >>> X, y = load_regression("fried")
  >>> classes, keep, edges = make_ordinal(y)
  >>> X_ordinal = X[keep]
  """
  y = np.asarray(y, dtype=float)
  ys = np.sort(y[np.isfinite(y)])
  if len(ys) < n_bins * min_count:
    raise ValueError(f"only {len(ys)} finite values; need at least {n_bins * min_count}")
  qs = np.linspace(0, 0.5, grid, endpoint=False)
  best = None
  for lo in np.unique(np.quantile(ys, qs)):
    for hi in np.unique(np.quantile(ys, 1 - qs)):
      if hi <= lo:
        continue
      edges = np.linspace(lo, hi, n_bins + 1)
      pos = np.searchsorted(ys, edges, side="left")
      pos[-1] = np.searchsorted(ys, hi, side="right")
      counts = np.diff(pos)
      if counts.min() >= min_count and (best is None or counts.sum() > best[0]):
        best = (counts.sum(), edges)
  if best is None:
    raise ValueError("no range gives every bin at least min_count items; lower min_count or n_bins")
  edges = best[1]
  keep = np.isfinite(y) & (y >= edges[0]) & (y <= edges[-1])
  classes = np.clip(np.searchsorted(edges, y[keep], side="right") - 1, 0, n_bins - 1)
  return classes.astype(int), keep, edges

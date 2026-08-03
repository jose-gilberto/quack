from abc import ABC, abstractmethod
from typing import Generator
import numpy as np
from sklearn.utils import check_X_y
from sklearn.base import BaseEstimator


class BaseBagGenerator(ABC, BaseEstimator):
  """Abstract base class for all dataset-shift bag generators.

  A bag generator draws repeated "bags" (labeled subsets) of a fixed
  size from a labeled dataset `(X, y)`, simulating a specific kind of
  dataset shift between training and test-time distributions. Bags
  produced this way follow the standard evaluation protocol in the
  quantification literature (the "Artificial Prevalence Protocol", APP),
  letting quantifiers be benchmarked across the full spectrum of
  possible test-time class prevalences (or covariate shifts) rather than
  relying on a single fixed train/test split.

  Parameters
  ----------
  n_bags : int, default = 100
    Number of bags to generate.
  bag_size : int, default = None
    Number of instances per bag. If None, defaults to `len(y)` (the size
    of the original dataset).
  random_state : int, RandomState instance or None, default = None
    Controls the randomness of the bag sampling process. Pass an int for
    reproducible bags across repeated calls to `generate` — particularly
    useful when comparing multiple quantifiers on the exact same
    sequence of bags (e.g. for `quack.visualization.prevalence_plot`).
  """

  def __init__(self, n_bags: int = 100, bag_size: int = None, random_state=None):
    self.n_bags = n_bags
    self.bag_size = bag_size
    self.random_state = random_state

  @staticmethod
  def _group_indices_by_class(y: np.ndarray, classes: np.ndarray) -> dict:
    """Maps each class label to the array of dataset indices holding it."""
    return {c: np.flatnonzero(y == c) for c in classes}

  def _validate(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X, y = check_X_y(X, y, accept_sparse=True)
    if self.n_bags <= 0:
      raise ValueError(f"n_bags must be a positive integer, got {self.n_bags}.")
    if self.bag_size is not None and self.bag_size <= 0:
      raise ValueError(f"bag_size must be a positive integer, got {self.bag_size}.")
    return X, y

  @abstractmethod
  def generate(self, X: np.ndarray, y: np.ndarray) -> Generator[tuple, None, None]:
    """Lazily yields `n_bags` labeled bags `(X_bag, y_bag)` drawn from `(X, y)`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
      The pool of features to draw bags from.
    y : array-like of shape (n_samples,)
      The corresponding labels.

    Yields
    ------
    X_bag : ndarray of shape (bag_size, n_features)
      Feature matrix of a single generated bag.
    y_bag : ndarray of shape (bag_size,)
      Corresponding labels for the generated bag.
    """
    pass

  def to_list(self, X: np.ndarray, y: np.ndarray) -> list[tuple]:
    """Eagerly materializes `generate(X, y)` into a list of `(X_bag, y_bag)`.

    Useful when the same set of bags needs to be iterated multiple times
    (e.g. once per quantifier being benchmarked), since generators can
    otherwise only be consumed once.

    Returns
    -------
    bags : list[tuple[np.ndarray, np.ndarray]]
      List of `(X_bag, y_bag)` pairs, of length `n_bags`.
    """
    return list(self.generate(X, y))
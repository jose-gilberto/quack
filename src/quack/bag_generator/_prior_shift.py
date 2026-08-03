from typing import Generator
import numpy as np
from sklearn.utils import check_random_state
from quack.bag_generator.base import BaseBagGenerator


class PriorShiftBagGenerator(BaseBagGenerator):
  """Simulates Prior Probability Shift by resampling bags across the
  class-prevalence simplex, preserving `P(X|y)`.

  For each bag, a target class-prevalence vector `p` is sampled (see
  `sampling_strategy`) and then, independently for each class `c`,
  `round(p[c] * bag_size)` instances are drawn from the pool of original
  instances of class `c` — so the class-conditional feature distribution
  `P(X|y=c)` is left untouched and only the marginal `P(y)` is shifted.
  This is the standard "Artificial Prevalence Protocol" (APP) used to
  benchmark quantifiers.

  Parameters
  ----------
  n_bags : int, default = 100
    Number of bags to generate.
  bag_size : int, default = None
    Number of instances per bag. If None, defaults to `len(y)`.
  sampling_strategy : {'uniform', 'dirichlet'}, default = 'uniform'
    Strategy used to sample each bag's target prevalence vector:
    - `'uniform'`: samples uniformly over the full probability simplex
      via the standard Kraemer algorithm (sorting `n_classes - 1`
      independent `Uniform(0, 1)` cut points).
    - `'dirichlet'`: samples from a `Dirichlet(dirichlet_alpha)`
      distribution, allowing control over how extreme/skewed the shifts
      are via `dirichlet_alpha` (values < 1 favor prevalences
      concentrated in a single class; values > 1 favor prevalences
      closer to uniform).
  dirichlet_alpha : float | array-like of shape (n_classes,), default = 1.0
    Concentration parameter(s) for the Dirichlet distribution. Only used
    when `sampling_strategy='dirichlet'`. A scalar is broadcast to all
    classes; `alpha=1.0` for every class is equivalent to `'uniform'`.
  with_replacement : bool, default = True
    Whether instances are drawn with replacement from each class pool.
    Automatically forced to True for a given class/bag whenever the
    requested count exceeds the number of available instances of that
    class, regardless of this setting.
  random_state : int, RandomState instance or None, default = None
    Controls the randomness of both the prevalence sampling and the
    instance resampling.

  Attributes
  ----------
  classes_ : ndarray of shape (n_classes,)
    The distinct class labels found in `y` the last time `generate` was
    called.
  sampled_prevalences_ : ndarray of shape (n_bags, n_classes)
    The realized class prevalence of each generated bag's `y_bag` (i.e.
    the target prevalence after the largest-remainder integer rounding).

  References
  ----------
  George Forman. Quantifying counts and costs via classification.
  Data Mining and Knowledge Discovery, 17(2):164-206, 2008.

  Examples
  --------
  >>> from sklearn.datasets import make_classification
  >>> from quack.bag_generator import PriorShiftBagGenerator
  >>> X, y = make_classification(n_samples=500, n_classes=2, random_state=0)
  >>> generator = PriorShiftBagGenerator(n_bags=5, bag_size=100, random_state=0)
  >>> bags = generator.to_list(X, y)
  >>> len(bags)
  5
  >>> generator.sampled_prevalences_.shape
  (5, 2)
  """

  def __init__(self,
               n_bags: int = 100,
               bag_size: int = None,
               sampling_strategy: str = 'uniform',
               dirichlet_alpha: float = 1.0,
               with_replacement: bool = True,
               random_state=None):
    super().__init__(n_bags=n_bags, bag_size=bag_size, random_state=random_state)
    self.sampling_strategy = sampling_strategy
    self.dirichlet_alpha = dirichlet_alpha
    self.with_replacement = with_replacement

  def _sample_prevalence(self, n_classes: int, rng) -> np.ndarray:
    """Draws a single target prevalence vector summing to 1.0."""
    if self.sampling_strategy == 'uniform':
      if n_classes == 1:
        return np.ones(1)
      cuts = np.sort(rng.uniform(0.0, 1.0, size=n_classes - 1))
      cuts = np.concatenate(([0.0], cuts, [1.0]))
      return np.diff(cuts)

    if self.sampling_strategy == 'dirichlet':
      alpha = self.dirichlet_alpha
      alpha = np.full(n_classes, alpha, dtype=float) if np.isscalar(alpha) else np.asarray(alpha, dtype=float)
      if alpha.shape[0] != n_classes:
        raise ValueError(
          f"dirichlet_alpha must be a scalar or have length n_classes={n_classes}, "
          f"got length {alpha.shape[0]}."
        )
      return rng.dirichlet(alpha)

    raise ValueError(
      f"Unknown sampling_strategy '{self.sampling_strategy}'. "
      "Supported options are 'uniform' and 'dirichlet'."
    )

  @staticmethod
  def _prevalence_to_counts(prevalence: np.ndarray, bag_size: int) -> np.ndarray:
    """Converts a real-valued prevalence vector into integer per-class
    counts summing to exactly `bag_size`, via the largest-remainder method."""
    raw_counts = prevalence * bag_size
    counts = np.floor(raw_counts).astype(int)

    remainder = bag_size - counts.sum()
    if remainder > 0:
      fractional_parts = raw_counts - counts
      top_indices = np.argsort(fractional_parts)[::-1][:remainder]
      counts[top_indices] += 1

    return counts

  def generate(self, X: np.ndarray, y: np.ndarray) -> Generator[tuple, None, None]:
    X, y = self._validate(X, y)
    rng = check_random_state(self.random_state)

    self.classes_ = np.unique(y)
    n_classes = len(self.classes_)
    class_pools = self._group_indices_by_class(y, self.classes_)

    bag_size = self.bag_size if self.bag_size is not None else len(y)
    self.sampled_prevalences_ = np.zeros((self.n_bags, n_classes))

    for i in range(self.n_bags):
      prevalence = self._sample_prevalence(n_classes, rng)
      counts = self._prevalence_to_counts(prevalence, bag_size)
      self.sampled_prevalences_[i] = counts / bag_size

      bag_indices = []
      for c_idx, count in enumerate(counts):
        if count == 0:
          continue
        pool = class_pools[self.classes_[c_idx]]
        replace = self.with_replacement or count > len(pool)
        bag_indices.append(rng.choice(pool, size=count, replace=replace))

      bag_indices = np.concatenate(bag_indices)
      rng.shuffle(bag_indices)

      yield X[bag_indices], y[bag_indices]

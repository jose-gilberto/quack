from typing import Generator
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_random_state
from quack.bag_generator.base import BaseBagGenerator


class CovariateShiftBagGenerator(BaseBagGenerator):
  """Simulates Covariate Shift by resampling bags biased towards random
  regions of the feature space, preserving `P(y|X)`.

  For each bag, a random "pivot" instance is drawn from the dataset and
  every instance's RBF kernel similarity to that pivot is computed:

    k(x, x_pivot) = exp( -gamma * ||x - x_pivot||^2 )

  Instances are then resampled with probability proportional to their
  similarity to the pivot, concentrating the bag around a random region
  of the feature space. Since instances — and their original labels —
  are drawn as-is (no label is ever altered), the conditional
  distribution `P(y|X)` is left untouched; only the marginal feature
  distribution `P(X)` (and, as a natural consequence in most real
  datasets, the marginal `P(y)` too) is shifted.

  Parameters
  ----------
  n_bags : int, default = 100
    Number of bags to generate.
  bag_size : int, default = None
    Number of instances per bag. If None, defaults to `len(y)`.
  gamma : float, default = None
    RBF kernel coefficient. Controls how concentrated each bag is around
    its pivot: larger values produce bags tightly clustered in feature
    space (stronger shift); smaller values approach the original,
    unshifted distribution. If None, defaults to `1 / n_features`
    (scikit-learn's `rbf_kernel` default).
  with_replacement : bool, default = True
    Whether instances are drawn with replacement. Automatically forced
    to True whenever `bag_size` exceeds the number of available
    instances, regardless of this setting.
  random_state : int, RandomState instance or None, default = None
    Controls the randomness of both the pivot selection and the
    instance resampling.

  Attributes
  ----------
  classes_ : ndarray of shape (n_classes,)
    The distinct class labels found in `y` the last time `generate` was
    called.
  pivot_indices_ : ndarray of shape (n_bags,)
    The dataset index of the pivot instance used to build each bag.
  sampled_prevalences_ : ndarray of shape (n_bags, n_classes)
    The realized class prevalence of each generated bag's `y_bag` — a
    side effect of the covariate shift, not directly controlled.

  References
  ----------
  Bickel, S., Brückner, M., & Scheffer, T. (2009). Discriminative
  learning under covariate shift. Journal of Machine Learning Research,
  10, 2137-2155.

  Examples
  --------
  >>> from sklearn.datasets import make_classification
  >>> from quack.bag_generator import CovariateShiftBagGenerator
  >>> X, y = make_classification(n_samples=500, n_classes=2, random_state=0)
  >>> generator = CovariateShiftBagGenerator(n_bags=5, bag_size=100, gamma=0.5, random_state=0)
  >>> bags = generator.to_list(X, y)
  >>> len(bags)
  5
  """

  def __init__(self,
               n_bags: int = 100,
               bag_size: int = None,
               gamma: float = None,
               with_replacement: bool = True,
               random_state=None):
    super().__init__(n_bags=n_bags, bag_size=bag_size, random_state=random_state)
    self.gamma = gamma
    self.with_replacement = with_replacement

  def generate(self, X: np.ndarray, y: np.ndarray) -> Generator[tuple, None, None]:
    X, y = self._validate(X, y)
    rng = check_random_state(self.random_state)

    self.classes_ = np.unique(y)
    n_classes = len(self.classes_)
    n_samples = X.shape[0]

    bag_size = self.bag_size if self.bag_size is not None else n_samples
    replace = self.with_replacement or bag_size > n_samples

    self.pivot_indices_ = np.zeros(self.n_bags, dtype=int)
    self.sampled_prevalences_ = np.zeros((self.n_bags, n_classes))

    for i in range(self.n_bags):
      pivot_idx = rng.randint(n_samples)
      self.pivot_indices_[i] = pivot_idx

      similarities = rbf_kernel(X, X[pivot_idx].reshape(1, -1), gamma=self.gamma).ravel()
      total_similarity = similarities.sum()
      weights = (similarities / total_similarity if total_similarity > 0
                else np.full(n_samples, 1.0 / n_samples))

      bag_indices = rng.choice(n_samples, size=bag_size, replace=replace, p=weights)
      self.sampled_prevalences_[i] = (y[bag_indices][:, None] == self.classes_[None, :]).mean(axis=0)

      yield X[bag_indices], y[bag_indices]
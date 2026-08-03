import numpy as np
import cvxpy as cvx
from sklearn.metrics import pairwise_distances_chunked
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from quack.quantifiers.base import BaseQuantifier


def _sum_pairwise_distances(samples_i: np.ndarray, samples_j: np.ndarray) -> float:
  """Sums all pairwise distances between two sample blocks, in memory-safe chunks.

  Defined at module level (rather than as a closure/method) so it can be
  pickled and dispatched to worker processes by `joblib`/`Parallel`, the
  same pattern used for the per-fold jobs in `quack.quantifiers.base`.
  """
  return float(sum(np.sum(chunk) for chunk in pairwise_distances_chunked(samples_i, samples_j)))


class ED(BaseQuantifier):
  """Energy Distance Minimization (ED) Quantifier.

  A non-parametric, feature-space mixture model that estimates target class 
  prevalences by minimizing the Energy Distance divergence between the joint 
  training distributions and the unlabelled test batch. It uses an exact 
  analytical solution for binary settings and a quadratic programming solver 
  (via CVXPY) for multiclass problems.

  Parameters
  ----------
  n_jobs : int, default = None
    Number of jobs to run in parallel while computing the pairwise-distance
    sums that make up `class_distances_matrix_` (during `fit`) and
    `test_cross_distances` (during `predict`). Each `(class_i, class_j)`
    pair (fit) or `(class_i, test_bag)` pair (predict) is mutually
    independent, so they are dispatched as independent `joblib` jobs.
    `None` means sequential (matching the previous behavior); `-1` uses
    all available processors.
  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the distance-sum jobs.

  Attributes
  ----------
  classes_ : ndarray of shape (n_classes,)
    The distinct class labels found during the training phase.

  n_classes_ : int
    The total number of unique classes.

  train_class_samples_ : list of ndarray
    A list of length `n_classes_` where each element stores a subset of the 
    training feature matrix belonging strictly to that specific class index.

  class_distances_matrix_ : ndarray of shape (n_classes, n_classes)
    Matrix 'A' representing the expected cross-class average pairwise distances 
    calculated across the training data subsets.

  quadratic_matrix_ : ndarray of shape (n_classes - 1, n_classes - 1)
    Matrix 'B' storing the transformed quadratic form coefficients used to 
    solve multiclass optimization steps. Only populated if `n_classes_ > 2`.

  References
  ----------
  Hideko Kawakubo, Marthinus Christoffel du Plessis, and Masashi Sugiyama.
  Computationally efficient class-prior estimation under class balance change using
  energy distance. IEICE Transactions on Information and Systems, 99(1):176-186, 2016.

  Examples
  --------
  >>> from sklearn.datasets import make_classification
  >>> X, y = make_classification(n_samples=500, n_classes=2, random_state=0)
  >>> quantifier = ED()
  >>> quantifier.fit(X, y)
  >>> X_test, _ = make_classification(n_samples=100, n_classes=2, random_state=7)
  >>> prevalences = quantifier.predict(X_test)
  """

  def __init__(self, n_jobs: int = None, parallel_backend: str = "loky"):
    # energy distance operates directly on raw features, bypassing an underlying classifier
    super().__init__(classifier=None)
    self.n_jobs = n_jobs
    self.parallel_backend = parallel_backend
    self.class_distances_matrix_ = None
    self.quadratic_matrix_ = None
    self.train_class_samples_ = None

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'ED':
    """Fits the ED quantifier by computing expected intra-class pairwise distances.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
      The training feature matrix.
    y : ndarray of shape (n_samples,)
      The target class labels.

    Returns
    -------
    self : object
      Returns the instance itself.
    """
    X, y = check_X_y(X, y, accept_sparse=False)
    self.classes_ = np.unique(y)
    self.n_classes_ = len(self.classes_)

    if self.n_classes_ < 2:
      raise ValueError("Energy Distance requires at least 2 distinct classes.")

    # isolate training coordinates grouped by class
    self.train_class_samples_ = [X[y == class_label] for class_label in self.classes_]
    class_sizes = np.array([samples.shape[0] for samples in self.train_class_samples_])

    # dispatch every upper-triangular (i, j) pair as an independent job,
    # since each pairwise distance sum only depends on its own two class
    # blocks; matters most when n_classes_ or the class blocks are large
    pairs = [(i, j) for i in range(self.n_classes_) for j in range(i, self.n_classes_)]
    jobs = [
      delayed(_sum_pairwise_distances)(self.train_class_samples_[i], self.train_class_samples_[j])
      for i, j in pairs
    ]
    pair_sums = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(jobs)

    self.class_distances_matrix_ = np.zeros((self.n_classes_, self.n_classes_))
    for (i, j), total_distance in zip(pairs, pair_sums):
      value = total_distance / (class_sizes[i] * class_sizes[j])
      self.class_distances_matrix_[i, j] = value
      self.class_distances_matrix_[j, i] = value  # exploit symmetry

    # construct the optimization matrix (Matrix B) for multi-dimensional spaces
    if self.n_classes_ > 2:
      last_idx = self.n_classes_ - 1
      A = self.class_distances_matrix_
      # fully vectorized: quadratic_matrix_[i, j] = -A[i,j] + A[i,last] + A[last,j] - A[last,last].
      # Since A is symmetric this expression is automatically symmetric in
      # (i, j) too, matching the previous loop's explicit upper-triangle-then-mirror.
      self.quadratic_matrix_ = (
        -A[:last_idx, :last_idx]
        + A[:last_idx, last_idx][:, np.newaxis]
        + A[last_idx, :last_idx][np.newaxis, :]
        - A[last_idx, last_idx]
      )

    return self

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Estimates class prevalences for the given unlabelled test data batch.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
      The testing data matrix.

    Returns
    -------
    final_prevalences : ndarray of shape (n_classes,)
      A normalized probability vector indicating the estimated prevalence 
      proportions for each class.
    """
    check_is_fitted(self)
    X = check_array(X, accept_sparse=False)

    n_test_samples = X.shape[0]
    class_sizes = np.array([samples.shape[0] for samples in self.train_class_samples_])

    # compute the average distance profile from each training class subset
    # to the test bag; independent per class, dispatched as parallel jobs
    jobs = [
      delayed(_sum_pairwise_distances)(self.train_class_samples_[i], X)
      for i in range(self.n_classes_)
    ]
    class_sums = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(jobs)
    test_cross_distances = np.array(class_sums) / (class_sizes * n_test_samples)

    A = self.class_distances_matrix_
    s = test_cross_distances

    # Route A: analytical optimization for binary scenarios (n_classes < 3)
    if self.n_classes_ < 3:
      p = (s[1] - s[0] + A[0, 1] - A[1, 1]) / (-A[0, 0] + 2 * A[0, 1] - A[1, 1])

      if p < 0:
        return np.array([0.0, 1.0])
      if p > 1:
        return np.array([1.0, 0.0])
      return np.array([p, 1.0 - p])

    # Route B: constrained Quadratic Programming optimization for multiclass scenarios
    else:
      last_idx = self.n_classes_ - 1
      # fully vectorized: linear_vector[i] = -s[i] + A[i,last] + s[last] - A[last,last]
      linear_vector = -s[:last_idx] + A[:last_idx, last_idx] + s[last_idx] - A[last_idx, last_idx]

      # set up the constrained convex problem: minimize (P.T @ B @ P) - (2 * P.T @ t)
      estimated_proportions = cvx.Variable(last_idx)
      constraints = [estimated_proportions >= 0, cvx.sum(estimated_proportions) <= 1.0]

      objective_function = cvx.Minimize(
        cvx.quad_form(estimated_proportions, self.quadratic_matrix_) - 2 * estimated_proportions.T @ linear_vector
      )
      problem = cvx.Problem(objective_function, constraints)
      problem.solve()

      # post-process and append the pivot remaining probability profile element
      solved_proportions = np.clip(np.array(estimated_proportions.value).squeeze(), 0.0, 1.0)
      final_prevalences = np.append(solved_proportions, 1.0 - np.sum(solved_proportions))

      return final_prevalences
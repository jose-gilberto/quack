import numpy as np
import cvxpy as cvx
from sklearn.metrics import pairwise_distances_chunked
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from quack.quantifiers.base import BaseQuantifier


class ED(BaseQuantifier):
  """Energy Distance Minimization (ED) Quantifier.

  A non-parametric, feature-space mixture model that estimates target class 
  prevalences by minimizing the Energy Distance divergence between the joint 
  training distributions and the unlabelled test batch. It uses an exact 
  analytical solution for binary settings and a quadratic programming solver 
  (via CVXPY) for multiclass problems.

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
  """

  def __init__(self):
    # energy distance operates directly on raw features, bypassing an underlying classifier
    super().__init__(classifier=None)
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

    # initialize and populate the cross-class expected pairwise distance matrix (Matrix A)
    self.class_distances_matrix_ = np.zeros((self.n_classes_, self.n_classes_))
    for i in range(self.n_classes_):
      for j in range(i, self.n_classes_):
        samples_i, samples_j = self.train_class_samples_[i], self.train_class_samples_[j]
        count_i, count_j = samples_i.shape[0], samples_j.shape[0]
        
        # linearly accumulate chunked combinations to minimize RAM overhead
        total_pairwise_distance = sum(np.sum(chunk) for chunk in pairwise_distances_chunked(samples_i, samples_j))
        self.class_distances_matrix_[i, j] = total_pairwise_distance / (count_i * count_j)
        
        # exploit symmetry properties to populate lower triangular blocks
        if j > i:
          self.class_distances_matrix_[j, i] = self.class_distances_matrix_[i, j]

    # construct the optimization matrix (Matrix B) for multi-dimensional spaces
    if self.n_classes_ > 2:
      last_idx = self.n_classes_ - 1
      self.quadratic_matrix_ = np.zeros((last_idx, last_idx))
      A = self.class_distances_matrix_

      for i in range(last_idx):
        for j in range(i, last_idx):
          self.quadratic_matrix_[i, j] = - A[i, j] + A[i, last_idx] + A[last_idx, j] - A[last_idx, last_idx]
          if j > i:
            self.quadratic_matrix_[j, i] = self.quadratic_matrix_[i, j]

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
    test_cross_distances = np.zeros(self.n_classes_)

    # compute the average distance profile from each training class subset to the test bag
    for i in range(self.n_classes_):
      samples_i = self.train_class_samples_[i]
      count_i = samples_i.shape[0]
      total_distance = sum(np.sum(chunk) for chunk in pairwise_distances_chunked(samples_i, X))
      test_cross_distances[i] = total_distance / (count_i * n_test_samples)

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
      linear_vector = np.zeros(last_idx)
      for i in range(last_idx):
        linear_vector[i] = - s[i] + A[i, last_idx] + s[last_idx] - A[last_idx, last_idx]

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

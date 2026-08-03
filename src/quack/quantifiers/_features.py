# src/quack/quantifiers/_features.py
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from quack.quantifiers.base import BaseMixtureQuantifier, BaseQuantifier


class HDx(BaseMixtureQuantifier):
  """Hellinger Distance x (HDx) quantifier.

  HDx is a non-parametric feature-space mixture model that operates directly on 
  categorical or discretized continuous features without training an underlying 
  classifier. It projects each feature column independently, constructs a global 
  marginal conditional probability matrix during training, and minimizes the 
  Hellinger Distance to estimate the test class prevalences.

  Parameters
  ----------
  use_convex_solver : bool, default=True
    If True, attempts to solve the statistical mixture distribution using `cvxpy`.
    If False, falls back to the Golden Section Search numerical solver.

  Attributes
  ----------
  classes_ : ndarray of shape (n_classes,)
    The distinct class labels found during the training phase.

  n_classes_ : int
    The total number of unique classes.

  train_prevalence_ : ndarray of shape (n_classes,)
    The baseline prevalence proportion of each class observed in the training data.

  feature_spaces_ : list of ndarray
    A list of length `n_features`, where each element contains the unique sorted 
    values observed for that specific feature column during training.

  conditional_matrix_ : ndarray of shape (n_total_unique_bins, n_classes)
    The stacked conditional probability matrix built during the `fit` phase.
    Represents the marginal distribution profiles for each class.

  References
  ----------
  Víctor González-Castro, Rocío Alaiz-Rodríguez, and Enrique Alegre. Class distribution
  estimation based on the Hellinger distance. Information Sciences, 218(1):146-164, 2013
  """
  def __init__(self, use_convex_solver: bool = True):
    # HDx operates on features directly (classifier=None) and strictly uses Hellinger Distance ("HD")
    super().__init__(classifier=None,
                     distance_metric="HD",
                     use_convex_solver=use_convex_solver)
    self.feature_spaces_ = None

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'HDx':
    """Fits the HDx mixture model by building the marginal conditional matrix.

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

    self.classes_, class_counts = np.unique(y, return_counts=True)
    self.n_classes_ = len(self.classes_)
    self.train_prevalence_ = class_counts / len(y)

    if self.n_classes_ < 2:
      raise ValueError("HDx requires at least 2 distinct classes to fit.")

    n_features = X.shape[1]
    class_idx = np.searchsorted(self.classes_, y)

    # map and store the unique token space for each individual feature column
    self.feature_spaces_ = [np.unique(X[:, j]) for j in range(n_features)]

    # build the conditional matrix (CM): the loop over features is
    # unavoidable (each column has a different number of unique values,
    # so the blocks can't be stacked into a single rectangular operation),
    # but within each feature the value x class crosstab is fully
    # vectorized via a combined-index bincount instead of the previous
    # nested (class x unique_value) Python loop
    conditional_blocks = []
    for j in range(n_features):
      unique_values = self.feature_spaces_[j]
      val_idx = np.searchsorted(unique_values, X[:, j])

      combined_idx = val_idx * self.n_classes_ + class_idx
      counts_flat = np.bincount(combined_idx, minlength=len(unique_values) * self.n_classes_)
      crosstab_counts = counts_flat.reshape(len(unique_values), self.n_classes_)

      # normalize counts by each class size to form conditional probabilities
      conditional_blocks.append(crosstab_counts / class_counts)

    # vertically stack all independent column representations into a single global system matrix
    self.conditional_matrix_ = np.vstack(conditional_blocks)

    return self

  def _compute_score(self, X: np.ndarray) -> np.ndarray:
    """Extracts the empirical marginal test frequencies across all features.

    Calculates the relative sample distribution frequency over the saved 
    training feature spaces for the incoming test batch.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
      The testing feature matrix.

    Returns
    -------
    test_frequencies : ndarray of shape (n_total_unique_bins,)
      The stacked vector representing empirical marginal frequencies of the test batch.
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    frequencies_list = []

    # compute marginal frequencies matching the exact bins established
    # during training; the per-feature loop remains for the same reason
    # as in fit (ragged bin counts across columns), but the inner
    # per-unique-value counting is now a single searchsorted + bincount
    # pass instead of one np.count_nonzero comparison per unique value.
    # Test values absent from the trained feature space (i.e. not an
    # exact match to any trained unique value) contribute 0, exactly
    # matching the original np.count_nonzero(X[:, j] == val) semantics.
    for j in range(n_features):
      unique_values = self.feature_spaces_[j]
      col = X[:, j]

      idx = np.searchsorted(unique_values, col)
      idx_clipped = np.clip(idx, 0, len(unique_values) - 1)
      exact_match = unique_values[idx_clipped] == col

      counts = np.bincount(idx_clipped[exact_match], minlength=len(unique_values))
      frequencies_list.append(counts / n_samples)

    # combine the individual frequency vectors into the stacked global vector
    return np.hstack(frequencies_list)


class _RawSubspaceMixture(BaseMixtureQuantifier):
  """Internal feature-space mixture quantifier for random subspaces.

  Fits a single conditional matrix matching joint distribution row-profiles 
  over a selected feature subspace. Used exclusively as a base learner 
  inside the ReadMe ensemble.
  """

  def __init__(self, distance_metric: str = "L2", use_convex_solver: bool = True):
    super().__init__(classifier=None,
                     distance_metric=distance_metric,
                     use_convex_solver=use_convex_solver)
    self.unique_rows_ = None

  def fit(self, X: np.ndarray, y: np.ndarray) -> '_RawSubspaceMixture':
    pass

  @staticmethod
  def _binary_search_row(row: np.ndarray,
                         unique_matrix: np.ndarray,
                         start_index: int = 0):
    """Optimized multi-column binary search over a lexically sorted 2D array.

    Speeds up row-matching allocations by leveraging sequential search window 
    narrowing via np.searchsorted across active columns. Used at prediction
    time, where test rows are scanned sequentially against `unique_rows_`
    (already sorted by `np.unique` during `fit_subspace`).
    """
    current_col = 0
    n_cols = unique_matrix.shape[1]
    n_rows = unique_matrix.shape[0]

    left_bound = start_index
    right_bound = n_rows

    while left_bound + 1 < right_bound and current_col < n_cols:
      # narrow down the row window based on the current column's coordinate value
      temp_index = np.searchsorted(
        a=unique_matrix[left_bound:right_bound, current_col],
        v=row[current_col],
        side='left'
      )
      left_bound += temp_index

      temp_index = np.searchsorted(
        a=unique_matrix[left_bound:right_bound, current_col],
        v=row[current_col],
        side='right'
      )
      right_bound = left_bound + temp_index
      
      current_col += 1

    if left_bound < n_rows and np.array_equal(row, unique_matrix[left_bound, :]):
      return left_bound
    return None

  def fit_subspace(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   classes: np.ndarray,
                   class_counts: np.ndarray) -> '_RawSubspaceMixture':
    """Fits the sub-matrix profile over the targeted feature subspace.

    Delegates row deduplication and index assignment to `np.unique(...,
    axis=0, return_inverse=True)`, which performs the exact same
    lexicographic sort `_binary_search_row` assumes at prediction time,
    but as a single vectorized C-level call instead of a per-row Python
    binary search loop.
    """
    self.classes_ = classes
    self.n_classes_ = len(classes)
    self.train_prevalence_ = class_counts / len(y)

    self.unique_rows_, row_indices = np.unique(X, axis=0, return_inverse=True)
    row_indices = row_indices.ravel()

    class_to_idx = np.searchsorted(self.classes_, y)
    combined_idx = row_indices * self.n_classes_ + class_to_idx
    counts_flat = np.bincount(combined_idx, minlength=self.unique_rows_.shape[0] * self.n_classes_)

    # normalize across class columns to create valid conditional probabilities
    self.conditional_matrix_ = counts_flat.reshape(self.unique_rows_.shape[0], self.n_classes_) / class_counts
    return self

  def _compute_score(self, X: np.ndarray) -> np.ndarray:
    """Extracts joint empirical row frequencies using sorted lexical scans."""
    # lexicographically sort test matrix rows to maximize binary search lookup speed
    lexical_indices = np.lexsort(np.rot90(X))
    X_sorted = X[lexical_indices]

    row_counts = np.zeros(self.unique_rows_.shape[0])
    last_found_index = 0
    
    # scan and count matching patterns
    for i in range(X_sorted.shape[0]):
      row_index = self._binary_search_row(X_sorted[i, :], self.unique_rows_, last_found_index)
      if row_index is None:
        continue
      last_found_index = row_index
      row_counts[row_index] += 1

    return row_counts * 1.0 / X.shape[0]


def _fit_subspace_job(X: np.ndarray,
                      y: np.ndarray,
                      classes: np.ndarray,
                      class_counts: np.ndarray,
                      feature_indices: np.ndarray,
                      distance_metric: str,
                      use_convex_solver: bool) -> '_RawSubspaceMixture':
  """Fits one independent random-subspace sub-quantifier.

  Defined at module level (rather than as a closure/method) so it can be
  pickled and dispatched to worker processes by `joblib`/`Parallel`, the
  same pattern used for the per-fold jobs in `quack.quantifiers.base`.
  """
  sub_quantifier = _RawSubspaceMixture(distance_metric=distance_metric, use_convex_solver=use_convex_solver)
  sub_quantifier.fit_subspace(X[:, feature_indices], y, classes, class_counts)
  return sub_quantifier


def _predict_subspace_job(sub_quantifier: '_RawSubspaceMixture', X_subspace: np.ndarray) -> np.ndarray:
  """Scores one test bag against a single fitted sub-quantifier.

  Defined at module level for the same pickling reasons as `_fit_subspace_job`.
  """
  return sub_quantifier.predict(X_subspace)


class ReadMe(BaseQuantifier):
  """ReadMe Ensemble Quantifier.

  ReadMe is an ensemble mixture model specifically designed for high-dimensional 
  categorical data or short text analysis (e.g., Bag-of-Words). It circumvents the 
  curse of dimensionality by training multiple independent sub-space mixture models 
  over randomized feature subsets, obtaining final test prevalences by averaging 
  individual predictions.

  Parameters
  ----------
  distance_metric : str, default='L2'
    The distance metric minimized by internal sub-quantifiers ('L1', 'L2', 'HD', 'TS').

  use_convex_solver : bool, default=True
    If True, internal sub-quantifiers utilize `cvxpy` optimization.

  n_features : int, default=None
    Number of random features selected per subset. If None, it automatically 
    defaults to `max(int(D/5), 2)` or bit length depending on dataset dimensionality.

  n_subsets : int, default=100
    The total number of random subspace sub-quantifiers to ensemble.

  n_jobs : int, default = None
    Number of jobs to run in parallel while fitting/predicting the
    `n_subsets` independent sub-quantifiers, since none of them depend
    on each other. `None` means sequential (matching the previous
    behavior); `-1` uses all available processors. See `joblib.Parallel`.

  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the subspace jobs (`"loky"` for
    process-based parallelism, `"threading"` for thread-based).

  random_state : int, RandomState instance or None, default = None
    Controls the randomness of the per-subset feature selection. Pass an
    int for reproducible subspaces across repeated `fit` calls.

  Attributes
  ----------
  classes_ : ndarray of shape (n_classes,)
    The distinct class labels found during the training phase.

  n_classes_ : int
    The total number of unique classes.

  train_prevalence_ : ndarray of shape (n_classes,)
    The baseline prevalence proportion of each class observed in the training data.

  feature_subsets_ : list of ndarray
    A list containing the chosen feature column indices for each random subset.

  sub_quantifiers_ : list of _RawSubspaceMixture
    The collection of fitted internal mixture models making up the ensemble.

  References
  ----------
  Hopkins, D., & King, G. (2010). A method of automated nonparametric content 
  analysis for social science. American Journal of Political Science, 54(1), 229-247.
  """

  def __init__(self,
               distance_metric: str = "L2",
               use_convex_solver: bool = True, 
               n_features: int = None,
               n_subsets: int = 100,
               n_jobs: int = None,
               parallel_backend: str = "loky",
               random_state=None):
    # ReadMe manages an internal collection of sub-quantifiers, bypassing a single core classifier
    super().__init__(classifier=None)
    self.distance_metric = distance_metric
    self.use_convex_solver = use_convex_solver
    self.n_features = n_features
    self.n_subsets = n_subsets
    self.n_jobs = n_jobs
    self.parallel_backend = parallel_backend
    self.random_state = random_state
    self.feature_subsets_ = []
    self.sub_quantifiers_ = []

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'ReadMe':
    """Fits the ReadMe ensemble by training multiple subspace mixture models.

    The `n_subsets` sub-quantifiers are mutually independent, so they are
    dispatched as independent `joblib` jobs (see `n_jobs`/`parallel_backend`)
    instead of a sequential Python loop.

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
    self.classes_, class_counts = np.unique(y, return_counts=True)
    self.n_classes_ = len(self.classes_)
    self.train_prevalence_ = class_counts / len(y)

    total_features = X.shape[1]

    # dynamically determine the subspace feature size if not explicitly provided
    if self.n_features is None:
      if total_features > 25:
        self.n_features = total_features.bit_length()
      else:
        self.n_features = max(int(total_features / 5), 2)

    rng = check_random_state(self.random_state)
    self.feature_subsets_ = [
      rng.choice(total_features, self.n_features, replace=False) for _ in range(self.n_subsets)
    ]

    jobs = [
      delayed(_fit_subspace_job)(X, y, self.classes_, class_counts, feature_indices,
                                 self.distance_metric, self.use_convex_solver)
      for feature_indices in self.feature_subsets_
    ]
    self.sub_quantifiers_ = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(jobs)

    return self

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Estimates class prevalences by averaging sub-quantifier predictions.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
      The testing feature matrix.

    Returns
    -------
    ensemble_prevalences : ndarray of shape (n_classes,)
      The final aggregated and normalized prevalence estimation vector.
    """
    check_is_fitted(self)
    X = check_array(X, accept_sparse=False)

    jobs = [
      delayed(_predict_subspace_job)(sub_quantifier, X[:, feature_indices])
      for sub_quantifier, feature_indices in zip(self.sub_quantifiers_, self.feature_subsets_)
    ]
    subspace_predictions = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(jobs)

    # compute the final ensemble mean distribution
    return np.mean(subspace_predictions, axis=0)
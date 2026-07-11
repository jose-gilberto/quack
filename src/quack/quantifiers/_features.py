import numpy as np
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
    
    # map and store the unique token space for each individual feature column
    self.feature_spaces_ = [np.unique(X[:, j]) for j in range(n_features)]

    # build the conditional matrix (CM) using an optimized pure NumPy block-stacking alternative
    conditional_blocks = []
    for j in range(n_features):
      unique_values = self.feature_spaces_[j]
      crosstab_counts = np.zeros((len(unique_values), self.n_classes_))

      # Count feature occurrences per class matching the target unique space
      for class_idx, class_label in enumerate(self.classes_):
        X_class_subset = X[y == class_label, j]
        for val_idx, val in enumerate(unique_values):
          crosstab_counts[val_idx, class_idx] = np.sum(X_class_subset == val)
      
      # normalize counts by each class size to form conditional probabilities
      conditional_blocks.append(crosstab_counts / class_counts)
        
    # vertically stack all independent column representations into a single global system matrix
    self.conditional_matrix_ = np.vstack(conditional_blocks)
    
    return self


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

  @staticmethod
  def _binary_search_row(row: np.ndarray,
                         unique_matrix: np.ndarray,
                         start_index: int = 0):
    """Optimized multi-column binary search over a lexically sorted 2D array.

    Speeds up row-matching allocations by leveraging sequential search window 
    narrowing via np.searchsorted across active columns.
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
    """Fits the sub-matrix profile over the targeted feature subspace."""
    self.classes_ = classes
    self.n_classes_ = len(classes)
    self.train_prevalence_ = class_counts / len(y)

    # extract unique joint combinations (rows) present in this feature subspace
    self.unique_rows_ = np.unique(X, axis=0)
    self.conditional_matrix_ = np.zeros((self.unique_rows_.shape[0], self.n_classes_))

    # populate joint distribution frequencies per class
    class_to_index = {class_label: idx for idx, class_label in enumerate(self.classes_)}
    for i in range(len(y)):
      row_index = self._binary_search_row(X[i, :], self.unique_rows_)
      if row_index is not None:
        self.conditional_matrix_[row_index, class_to_index[y[i]]] += 1

    # normalize across class columns to create valid conditional probabilities
    self.conditional_matrix_ = self.conditional_matrix_ / class_counts
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
               n_subsets: int = 100):
    # ReadMe manages an internal collection of sub-quantifiers, bypassing a single core classifier
    super().__init__(classifier=None)
    self.distance_metric = distance_metric
    self.use_convex_solver = use_convex_solver
    self.n_features = n_features
    self.n_subsets = n_subsets
    self.feature_subsets_ = []
    self.sub_quantifiers_ = []

  def fit(self, X: np.ndarray, y: np.ndarray) -> 'ReadMe':
    """Fits the ReadMe ensemble by training multiple subspace mixture models.

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

    self.feature_subsets_ = []
    self.sub_quantifiers_ = []

    # construct and fit independent random subspace models
    for _ in range(self.n_subsets):
      chosen_feature_indices = np.random.choice(range(total_features), self.n_features, replace=False)
      self.feature_subsets_.append(chosen_feature_indices)

      sub_quantifier = _RawSubspaceMixture(
        distance_metric=self.distance_metric, 
        use_convex_solver=self.use_convex_solver
      )
      # train the subspace model using shared class counts to reduce redundant allocations
      sub_quantifier.fit_subspace(X[:, chosen_feature_indices], y, self.classes_, class_counts)
      self.sub_quantifiers_.append(sub_quantifier)

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

    ensemble_prevalences = np.zeros(self.n_classes_)
    
    # aggregate the predicted distribution vectors from all feature subsets
    for i in range(self.n_subsets):
      subspace_features = self.feature_subsets_[i]
      ensemble_prevalences += self.sub_quantifiers_[i].predict(X[:, subspace_features])

    # compute the final ensemble mean distribution
    return ensemble_prevalences / self.n_subsets

import warnings
import cvxpy as cvx
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from quack.quantifiers.base import BaseCalibratedQuantifier, BaseMixtureQuantifier, BaseScoreMixtureQuantifier


class DyS(BaseScoreMixtureQuantifier):
  """Distribution y-Similarity (DyS) Quantifier.

  An adjusting prediction mixture model built strictly for binary quantification. 
  It partitions out-of-fold continuous classification scores into a specified number 
  of histograms bins to match training and testing distributions.

  Parameters
  ----------
  classifier : estimator object, default=None
    The underlying base classifier. If None, defaults to `SVC()`.

  distance_metric : str, default='TS'
    The distance metric minimized ('L1', 'L2', 'HD', 'TS').

  n_bins : int, default=10
    The total number of histogram bins used to slice the distribution profiles.

  cv : int, default=10
    The number of cross-validation folds for out-of-fold scoring.

  use_convex_solver : bool, default=True
    If True, optimizes via CVXPY.

  predict_proba : bool, default=False
    If True, forces the model to use probabilistic `predict_proba` outputs. 
    If False, falls back to raw decision boundary scores.

  n_jobs : int, default = None
    Number of jobs to run in parallel while fitting the `cv` folds. See
    `BaseCalibratedQuantifier`.

  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the CV/final-fit jobs.

  Attributes
  ----------
  classes_ : ndarray of shape (n_classes,)
    The distinct class labels found during the training phase.

  n_classes_ : int
    The total number of unique classes.

  train_prevalence_ : ndarray of shape (n_classes,)
    The baseline prevalence proportion of each class observed in the training data.

  conditional_matrix_ : ndarray of shape (n_bins, n_classes)
    The binned score conditional matrix built using out-of-fold calibration data.

  score_range_ : tuple of float (min, max)
    The minimum and maximum boundaries used to define histogram bins.

  References
  ----------
  André Maletzke, Denis dos Reis, Everton Cherman, and Gustavo Batista. DyS: A framework
  for mixture models in quantification. In Proceedings of the AAAI Conference on Artificial
  Intelligence, pages 4552-4560, Honolulu, Hawaii, 2019.
  """

  def __init__(self,
               classifier: BaseEstimator = SVC(),
               distance_metric: str = "TS",
               n_bins: int = 10, cv: int = 10, use_convex_solver: bool = True,
               predict_proba: bool = False, n_jobs: int = None, parallel_backend: str = "loky"):
    super().__init__(classifier=classifier, distance_metric=distance_metric, cv=cv,
                     use_convex_solver=use_convex_solver, predict_proba=predict_proba,
                     n_jobs=n_jobs, parallel_backend=parallel_backend)
    self.n_bins = n_bins
    self.score_range_ = None

  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    y_scores = self._extract_1d_scores(y_pred_oof)
    self.score_range_ = (0.0, 1.0) if self.predict_proba else (np.min(y_scores), np.max(y_scores))
    
    conditional_blocks = []
    for class_label in self.classes_:
      class_mask = (y_true_oof == class_label)
      counts, _ = np.histogram(y_scores[class_mask], bins=self.n_bins, range=self.score_range_)
      conditional_blocks.append(counts)
        
    _, class_counts = np.unique(y_true_oof, return_counts=True)
    self.conditional_matrix_ = np.vstack(conditional_blocks).T / class_counts

  def _compute_score(self, X: np.ndarray) -> np.ndarray:
    prediction_method = getattr(self.classifier_, self._get_oof_method())
    raw_predictions = prediction_method(X)
    y_scores = self._extract_1d_scores(raw_predictions)

    test_frequencies, _ = np.histogram(y_scores, bins=self.n_bins, range=self.score_range_)
    if not self.predict_proba:
      test_frequencies[0] += np.sum(y_scores < self.score_range_[0])
      test_frequencies[-1] += np.sum(y_scores > self.score_range_[1])

    return test_frequencies / X.shape[0]


class HDy(DyS):
  """Hellinger Distance y (HDy) Quantifier.

  A specialized instance of the DyS framework that minimizes the Hellinger 
  Distance over binned score histograms using a Logistic Regression classifier.

  Parameters
  ----------
  classifier : estimator object, default=None
      The underlying base classifier. Defaults to `LogisticRegression()`.

  n_bins : int, default=10
      The total number of histogram bins.

  cv : int, default=10
      The number of cross-validation folds.

  use_convex_solver : bool, default=True
      If True, optimizes via CVXPY.

  predict_proba : bool, default=False
      If True, forces the model to use probabilistic `predict_proba` outputs.

  n_jobs : int, default = None
    Number of jobs to run in parallel while fitting the `cv` folds.

  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the CV/final-fit jobs.

  References
  ----------
  Víctor González-Castro, Rocío Alaiz-Rodríguez, and Enrique Alegre. Class distribution
  estimation based on the Hellinger distance. Information Sciences, 218(1):146-164, 2013.
  """

  def __init__(self,
               classifier: BaseEstimator = LogisticRegression(),
               n_bins: int = 10,
               cv: int = 10,
               use_convex_solver: bool = True,
               predict_proba: bool = False,
               n_jobs: int = None,
               parallel_backend: str = "loky"):
    super().__init__(classifier=classifier, distance_metric="HD", n_bins=n_bins, 
                     cv=cv, use_convex_solver=use_convex_solver, predict_proba=predict_proba,
                     n_jobs=n_jobs, parallel_backend=parallel_backend)


class FormanMM(BaseScoreMixtureQuantifier):
  """Forman's Mixture Model (FormanMM) Quantifier.

  An adjusting binary quantifier that optimizes the `L1` distance over the 
  Cumulative Distribution Function (CDF) profiles of classification scores.

  Parameters
  ----------
  classifier : estimator object, default = None
    The underlying base classifier. Defaults to `SVC()`.

  cv : int, default = 10
    The number of cross-validation folds.

  use_convex_solver : bool, default = True
    If True, optimizes via CVXPY.

  predict_proba : bool, default = False
    If True, forces the model to use probabilistic `predict_proba` outputs.

  n_jobs : int, default = None
    Number of jobs to run in parallel while fitting the `cv` folds.

  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the CV/final-fit jobs.

  Attributes
  ----------
  classes_ : ndarray of shape (n_classes,)
    The distinct class labels found during the training phase.

  n_classes_ : int
    The total number of unique classes.

  train_prevalence_ : ndarray of shape (n_classes,)
    The baseline prevalence proportion of each class observed in the training data.

  conditional_matrix_ : ndarray of shape (n_bins, n_classes)
    The binned CDF score conditional matrix built using out-of-fold calibration data.

  bins_ : ndarray
    The array of unique out-of-fold score values used as thresholds for the CDF bins.

  References
  ----------
  George Forman. Quantifying counts and costs via classification.
  Data Mining and Knowledge Discovery, 17(2):164-206, 2008.
  """

  def __init__(self, classifier: BaseEstimator = SVC(), cv: int = 10, 
               use_convex_solver: bool = True, predict_proba: bool = False,
               n_jobs: int = None, parallel_backend: str = "loky"):
    super().__init__(classifier=classifier, distance_metric="L1", cv=cv,
                     use_convex_solver=use_convex_solver, predict_proba=predict_proba,
                     n_jobs=n_jobs, parallel_backend=parallel_backend)
    self.bins_ = None

  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    y_scores = self._extract_1d_scores(y_pred_oof)
    self.bins_ = np.unique(y_scores)

    # exclude the largest edge score to avoid overlapping conditional values equal to 1.0
    if len(self.bins_) > 1 and (self.bins_[-1] - np.finfo(float).eps > self.bins_[-2]):
      self.bins_[-1] -= np.finfo(float).eps

    conditional_blocks = []
    for class_label in self.classes_:
      class_mask = (y_true_oof == class_label)
      counts, _ = np.histogram(y_scores[class_mask], bins=self.bins_)
      conditional_blocks.append(np.cumsum(counts))

    _, class_counts = np.unique(y_true_oof, return_counts=True)
    self.conditional_matrix_ = np.vstack(conditional_blocks).T / class_counts

  def _compute_score(self, X: np.ndarray) -> np.ndarray:
    prediction_method = getattr(self.classifier_, self._get_oof_method())
    raw_predictions = prediction_method(X)
    y_scores = self._extract_1d_scores(raw_predictions)

    test_frequencies = np.cumsum(np.histogram(y_scores, bins=self.bins_)[0])
    test_frequencies += np.sum(y_scores < self.bins_[0])

    return test_frequencies / X.shape[0]


class GAC(BaseCalibratedQuantifier, BaseMixtureQuantifier):
  """Generalized Adjusting Confusion Matrix (GAC) Quantifier.

  A distance-minimizing multi-class generalization of the Adjusting Count (AC) 
  algorithm that optimizes target distributions across the discrete labels 
  confusion matrix profile.

  Parameters
  ----------
  classifier : estimator object, default = LogisticRegression
    The underlying base classifier. Defaults to `LogisticRegression()`.

  distance_metric : str, default = 'L2'
    The distance metric minimized.

  cv : int, default = 10
    The number of cross-validation folds.

  use_convex_solver : bool, default = True
    If True, optimizes via CVXPY.

  n_jobs : int, default = None
    Number of jobs to run in parallel while fitting the `cv` folds.

  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the CV/final-fit jobs.
    
  References
  ----------
  Aykut Firat. Unified framework for quantification. arXiv preprint arXiv:1606.00868, 2016.
  """

  def __init__(self,
               classifier: BaseEstimator = LogisticRegression(),
               distance_metric: str = "L2", 
               cv: int = 10,
               use_convex_solver: bool = True,
               n_jobs: int = None,
               parallel_backend: str = "loky"):
    BaseCalibratedQuantifier.__init__(self, classifier=classifier, cv=cv,
                                      n_jobs=n_jobs, parallel_backend=parallel_backend)
    BaseMixtureQuantifier.__init__(self, classifier=classifier, distance_metric=distance_metric, 
                                  use_convex_solver=use_convex_solver)

  def _get_oof_method(self) -> str:
    return "predict"

  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    # fully vectorized confusion matrix: one-hot encode both true and
    # predicted labels against the sorted classes_ grid, then a single
    # matmul (n_classes, n_samples) @ (n_samples, n_classes) tallies
    # every (pred, true) pair at once, replacing the double Python loop
    # this used to run over classes_ x classes_
    true_idx = np.searchsorted(self.classes_, y_true_oof)
    pred_idx = np.searchsorted(self.classes_, y_pred_oof)
    one_hot_true = np.eye(self.n_classes_)[true_idx]
    one_hot_pred = np.eye(self.n_classes_)[pred_idx]
    confusion_matrix = one_hot_pred.T @ one_hot_true

    _, class_counts = np.unique(y_true_oof, return_counts=True)
    self.conditional_matrix_ = confusion_matrix / class_counts

  def _compute_score(self, X: np.ndarray) -> np.ndarray:
    y_predictions = self.classifier_.predict(X)
    return np.array([np.mean(y_predictions == class_label) for class_label in self.classes_])

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    test_frequencies = self._compute_score(X)
    return self._solve_mixture(test_frequencies)


class GPAC(BaseCalibratedQuantifier, BaseMixtureQuantifier):
  """Generalized Probabilistic Adjusting Confusion Matrix (GPAC) Quantifier.

  A distance-minimizing multi-class generalization of the Probabilistic Adjusting 
  Count (PAC) algorithm that optimizes target distributions using soft-probability profiles.

  Parameters
  ----------
  classifier : estimator object, default = LogisticRegression
    The underlying base classifier. Defaults to `LogisticRegression()`.

  distance_metric : str, default = 'L2'
    The distance metric minimized.

  cv : int, default = 10
    The number of cross-validation folds.

  use_convex_solver : bool, default = True
    If True, optimizes via CVXPY.

  n_jobs : int, default = None
    Number of jobs to run in parallel while fitting the `cv` folds.

  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the CV/final-fit jobs.

  References
  ----------
  Aykut Firat. Unified framework for quantification. arXiv preprint arXiv:1606.00868, 2016.
  """

  def __init__(self,
               classifier: BaseEstimator = LogisticRegression(),
               distance_metric: str = "L2", 
               cv: int = 10,
               use_convex_solver: bool = True,
               n_jobs: int = None,
               parallel_backend: str = "loky"):
    BaseCalibratedQuantifier.__init__(self, classifier=classifier, cv=cv,
                                      n_jobs=n_jobs, parallel_backend=parallel_backend)
    BaseMixtureQuantifier.__init__(self, classifier=classifier, distance_metric=distance_metric, 
                                  use_convex_solver=use_convex_solver)

  def _get_oof_method(self) -> str:
    return "predict_proba"

  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    # single matmul replaces the loop over classes_: column l of the
    # result is the sum of predicted-probability rows whose true label is
    # class l, i.e. y_pred_oof.T @ one_hot(true_labels)
    true_idx = np.searchsorted(self.classes_, y_true_oof)
    one_hot_true = np.eye(self.n_classes_)[true_idx]
    probabilistic_matrix = y_pred_oof.T @ one_hot_true

    _, class_counts = np.unique(y_true_oof, return_counts=True)
    self.conditional_matrix_ = probabilistic_matrix / class_counts

  def _compute_score(self, X: np.ndarray) -> np.ndarray:
    return self.classifier_.predict_proba(X).sum(axis=0) / X.shape[0]

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    test_frequencies = self._compute_score(X)
    return self._solve_mixture(test_frequencies)


class FM(BaseCalibratedQuantifier, BaseMixtureQuantifier):
  """Friedman's Method (FM) Quantifier.

  An adjusting prediction mixture model that maps soft classifier probabilities 
  into binary indicator matrices by comparing them against baseline training priors.

  Parameters
  ----------
  classifier : estimator object, default = LogisticRegression
    The underlying base classifier. Defaults to `LogisticRegression()`.

  distance_metric : str, default = 'L2'
    The distance metric minimized.

  cv : int, default = 10
    The number of cross-validation folds.

  use_convex_solver : bool, default = True
    If True, optimizes via CVXPY.

  n_jobs : int, default = None
    Number of jobs to run in parallel while fitting the `cv` folds.

  parallel_backend : str, default = "loky"
    `joblib.Parallel` backend used for the CV/final-fit jobs.

  References
  ----------
  Jerome H. Friedman. Class counts in future unlabeled samples, 2014.
  Presentation at MIT CSAIL Big Data Event.
  """

  def __init__(self,
               classifier: BaseEstimator = LogisticRegression(),
               distance_metric: str = "L2",
               cv: int = 10,
               use_convex_solver: bool = True,
               n_jobs: int = None,
               parallel_backend: str = "loky"):
    BaseCalibratedQuantifier.__init__(self, classifier=classifier, cv=cv,
                                      n_jobs=n_jobs, parallel_backend=parallel_backend)
    BaseMixtureQuantifier.__init__(self, classifier=classifier, distance_metric=distance_metric, 
                                  use_convex_solver=use_convex_solver)

  def _get_oof_method(self) -> str:
    return "predict_proba"

  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    # single matmul replaces the loop over classes_, mirroring GPAC's
    # vectorization but comparing against train_prevalence_ first
    true_idx = np.searchsorted(self.classes_, y_true_oof)
    one_hot_true = np.eye(self.n_classes_)[true_idx]
    above_prior = (y_pred_oof > self.train_prevalence_).astype(float)
    threshold_matrix = above_prior.T @ one_hot_true

    _, class_counts = np.unique(y_true_oof, return_counts=True)
    self.conditional_matrix_ = threshold_matrix / class_counts

  def _compute_score(self, X: np.ndarray) -> np.ndarray:
    return np.sum(self.classifier_.predict_proba(X) > self.train_prevalence_, axis=0) / X.shape[0]

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    test_frequencies = self._compute_score(X)
    return self._solve_mixture(test_frequencies)
import numpy as np
import cvxpy as cp
from quack.quantifiers.base import BaseCalibratedQuantifier
from sklearn.base import BaseEstimator


class FormanMM(BaseCalibratedQuantifier):
  """
  FormanMM (Forman's Mixture Method).
  
  It uses cvxpy to solve the minimum squeares convex optimization problem (L2)
  about probabilities histograms of positive class.
  
  This method assumes that the predicted distribution on the test set is a linear mixture
  of distributions seen on calibration for positive class (D+) and negative (D-),
  weighted by their respective prevalences.
  
  Refs
  [1] Forman, G. Quantifying counts and costs via classification.
      Data Min Knowl Disc 17, 164-206 (2008).
      https://doi.org/10.1007/s10618-008-0097-y
  """
  _strictly_binary = True

  def __init__(self, classifier: BaseEstimator = None, cv: int = 5, n_bins: int = 10):
    super().__init__(classifier=classifier, cv=cv)
    self.n_bins = n_bins

  def _get_oof_method(self) -> str:
    return "predict_proba"

  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    """ 1-step: build the baseline distributions
    
    D+: distribution scores from validation for positive class
    D-: distribution scores

    Args:
      y_true_oof (np.ndarray): _description_
      y_pred_oof (np.ndarray): _description_
    """
    neg_class = self.classes_[0]
    pos_class = self.classes_[1]

    # split the masks based on the ground-truth (y_train)
    actual_pos_mask = (y_true_oof == pos_class)
    actual_neg_mask = (y_true_oof == neg_class)

    # define the thresholds for each histograms bins (B)
    self.bins_ = np.linspace(0.0, 1.0, self.n_bins + 1)

    # get scores for P(Y=1|X) generated via Out-of-Fold Cross Validation
    pos_probs_oof = y_pred_oof[:, 1]

    # Create frequency histograms for each individual class
    h_pos, _ = np.histogram(pos_probs_oof[actual_pos_mask], bins=self.bins_)
    h_neg, _ = np.histogram(pos_probs_oof[actual_neg_mask], bins=self.bins_)

    # We add 1e-12 in each value to avoid division per 0 in empty bins
    # self.h_pos_ represents the vector D+ (probability score that Y=1)
    self.h_pos_ = h_pos / (np.sum(h_pos) + 1e-12)
    # self.h_neg_ represents the vector D- (probability score that Y=0)
    self.h_neg_ = h_neg / (np.sum(h_neg) + 1e-12)

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    if not hasattr(self.classifier_, "predict_proba"):
      raise AttributeError(
        f"The base classifier '{self.classifier_.__class__.__name__}' "
        f"do not have the method 'predict_proba' necessary to FormanMM."
      )

    # step-2: capture the test mix distribution (D_test)
    # get the probabilities of an unlabeled test bag
    pos_probs_test = self.classifier_.predict_proba(X)[:, 1]

    # build the test histogram using the same bins partitions of the training phase
    h_test, _ = np.histogram(pos_probs_test, bins=self.bins_)
    # h_test represents the agg distribution observed in the test set
    h_test = h_test / (np.sum(h_test) + 1e-12)
    
    # mixture matrix (M) stores the calibration columns (D-, D+)
    # (n_bins, 2)
    calib_matrix = np.column_stack((self.h_neg_, self.h_pos_))
    
    # decision vector (alpha) represents the variables for prevalence estimation
    # alpha[0] is the negative class prevalence (1 - p)
    # alpha[1] is the positive class prevalence (p)
    alpha = cp.Variable(2)
  
    # objetive function (minimum squares)
    # @ operator executes the product calib_matrix @ alpha
    # alpha[0] * D- + alpha[1] * D+
    # objetive is to minimize the sum of squares errors between observed distributions
    # and the linear combination: ||D_test - [(1 - p) * D- + p * D+]||^2
    objective = cp.Minimize(cp.sum_squares(h_test - calib_matrix @ alpha))

    # simplex probabilities
    # cp.sum(alpha) == 1. this ensure that (1 - p) + p = 1.0
    # alpha >= 0 ensure the non-negativity restriction
    constraints = [cp.sum(alpha) == 1, alpha >= 0]
    problem = cp.Problem(objective, constraints) # finds the values for p that satisfies our restrictions
    problem.solve()

    # extract the calculated prevalences
    p_adjusted = alpha.value

    if p_adjusted is None:
      # fallback if the solver didnt find any solutions
      return self.train_prevalence_.copy()

    # eliminate any noise or float point errors
    p_adjusted = np.clip(p_adjusted, 0.0, 1.0)
    p_adjusted /= np.sum(p_adjusted)

    return p_adjusted


class GAC(BaseCalibratedQuantifier):
  """
  GAC (Generalized Adjusted Classify and Count).

  GAC extends the multiclass problem for ACC (Adjusted Classify and Count).
  Works with hard predictions (crisp labels via predict call) to build the
  confusion matrix CxC normalized.

  Refs
  [1] Aykut Firat. Unified framework for quantification.
      arXiv preprint arXiv:1606.00868, 2016.
  """
  _strictly_binary = False

  def __init__(self, classifier: BaseEstimator = None, cv: int = 5):
    super().__init__(classifier=classifier, cv=cv)

  def _get_oof_method(self) -> str:
    return "predict"

  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    """
    Build the Confusion Matrix (M).
    Each column j represents the distribution of predictions given that the real class
    is c_j. The resulting matrix have the shape (n_classes, n_classes).
    """
    n_classes = self.n_classes_
    self.cm_ = np.zeros((n_classes, n_classes))

    for j, true_class in enumerate(self.classes_):
      # we isolate the instances where the validation ground truth is the class 'true class'
      mask = (y_true_oof == true_class)
      total_true = np.sum(mask)
      
      if total_true > 0:
        for i, pred_class in enumerate(self.classes_):
          # M[i, j] = P(Y_pred = c_i | Y_true = c_j)
          self.cm_[i, j] = np.sum(y_pred_oof[mask] == pred_class) / total_true
      else:
        # guardrail for cases where a rare class isnt found in the fold
        self.cm_[:, j] = 1.0 / n_classes

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    if not hasattr(self.classifier_, "predict"):
      raise AttributeError(
        f"The base classifier '{self.classifier_.__class__.__name__}' "
        f"do not have the method 'predict' required for GAC."
      )

    
    y_pred_test = self.classifier_.predict(X) # direct count for test bag (using CC)
    # prevalence vector
    p_cc = np.zeros(self.n_classes_)
    total_test = len(y_pred_test)
    for i, pred_class in enumerate(self.classes_):
      p_cc[i] = np.sum(y_pred_test == pred_class) / (total_test + 1e-12)

    # alpha represent the vector of real prevalences adjusted that
    # we want to estimate
    alpha = cp.Variable(self.n_classes_)

    # GAC solve by minimum squares (L2) the linear sistem M @ alpha = p_cc
    objective = cp.Minimize(cp.sum_squares(p_cc - self.cm_ @ alpha))

    # restrictions are 1-the sum must be 1.0, and 2-none can be negative
    constraints = [cp.sum(alpha) == 1, alpha >= 0]

    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    p_adjusted = alpha.value
    if p_adjusted is None:
      # fallback if the solver didnt solve the problem
      return self.train_prevalence_.copy()

    # handle noise in p_adjusted
    p_adjusted = np.clip(p_adjusted, 0.0, 1.0)
    p_adjusted /= np.sum(p_adjusted)

    return p_adjusted


class GPAC(BaseCalibratedQuantifier):
  """
  GPAC (Generalized Probabilistic Adjusted Classify and Count).
  
  GPAC is the multiclass extension for PACC (Probabilistic Adjusted Classify and Count)
  Its replace the confusion matrix built with hard labels with the mean of predicted 
  probabilities, avoiding the threshold impact that hard labels have.
  
  Refs
  [1] Aykut Firat. Unified framework for quantification.
      arXiv preprint arXiv:1606.00868, 2016.
  """
  _strictly_binary = False

  def __init__(self, classifier: BaseEstimator = None, cv: int = 5):
      super().__init__(classifier=classifier, cv=cv)

  def _get_oof_method(self) -> str:
    return "predict_proba"

  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    """
    Build the probabilities confusion matrix (M)
    Each element M[i, j] stores the mean of predicted probabilities for class i given
    that the real ground truth is the class j.
    Shape of M is (n_classes, n_classes)
    """
    n_classes = self.n_classes_
    self.cm_ = np.zeros((n_classes, n_classes))

    # loop for each real class j (columns for the matrix)
    for j, true_class in enumerate(self.classes_):
      # isolate the scores for each instance that real belong to that class (true class)
      mask = (y_true_oof == true_class)
      total_true = np.sum(mask)
      
      if total_true > 0:
        # loop each prob column i (rows for M)
        for i in range(n_classes):
          # M[i, j] = mean( P(Y_pred = c_i | Y_true = c_j) )
          self.cm_[i, j] = np.mean(y_pred_oof[mask, i])
      else:
        # guardrail: uniform att in case that the class do not appear in this fold
        self.cm_[:, j] = 1.0 / n_classes

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    if not hasattr(self.classifier_, "predict_proba"):
      raise AttributeError(
        f"The base classifier '{self.classifier_.__class__.__name__}' "
        f"do not have the method 'predict_proba' required for GPAC."
      )

    # step-2 calculate the prevalence probability in this test bag using PCC
    y_pred_test = self.classifier_.predict_proba(X)
    
    # Array containing the mean probabilities predicted for each class in the test bag
    # p_pcc = [ mean(P(c_0)), mean(P(c_1)), ..., mean(P(c_C)) ]
    p_pcc = np.mean(y_pred_test, axis=0)

    # alpha is the decision array containing the real prevalences adjusted (C,)
    alpha = cp.Variable(self.n_classes_)

    # GPAC solves using minimum squares (L2) the linear system M @ alpha = p_pcc
    objective = cp.Minimize(cp.sum_squares(p_pcc - self.cm_ @ alpha))

    # restrictions: sum of prevalences must be 1.0 and only have positive values
    constraints = [cp.sum(alpha) == 1, alpha >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    p_adjusted = alpha.value

    if p_adjusted is None:
      return self.train_prevalence_.copy()

    p_adjusted = np.clip(p_adjusted, 0.0, 1.0)
    p_adjusted /= np.sum(p_adjusted)

    return p_adjusted


class HDy(BaseCalibratedQuantifier):
  """
  HDy (Hellinger Distance on y-scores).
  
  minimizes the hellinger distance between probabilities histograms on tests bags
  and the linear combinations of validations.
  """
  _strictly_binary = False

  def __init__(self, classifier=None, cv=5, n_bins=10):
    super().__init__(classifier=classifier, cv=cv)
    self.n_bins = n_bins

  def _get_oof_method(self) -> str:
    return "predict_proba"

  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    """
    Build the calibration matrix of scores (M).
    For each real class, we create a histogram of the predicted probabilities.
    In a multi-class scenario, we concatenate the histograms of all columns of probabilities.
    """
    # fixed bins in the space of probs[0, 1]
    self.bin_edges_ = np.linspace(0.0, 1.0, self.n_bins + 1)
    
    calib_cols = []
    # loop over each real class j (columns of M)
    for j, true_class in enumerate(self.classes_):
      mask = (y_true_oof == true_class)
      class_components = []
      
      if np.sum(mask) > 0:
        # loop over each column of pred probs c
        for c in range(self.n_classes_):
          scores_c = y_pred_oof[mask, c]
          h, _ = np.histogram(scores_c, bins=self.bin_edges_)
          # local histogram normalization for each column
          h = h / (np.sum(h) + 1e-12)
          class_components.append(h)
      else:
        # guardrail uniform if a class do not appear in the validation fold
        for _ in range(self.n_classes_):
          class_components.append(np.ones(self.n_bins) / self.n_bins)

      # concatenate the histograms of all probabilities of class j
      # result (n_classes * n_bins, )
      super_vector_j = np.concatenate(class_components)
      calib_cols.append(super_vector_j)

    # Matrix M (n_classes * n_bins, n_classes)
    self.calib_matrix_ = np.column_stack(calib_cols)

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    if not hasattr(self.classifier_, "predict_proba"):
      raise AttributeError(
        f"The base classifier '{self.classifier_.__class__.__name__}' "
        f"do not have the method 'predict_proba' required for HDy."
      )

    y_pred_test = self.classifier_.predict_proba(X)
    
    test_components = []
    for c in range(self.n_classes_):
      h, _ = np.histogram(y_pred_test[:, c], bins=self.bin_edges_)
      h = h / (np.sum(h) + 1e-12)
      test_components.append(h)
        
    # continuous vector for test (n_classes * n_bins,)
    h_test = np.concatenate(test_components)


    alpha = cp.Variable(self.n_classes_)

    # Transformação matemática crucial para conformidade DCP:
    # minimize Hellinger is equivalent to maximize the afinity (Bhattacharyya)
    # sum(sqrt(h_test) * sqrt(M @ alpha))
    predicted_mixture = self.calib_matrix_ @ alpha
    
    objective = cp.Maximize(
      cp.sum(cp.multiply(np.sqrt(h_test), cp.sqrt(predicted_mixture)))
    )
    constraints = [cp.sum(alpha) == 1, alpha >= 0]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    p_adjusted = alpha.value

    if p_adjusted is None:
      return self.train_prevalence_.copy()

    p_adjusted = np.clip(p_adjusted, 0.0, 1.0)
    p_adjusted /= np.sum(p_adjusted)

    return p_adjusted

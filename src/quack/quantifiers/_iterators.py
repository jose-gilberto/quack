import numpy as np
from sklearn.utils.validation import check_array
from sklearn.base import BaseEstimator
from quack.quantifiers.base import BaseCalibratedQuantifier


class EM(BaseCalibratedQuantifier):
    """
    EM (Expectation Maximization Quantifier / Saerens-Latinne-Decaestecker - SLD).
    
    EM adjusts the posterior probabilities of test bags using the rate between
    predicted prevalences in the test and the prevalences in the training phase,
    solving it dinamically in scenarios with Prior Probability Shift.
    
    Refs
    """
    _strictly_binary = False

    def __init__(self,
                 classifier: BaseEstimator = None,
                 cv: int = 5, max_iter: int = 1000,
                 epsilon: float = 1e-6):
      super().__init__(classifier=classifier, cv=cv)
      self.max_iter = max_iter
      self.epsilon = epsilon

    def _get_oof_method(self) -> str:
      return "predict_proba"

    def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
      """
      EM do not require the calibration step on validation data.
      """
      pass

    def _quantify(self, X: np.ndarray) -> np.ndarray:
      if not hasattr(self.classifier_, "predict_proba"):
        raise AttributeError(
          f"The base classifier '{self.classifier_.__class__.__name__}' "
          f"do not have the method 'predict_proba' required for EM."
        )

      # extract pred_probs from test bag
      # p_clf(c | x)
      y_pred_test = self.classifier_.predict_proba(X)

      # guardrail to avoid problems with logs or division per 0
      y_pred_test = np.maximum(y_pred_test, 1e-12)

      # capture the probs from training
      p_train = self.train_prevalence_.copy()
      p_train = np.maximum(p_train, 1e-12)

      # initial step = same probs from training
      alpha = p_train.copy()

      for iteration in range(self.max_iter):
        alpha_old = alpha.copy()

        # --- step E (expectation) ---
        # update posterior probs for each instance individually
        # applies the bayes theorem, multiplying the raw score by
        # the rate (actual_alpha / p_train)
        weight_factor = alpha / p_train
        adjusted_posteriors = y_pred_test * weight_factor

        # normalization per row
        row_sums = np.sum(adjusted_posteriors, axis=1, keepdims=True)
        adjusted_posteriors /= (row_sums + 1e-12)

        # --- step M (maximization) ---
        # update the global estimative calculating the mean probs adjusted
        # for all test bag
        alpha = np.mean(adjusted_posteriors, axis=0)

        # --- stop criteria ---
        # evaluates stabilization using the uniform norm (maximum absolute variation across classes)
        if np.max(np.abs(alpha - alpha_old)) < self.epsilon:
          break

      alpha = np.clip(alpha, 0.0, 1.0)
      alpha /= np.sum(alpha)

      return alpha
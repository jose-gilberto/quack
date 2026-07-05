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

    # =====================================================================
    # CONVEX OPTIMIZATION (CVXPY)
    # =====================================================================
    
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
  GAC (Generalized Area Correction).
  
  Implementação alinhada ao framework QFY e formulação original de Bella et al.
  Diferente do FormanMM, o GAC suporta problemas multiclasse concatenando os 
  histogramas de todas as colunas de probabilidade e resolve a otimização 
  utilizando a norma L1 (distância de Manhattan).
  """
  _strictly_binary = False  # GAC suporta nativamente 2, 3 ou mais classes.

  def __init__(self, classifier=None, cv=5, n_bins=10):
    super().__init__(classifier=classifier, cv=cv)
    self.n_bins = n_bins

  def _get_oof_method(self) -> str:
    return "predict_proba"

  def _calibrate(self, y_true_oof: np.ndarray, y_pred_oof: np.ndarray):
    """
    PASSO 1 DO QFY (MULTICLASSE): Construção da Matriz de Perfis de Distribuição.
    Para cada classe real, geramos um vetor característico concatenando os 
    histogramas de probabilidade de cada uma das colunas do classificador.
    """
    self.bins_ = np.linspace(0.0, 1.0, self.n_bins + 1)
    calib_cols = []

    # Varre cada classe presente no conjunto de treinamento
    for c_idx, c in enumerate(self.classes_):
      # Filtra as predições fora do fold pertencentes estritamente à classe real 'c'
      preds_c = y_pred_oof[y_true_oof == c]
      
      class_hist_components = []
      # EQUIVALENTE AO QFY MULTICLASSE: Varre todas as colunas de probabilidade preditas
      for col in range(self.n_classes_):
        h, _ = np.histogram(preds_c[:, col], bins=self.bins_)
        # Normalização local do bin para representar a densidade de probabilidade da coluna
        h = h / (np.sum(h) + 1e-12)
        class_hist_components.append(h)
      
      # Concatena os histogramas de todas as colunas em um único vetor longo de tamanho (n_classes * n_bins)
      super_histogram_c = np.concatenate(class_hist_components)
      calib_cols.append(super_histogram_c)

    # EQUIVALENTE AO ARTIGO: Matriz M de calibração global. 
    # Dimensões: (n_classes * n_bins, n_classes)
    self.calib_matrix_ = np.column_stack(calib_cols)

  def _quantify(self, X: np.ndarray) -> np.ndarray:
    if not hasattr(self.classifier_, "predict_proba"):
      raise AttributeError(
        f"O classificador base '{self.classifier_.__class__.__name__}' "
        f"não possui o método 'predict_proba' exigido pelo GAC."
      )

    # PASSO 2 DO QFY: Captura do perfil misto do lote de teste
    y_pred_test = self.classifier_.predict_proba(X)
    
    test_hist_components = []
    # Constrói o super-histograma do teste seguindo o exato mesmo espelhamento do treino
    for col in range(self.n_classes_):
      h, _ = np.histogram(y_pred_test[:, col], bins=self.bins_)
      h = h / (np.sum(h) + 1e-12)
      test_hist_components.append(h)
    
    # Vetor contínuo observado no teste. Dimensão: (n_classes * n_bins)
    h_test = np.concatenate(test_hist_components)

    # =====================================================================
    # ENGINE DE CONVEX OPTIMIZATION (CVXPY) - FORMULAÇÃO NORMA L1 DO GAC
    # =====================================================================
    
    # 1. Vetor de Decisão (alpha): Contém as prevalências latentes para cada classe
    # Dimensão correspondente ao número total de classes do problema (C)
    alpha = cp.Variable(self.n_classes_)

    # 2. Função Objetivo Estrita do GAC: Minimizar a Norma L1 (Erro Absoluto)
    # No CVXPY, cp.norm(..., 1) calcula: Sum( | h_test - (M @ alpha) | )
    # Essa formulação matemática caracteriza o método "Generalized Area Correction"
    objective = cp.Minimize(cp.norm(h_test - self.calib_matrix_ @ alpha, 1))

    # 3. Restrições do Simplex de Probabilidade Multiclasse
    # Garante que a soma de todas as prevalências estimadas seja exatamente 1.0 
    # e nenhuma classe receba atribuição de prevalência negativa.
    constraints = [cp.sum(alpha) == 1, alpha >= 0]

    # 4. Resolução Convexa via Solver
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Extração do vetor de prevalências ótimas ajustadas
    p_adjusted = alpha.value

    # --- Salvaguardas e Estabilização Numérica ---
    if p_adjusted is None:
        # Fallback seguro para as proporções originais do treino caso o solver falhe
        return self.train_prevalence_.copy()

    # Sanatização de resíduos numéricos infinitesimais de ponto flutuante
    p_adjusted = np.clip(p_adjusted, 0.0, 1.0)
    p_adjusted /= np.sum(p_adjusted)

    return p_adjusted

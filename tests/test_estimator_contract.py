"""Regression tests for bugs that only surface when a quantifier's public
contract (scikit-learn's fitted-state protocol, the optimality of the convex
solver) is checked explicitly rather than through "does it sum to 1.0".
"""
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from quack.quantifiers import CC, PCC, ACC, PACC, DyS, HDy, FormanMM, HDx, ReadMe, ED, GAC
from quack.quantifiers._features import _RawSubspaceMixture
from quack.quantifiers.base import BaseMixtureQuantifier


class _TopsoeProbe(BaseMixtureQuantifier):
  """Minimal mixture quantifier used to compare the convex solver against a
  brute-force search on the very distance it claims to minimize."""

  def __init__(self, distance_metric="TS", use_convex_solver=True):
    super().__init__(classifier=None, distance_metric=distance_metric,
                     use_convex_solver=use_convex_solver)

  def fit(self, X, y):  # pragma: no cover - not used by these tests
    return self

  def _compute_score(self, X):  # pragma: no cover - not used by these tests
    return np.zeros(self.conditional_matrix_.shape[0])


class TestUnfittedEstimatorsRaise:
  """`check_is_fitted` considers *any* trailing-underscore attribute as proof
  of a fitted estimator, so initializing them in `__init__` silently breaks
  the NotFittedError contract and turns clear errors into obscure ones."""

  @pytest.mark.parametrize("quantifier", [
    CC(), PCC(), ACC(), PACC(), DyS(), HDy(), FormanMM(), HDx(), ReadMe(), ED(), GAC(),
    _RawSubspaceMixture(),
  ])
  def test_check_is_fitted_rejects_unfitted_quantifier(self, quantifier):
    with pytest.raises(NotFittedError):
      check_is_fitted(quantifier)

  @pytest.mark.parametrize("quantifier", [HDx(), ED(), ReadMe(), DyS(), CC()])
  def test_predict_before_fit_raises_not_fitted(self, quantifier):
    with pytest.raises(NotFittedError):
      quantifier.predict(np.zeros((4, 3)))


class TestReadMeDoesNotMutateHyperparameters:
  def test_fit_resolves_n_features_into_fitted_attribute(self):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 4, size=(120, 6)).astype(float)
    y = (X[:, 0] > 1).astype(int)

    quantifier = ReadMe(n_subsets=3, random_state=0)
    assert quantifier.n_features is None

    quantifier.fit(X, y)

    # the hyper-parameter must survive `fit` untouched, so that `clone`
    # reproduces the estimator the user actually configured
    assert quantifier.n_features is None
    assert quantifier.n_features_ == 2
    assert clone(quantifier).get_params()["n_features"] is None

  def test_explicit_n_features_is_echoed_into_fitted_attribute(self):
    rng = np.random.default_rng(1)
    X = rng.integers(0, 4, size=(120, 6)).astype(float)
    y = (X[:, 0] > 1).astype(int)

    quantifier = ReadMe(n_subsets=3, n_features=3, random_state=0).fit(X, y)
    assert quantifier.n_features == 3
    assert quantifier.n_features_ == 3


class TestConvexSolverMinimizesTheAdvertisedDistance:
  """The convex program must optimize the *same* objective as
  `_compute_distance`; a formulation mismatch still returns a feasible,
  normalized vector, so only an optimality check can catch it."""

  @staticmethod
  def _brute_force_binary(quantifier, test_frequencies, n_grid=20001):
    grid = np.linspace(0.0, 1.0, n_grid)
    losses = [quantifier._compute_distance(np.array([p, 1.0 - p]), test_frequencies) for p in grid]
    return float(np.min(losses))

  @pytest.mark.parametrize("distance_metric", ["L1", "L2", "HD", "TS"])
  def test_convex_solution_is_not_worse_than_brute_force(self, distance_metric):
    rng = np.random.default_rng(0)
    quantifier = _TopsoeProbe(distance_metric=distance_metric)
    quantifier.conditional_matrix_ = rng.dirichlet(np.ones(6), size=2).T
    test_frequencies = rng.dirichlet(np.ones(6))

    solution = quantifier._solve_mixture(test_frequencies)
    convex_loss = quantifier._compute_distance(solution, test_frequencies)
    best_loss = self._brute_force_binary(quantifier, test_frequencies)

    assert convex_loss == pytest.approx(best_loss, abs=1e-6)

  @pytest.mark.parametrize("distance_metric", ["L1", "L2", "HD", "TS"])
  def test_convex_solver_agrees_with_golden_section_fallback(self, distance_metric):
    rng = np.random.default_rng(7)
    conditional_matrix = rng.dirichlet(np.ones(8), size=2).T
    test_frequencies = rng.dirichlet(np.ones(8))

    convex = _TopsoeProbe(distance_metric=distance_metric, use_convex_solver=True)
    gss = _TopsoeProbe(distance_metric=distance_metric, use_convex_solver=False)
    convex.conditional_matrix_ = conditional_matrix
    gss.conditional_matrix_ = conditional_matrix

    p_convex = convex._solve_mixture(test_frequencies)
    p_gss = gss._solve_mixture(test_frequencies)

    np.testing.assert_allclose(p_convex, p_gss, atol=1e-3)

  def test_topsoe_objective_matches_closed_form_on_a_known_case(self):
    # a conditional matrix with a unique, easily verifiable optimum: the
    # test frequencies are exactly reproduced by p = [0.25, 0.75]
    quantifier = _TopsoeProbe(distance_metric="TS")
    quantifier.conditional_matrix_ = np.array([[1.0, 0.0], [0.0, 1.0]])
    test_frequencies = np.array([0.25, 0.75])

    solution = quantifier._solve_mixture(test_frequencies)

    np.testing.assert_allclose(solution, [0.25, 0.75], atol=1e-4)
    assert quantifier._compute_distance(solution, test_frequencies) == pytest.approx(0.0, abs=1e-8)

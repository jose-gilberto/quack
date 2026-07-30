# tests/test_metrics.py
import numpy as np
import pytest

from quack.metrics import (
  ae, rae, nae, kld, nkld,
  AbsoluteError, RelativeAbsoluteError, NormalizedAbsoluteError,
  KullbackLeiblerDivergence, NormalizedKullbackLeiblerDivergence,
  MetricRegistry,
)


class TestAbsoluteError:
  def test_perfect_prediction_is_zero(self):
    p = np.array([0.3, 0.7])
    assert ae(p, p) == pytest.approx(0.0)

  def test_is_averaged_over_classes_binary_bounded_by_one(self):
    p_true = np.array([1.0, 0.0])
    p_pred = np.array([0.0, 1.0])
    # worst possible binary case must be bounded by 1.0, not 2.0
    assert ae(p_true, p_pred) == pytest.approx(1.0)

  def test_matches_manual_mean_formula(self):
    p_true = np.array([0.2, 0.5, 0.3])
    p_pred = np.array([0.3, 0.4, 0.3])
    expected = np.mean(np.abs(p_true - p_pred))
    assert ae(p_true, p_pred) == pytest.approx(expected)


class TestRelativeAbsoluteError:
  def test_perfect_prediction_is_near_zero(self):
    p = np.array([0.4, 0.6])
    assert rae(p, p) == pytest.approx(0.0, abs=1e-6)

  def test_handles_zero_true_prevalence_without_error(self):
    p_true = np.array([1.0, 0.0])
    p_pred = np.array([0.9, 0.1])
    result = rae(p_true, p_pred)
    assert np.isfinite(result)
    assert result > 0

  def test_is_averaged_not_summed(self):
    metric = RelativeAbsoluteError(epsilon=1e-5)
    p_true = np.array([0.25, 0.25, 0.25, 0.25])
    p_pred = np.array([0.3, 0.2, 0.3, 0.2])
    result = metric(p_true, p_pred)
    assert result < 1.0  # would trivially exceed 1 for many classes if summed


class TestNormalizedAbsoluteError:
  def test_perfect_prediction_is_zero(self):
    p = np.array([0.3, 0.7])
    assert nae(p, p) == pytest.approx(0.0)

  def test_bounded_between_zero_and_one(self):
    rng = np.random.default_rng(0)
    for _ in range(20):
      p_true = rng.dirichlet(np.ones(4))
      p_pred = rng.dirichlet(np.ones(4))
      result = nae(p_true, p_pred)
      assert 0.0 <= result <= 1.0 + 1e-9

  def test_worst_case_binary_equals_one(self):
    p_true = np.array([0.5, 0.5])
    p_pred = np.array([0.0, 1.0])  # maximal deviation given p_true
    assert nae(p_true, p_pred) == pytest.approx(1.0)


class TestKullbackLeiblerDivergence:
  def test_perfect_prediction_is_near_zero(self):
    p = np.array([0.4, 0.6])
    assert kld(p, p) == pytest.approx(0.0, abs=1e-6)

  def test_is_non_negative(self):
    rng = np.random.default_rng(1)
    for _ in range(20):
      p_true = rng.dirichlet(np.ones(3))
      p_pred = rng.dirichlet(np.ones(3))
      assert kld(p_true, p_pred) >= -1e-9

  def test_smoothing_keeps_distribution_normalized(self):
    metric = KullbackLeiblerDivergence(epsilon=1e-5)
    p = np.array([1.0, 0.0, 0.0])
    smoothed = metric._smooth(p, metric.epsilon)
    assert smoothed.sum() == pytest.approx(1.0)


class TestNormalizedKullbackLeiblerDivergence:
  def test_does_not_raise_type_error(self):
    # regression test for the previous `self.kld(p_true, p_pred, eps=...)` bug
    p_true = np.array([0.3, 0.7])
    p_pred = np.array([0.4, 0.6])
    result = nkld(p_true, p_pred)
    assert np.isfinite(result)

  def test_perfect_prediction_is_near_zero(self):
    p = np.array([0.4, 0.6])
    assert nkld(p, p) == pytest.approx(0.0, abs=1e-6)

  def test_bounded_between_zero_and_one(self):
    rng = np.random.default_rng(2)
    for _ in range(20):
      p_true = rng.dirichlet(np.ones(3))
      p_pred = rng.dirichlet(np.ones(3))
      result = nkld(p_true, p_pred)
      assert 0.0 <= result < 1.0 + 1e-9


class TestInputValidation:
  def test_raises_on_shape_mismatch(self):
    with pytest.raises(ValueError, match="Shape Error"):
      ae(np.array([0.5, 0.5]), np.array([0.3, 0.3, 0.4]))

  def test_raises_on_non_1d_input(self):
    with pytest.raises(ValueError, match="1D"):
      ae(np.array([[0.5, 0.5]]), np.array([[0.3, 0.7]]))

  def test_warns_when_prevalence_does_not_sum_to_one(self):
    with pytest.warns(UserWarning, match="does not sum to 1.0"):
      ae(np.array([0.3, 0.3]), np.array([0.5, 0.5]))


class TestMetricRegistry:
  def test_get_known_metric_returns_instance(self):
    metric = MetricRegistry.get("kld", epsilon=1e-3)
    assert isinstance(metric, KullbackLeiblerDivergence)
    assert metric.epsilon == 1e-3

  def test_mae_alias_returns_absolute_error(self):
    assert isinstance(MetricRegistry.get("mae"), AbsoluteError)

  def test_nae_is_registered(self):
    assert isinstance(MetricRegistry.get("nae"), NormalizedAbsoluteError)

  def test_raises_for_unknown_metric(self):
    with pytest.raises(KeyError, match="not supported"):
      MetricRegistry.get("not-a-real-metric")

  def test_available_metrics_lists_all_keys(self):
    keys = MetricRegistry.available_metrics()
    assert set(keys) == {"ae", "mae", "rae", "nae", "kld", "nkld"}

import matplotlib
matplotlib.use("Agg")  # no display needed in CI

import numpy as np
import matplotlib.pyplot as plt
import pytest

from quack.visualization import (
  prevalence_plot,
  bias_plot,
  class_distribution_plot,
  get_color_palette,
  get_marker_cycle,
  COLORBLIND_PALETTE,
)


@pytest.fixture
def binary_experiment():
  rng = np.random.default_rng(0)
  true_prev = rng.uniform(0, 1, 300)
  estim_prev = np.clip(true_prev + rng.normal(0, 0.05, 300), 0, 1)
  return true_prev, estim_prev


class TestColorUtils:
  def test_get_color_palette_within_base_size(self):
    colors = get_color_palette(3)
    assert colors == COLORBLIND_PALETTE[:3]

  def test_get_color_palette_extends_for_many_methods(self):
    n = len(COLORBLIND_PALETTE) + 10
    colors = get_color_palette(n)
    assert len(colors) == n
    assert len(set(map(str, colors))) == n  # all distinct

  def test_get_color_palette_respects_custom_palette(self):
    custom = ["#111111", "#222222"]
    assert get_color_palette(2, palette=custom) == custom

  def test_get_marker_cycle_repeats_when_exceeding_base(self):
    markers = get_marker_cycle(20)
    assert len(markers) == 20


class TestPrevalencePlot:
  def test_returns_figure_single_method(self, binary_experiment):
    true_prev, estim_prev = binary_experiment
    fig = prevalence_plot("CC", true_prev, estim_prev, train_prevalence=0.5)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

  def test_returns_figure_multiple_methods(self, binary_experiment):
    true_prev, estim_prev = binary_experiment
    fig = prevalence_plot(
      ["CC", "ACC"], [true_prev, true_prev], [estim_prev, estim_prev - 0.05],
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

  def test_merges_repeated_method_names(self, binary_experiment):
    true_prev, estim_prev = binary_experiment
    half = len(true_prev) // 2
    fig = prevalence_plot(
      ["CC", "CC"],
      [true_prev[:half], true_prev[half:]],
      [estim_prev[:half], estim_prev[half:]],
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

  def test_raises_on_mismatched_lengths(self, binary_experiment):
    true_prev, estim_prev = binary_experiment
    with pytest.raises(ValueError):
      prevalence_plot(["CC", "ACC"], [true_prev], [estim_prev])

  def test_accepts_external_axes(self, binary_experiment):
    true_prev, estim_prev = binary_experiment
    fig, ax = plt.subplots()
    returned_fig = prevalence_plot("CC", true_prev, estim_prev, ax=ax)
    assert returned_fig is fig
    plt.close(fig)


class TestQuantificationBiasPlot:
  def test_returns_figure_global(self, binary_experiment):
    true_prev, estim_prev = binary_experiment
    fig = bias_plot("CC", true_prev, estim_prev, n_bins=1)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

  def test_returns_figure_binned(self, binary_experiment):
    true_prev, estim_prev = binary_experiment
    fig = bias_plot(["CC", "ACC"], [true_prev, true_prev],
                                   [estim_prev, estim_prev - 0.05], n_bins=3)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

  def test_raises_on_mismatched_lengths(self, binary_experiment):
    true_prev, estim_prev = binary_experiment
    with pytest.raises(ValueError):
      bias_plot(["CC", "ACC"], [true_prev], [estim_prev])


class TestClassDistributionPlot:
  def test_returns_figure_counts(self):
    y = np.array([0, 0, 1, 1, 1, 2])
    fig = class_distribution_plot(y)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

  def test_returns_figure_normalized(self):
    y = np.array([0, 0, 1, 1, 1, 2])
    fig = class_distribution_plot(y, normalize=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

  def test_horizontal_orientation(self):
    y = np.array([0, 1, 1, 2, 2, 2])
    fig = class_distribution_plot(y, horizontal=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

  def test_many_classes_get_distinct_colors(self):
    y = np.arange(15).repeat(3)
    fig = class_distribution_plot(y)
    ax = fig.axes[0]
    facecolors = {tuple(p.get_facecolor()) for p in ax.patches}
    assert len(facecolors) == 15
    plt.close(fig)
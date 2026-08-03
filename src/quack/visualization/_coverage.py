# src/quack/visualization/_coverage.py
"""Prevalence coverage plot.

Visualizes the range of class prevalences actually represented across a
set of generated bags (e.g. from `quack.bag_generator`), for a single
target class. Combines a histogram of bag counts per prevalence bin with
a rug plot marking each individual bag, making gaps in the coverage of
the `[0, 1]` prevalence range immediately visible — useful to sanity-check
that a `BaseBagGenerator` (or a custom sampling strategy) is actually
producing the intended spread of test-time conditions before running a
full quantification benchmark.
"""
from typing import Sequence
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from quack.visualization._colors import get_color_palette, REFERENCE_COLOR


def _normalize_experiments(value) -> list:
  if isinstance(value, np.ndarray):
    return [value]
  return list(value)


def prevalence_coverage_plot(
  prevalences: np.ndarray | Sequence[np.ndarray],
  labels: str | Sequence[str] = None,
  class_name: str = "positive class",
  train_prevalence: float | Sequence[float] = None,
  n_bins: int = 20,
  show_rug: bool = True,
  show_stats: bool = True,
  density: bool = False,
  colors: Sequence = None,
  fig_size: tuple[float, float] = (8, 5),
  font_size: int = 11,
  legend_font_size: int = None,
  bar_alpha: float = 0.65,
  rug_height: float = 0.04,
  title: str = "Prevalence Coverage",
  grid: bool = True,
  ax: matplotlib.axes.Axes = None,
) -> matplotlib.figure.Figure:
  """Plot how well a set of bags covers the `[0, 1]` prevalence range for a class.

  Each entry in `prevalences` is a 1D array with the realized prevalence
  of `class_name` for every generated bag in one experiment (typically
  `generator.sampled_prevalences_[:, class_index]` from a
  `quack.bag_generator.BaseBagGenerator` subclass). A histogram over
  `n_bins` equal-width bins shows how many bags fall in each prevalence
  range, while an optional rug plot marks every individual bag along the
  x-axis, so isolated or empty regions of the simplex are easy to spot
  even when the histogram bin is technically non-empty.

  Multiple series can be overlaid (e.g. to compare `PriorShiftBagGenerator`
  vs. `CovariateShiftBagGenerator`, or different `sampling_strategy`/
  `dirichlet_alpha` configurations) using semi-transparent, colorblind-safe
  colors.

  Parameters
  ----------
  prevalences: np.ndarray | Sequence[np.ndarray]
    One 1D array of per-bag prevalences for `class_name` per experiment/series. A
    single array plots one series.
  labels: str | Sequence[str], default = None
    Name of each series, used in the legend and coverage statistics. Defaults to
    `"Bags"` for a single series, or `"Series 1"`, `"Series 2"`, ... for multiple.
  class_name: str, default = "positive class"
    Label used on the x-axis for the target class. Defaults to "positive class".
  train_prevalence: float | Sequence[float], default = None
    One or more training prevalences to mark as vertical reference lines.
    Defaults to None.
  n_bins: int, default = 20
    Number of equal-width histogram bins over `[0, 1]`. Defaults to 20.
  show_rug: bool, default = True
    Whether to draw a rug plot (one tick per bag) below the histogram.
    Defaults to True.
  show_stats: bool, default = True
    Whether to annotate the plot with per-series min/max/mean prevalence
    and simplex-bin coverage (the fraction of the `n_bins` bins that contain
    at least one bag). Defaults to True.
  density: bool, default = False
    If True, normalize histograms to a density (area sums to 1) instead of raw
    bag counts — useful when comparing series generated with a different number
    of bags. Defaults to False.
  colors: Sequence, default = None 
    Custom colors, one per series. Defaults to a colorblind-safe palette,
    auto-extended as needed.
  fig_size: tuple[float, float], default = (8, 6)
    Figure size in inches. Defaults to (8, 5).
  font_size: int, default = 11
    Base font size for axis labels/title. Defaults to 11.
  legend_font_size: int, default = None
    Legend font size. Defaults to `font_size - 1` when None.
  bar_alpha: float, default = 0.65
    Opacity of the histogram bars, low enough for overlapping
    series to remain distinguishable. Defaults to 0.65.
  rug_height: float, default = 0.04
    Height of each rug row, as a fraction of the histogram's y-range.
    Defaults to 0.04.
  title: str, default = "Prevalence Coverage"
    Plot title. Defaults to "Prevalence Coverage".
  grid: bool, default = True
    Whether to draw a background grid. Defaults to True.
  ax: matplotlib.axes.Axes, default = None
    Existing axes to draw on. A new figure/axes pair is created
    when None. Defaults to None.

  Returns
  -------
  fig: matplotlib.figure.Figure
    The generated figure. Call `fig.savefig(path)` to persist it
    in any Matplotlib-supported format.

  Raises
  ------
  ValueError: If `labels` length does not match the number of series,
    any series is not 1D, or any prevalence value falls outside
    `[0, 1]`.

  Examples
  --------
  >>> import numpy as np
  >>> from quack.bag_generator import PriorShiftBagGenerator, CovariateShiftBagGenerator
  >>> from quack.visualization import prevalence_coverage_plot
  >>> from sklearn.datasets import make_classification
  >>> X, y = make_classification(n_samples=500, n_classes=2, random_state=0)
  >>> prior_gen = PriorShiftBagGenerator(n_bags=200, bag_size=100, random_state=0)
  >>> cov_gen = CovariateShiftBagGenerator(n_bags=200, bag_size=100, random_state=0)
  >>> prior_gen.to_list(X, y)
  >>> cov_gen.to_list(X, y)
  >>> fig = prevalence_coverage_plot(
  ...     [prior_gen.sampled_prevalences_[:, 1], cov_gen.sampled_prevalences_[:, 1]],
  ...     labels=["Prior Shift", "Covariate Shift"],
  ...     class_name="positive class",
  ... )
  >>> fig.savefig("prevalence_coverage.png", dpi=300)
  """
  prevalences = _normalize_experiments(prevalences)
  n_series = len(prevalences)

  if labels is None:
    labels = [f"Series {i + 1}" for i in range(n_series)] if n_series > 1 else ["Bags"]
  else:
    labels = [labels] if isinstance(labels, str) else list(labels)

  if len(labels) != n_series:
    raise ValueError(
      f"labels must have the same length as prevalences ({n_series}), got {len(labels)}."
    )

  clean_prevalences = []
  for label, p in zip(labels, prevalences):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
      raise ValueError(f"Each prevalence array must be 1D, got shape {p.shape} for series '{label}'.")
    if p.size and np.any((p < 0) | (p > 1)):
      raise ValueError(f"Prevalence values must lie within [0, 1]; series '{label}' violates this.")
    clean_prevalences.append(p)
  prevalences = clean_prevalences

  colors = get_color_palette(n_series, palette=colors)
  legend_font_size = legend_font_size if legend_font_size is not None else max(font_size - 1, 6)

  own_axes = ax is None
  if own_axes:
    fig, ax = plt.subplots(figsize=fig_size)
  else:
    fig = ax.get_figure()

  bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

  for i, p in enumerate(prevalences):
    ax.hist(p, bins=bin_edges, density=density, color=colors[i], alpha=bar_alpha,
            edgecolor=REFERENCE_COLOR, linewidth=0.5, label=labels[i], zorder=2)

  if show_rug:
    y_min, y_max = ax.get_ylim()
    rug_span = rug_height * (y_max - y_min) if y_max > y_min else rug_height
    for i, p in enumerate(prevalences):
      row_offset = -rug_span * (i + 1) * 1.3
      ax.plot(p, np.full_like(p, y_min + row_offset), '|', color=colors[i],
              markersize=8, markeredgewidth=1.2, alpha=0.85, zorder=3)
    ax.set_ylim(y_min - rug_span * 1.3 * (n_series + 1), y_max)

  if train_prevalence is not None:
    train_prevalence = (train_prevalence if isinstance(train_prevalence, (list, tuple, np.ndarray))
                        else [train_prevalence])
    for p in train_prevalence:
      ax.axvline(p, color=REFERENCE_COLOR, linestyle='--', linewidth=1.3, zorder=4,
                 label=f'Training prevalence (p={p:.2f})')

  ax.set_xlim(0, 1)
  ax.set_xlabel(f"Prevalence ({class_name})", fontsize=font_size)
  ax.set_ylabel("Density" if density else "# bags", fontsize=font_size)
  if title:
    ax.set_title(title, fontsize=font_size + 2)
  if grid:
    ax.grid(axis='y', alpha=0.3)

  if show_stats:
    stats_lines = []
    for label, p in zip(labels, prevalences):
      if p.size == 0:
        stats_lines.append(f"{label}: no bags")
        continue
      bin_idx = np.clip(np.digitize(p, bin_edges[1:-1]), 0, n_bins - 1)
      coverage = np.unique(bin_idx).size / n_bins
      stats_lines.append(
        f"{label}: min={p.min():.2f} max={p.max():.2f} mean={p.mean():.2f} coverage={coverage:.0%}"
      )
    ax.text(0.01, 0.98, "\n".join(stats_lines), transform=ax.transAxes,
            fontsize=max(font_size - 2, 6), va='top', ha='left', color=REFERENCE_COLOR,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=REFERENCE_COLOR, alpha=0.8))

  handles, hlabels = ax.get_legend_handles_labels()
  by_label = dict(zip(hlabels, handles))  # de-duplicate repeated train-prevalence labels
  ax.legend(by_label.values(), by_label.keys(), fontsize=legend_font_size,
            loc='upper center', bbox_to_anchor=(0.5, -0.15),
            ncol=min(3, len(by_label)), frameon=False)

  if own_axes:
    fig.tight_layout()

  return fig
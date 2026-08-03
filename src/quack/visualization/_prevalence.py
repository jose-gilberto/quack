# src/quack/visualization/_prevalence.py
"""Prevalence (diagonal) plot for binary quantification.

plots the predicted prevalence of a target class (y-axis) against its
true prevalence (x-axis) for one or more quantification methods, binning
repeated experiments to show the mean trend and +/- 1 std bands.
"""
from typing import Sequence
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from quack.visualization._colors import get_color_palette, get_marker_cycle, REFERENCE_COLOR


def _normalize_experiments(value) -> list:
  """Wraps a single array into a list, allowing a method name to be
  repeated across several experiments."""
  if isinstance(value, np.ndarray):
    return [value]
  return list(value)


def prevalence_plot(
  method_names: str | Sequence[str],
  true_prevalences: np.ndarray | Sequence[np.ndarray],
  estim_prevalences: np.ndarray | Sequence[np.ndarray],
  class_name: str = "positive class",
  train_prevalence: float | Sequence[float] = None,
  n_bins: int = 21,
  show_std: bool = True,
  colors: Sequence = None,
  markers: Sequence[str] = None,
  fig_size: tuple[float, float] = (6, 6),
  font_size: int = 11,
  legend_font_size: int = None,
  marker_size: float = 5.0,
  line_width: float = 1.5,
  band_alpha: float = 0.2,
  title: str = "Prevalence Plot",
  grid: bool = True,
  ax: matplotlib.axes.Axes = None,
) -> matplotlib.figure.Figure:
  """Plot predicted vs. true prevalence for a target class (binary diagonal plot).

  For each unique entry in `method_names`, all `(true, estim)` pairs across
  every experiment sharing that name are pooled, binned by true prevalence
  into `n_bins` equal-width intervals, and summarized by the mean predicted
  prevalence per bin (connected by a line) with an optional +/- 1 standard
  deviation shaded band. A dashed diagonal (`y = x`) marks the ideal,
  unbiased quantifier.

  Parameters
  ----------
  method_names: str | Sequence[str]
    Name of the method for each experiment. A name can repeat (e.g. one entry per dataset/fold); all
    matching experiments are merged before binning.
  true_prevalences: np.ndarray | Sequence[np.ndarray]
    True prevalence of `class_name` for each test bag, one 1D array per experiment
    (aligned with `method_names`).
  estim_prevalences: np.ndarray | Sequence[np.ndarray]
    Predicted prevalence of `class_name`, same shape as `true_prevalences`.
  class_name: str, default = "positive class"
    Label used on the axes/legend for the target class. Defaults to "positive class".
  train_prevalence: float | Sequence[float], default = None
    One or more training prevalences to mark on the diagonal. Defaults to None.
  n_bins: int, default = 21
    Number of equal-width bins over `[0, 1]` used to aggregate repeated experiments. Defaults to 21.
  show_st: bool, default = True
    Whether to draw +/- 1 std shaded bands around each method's line. Defaults to True.
  colors: Sequence, default = colorblind-sage palette
    Custom colors, one per unique method. Defaults to a colorblind-safe palette, auto-extended as needed.
  markers: Sequence[str], default = None
    Custom marker styles, one per unique method. Defaults to a built-in marker cycle.
  fig_size: tuple[float, float], default = (6, 6)
    Figure size in inches. Defaults to (6, 6).
  font_size: int, default = 11
    Base font size for axis labels/title. Defaults to 11.
  legend_font_size: int, default = None
    Font size for the legend. Defaults to `font_size - 1` when None.
  marker_size: float, default = 5.0
    Marker size. Defaults to 5.0.
  line_width: float, default = 1.5
    Line width. Defaults to 1.5.
  band_alpha: float, default = 0.2
    Opacity of the +/- 1 std band. Defaults to 0.2.
  title: str, default = "Prevalence Plot"
    Plot title. Defaults to "Prevalence Plot".
  grid: bool, default = True
    Whether to draw a background grid. Defaults to True.
  ax: matplotlib.axes.Axes, default = None
    Existing axes to draw on. A new figure/axes pair is created when None. Defaults to None.

  Returns
  -------
  fig: matplotlib.figure.Figure
    The generated figure. Call `fig.savefig(path)` (png, pdf, svg, ... — any Matplotlib-supported format)
    to persist it.

  Examples
  --------
  >>> import numpy as np
  >>> from quack.visualization import prevalence_plot
  >>> rng = np.random.default_rng(0)
  >>> true_prev = rng.uniform(0, 1, 200)
  >>> estim_prev = np.clip(true_prev + rng.normal(0, 0.05, 200), 0, 1)
  >>> fig = prevalence_plot("CC", true_prev, estim_prev, train_prevalence=0.5)
  >>> fig.savefig("prevalence.png", dpi=300)
  """
  method_names = [method_names] if isinstance(method_names, str) else list(method_names)
  true_prevalences = _normalize_experiments(true_prevalences)
  estim_prevalences = _normalize_experiments(estim_prevalences)

  if not (len(method_names) == len(true_prevalences) == len(estim_prevalences)):
    raise ValueError(
      "method_names, true_prevalences and estim_prevalences must have the "
      "same length (one entry per experiment)."
    )

  unique_methods = list(dict.fromkeys(method_names))  # preserves first-seen order
  colors = get_color_palette(len(unique_methods), palette=colors)
  markers = get_marker_cycle(len(unique_methods), markers=markers)
  legend_font_size = legend_font_size if legend_font_size is not None else max(font_size - 1, 6)

  bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
  bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

  own_axes = ax is None
  if own_axes:
    fig, ax = plt.subplots(figsize=fig_size)
  else:
    fig = ax.get_figure()

  ax.set_aspect('equal')
  ax.plot([0, 1], [0, 1], color=REFERENCE_COLOR, linestyle='--', linewidth=1,
          label='Ideal quantifier (y=x)', zorder=1)

  for idx, method in enumerate(unique_methods):
    true_pool = np.concatenate([
      np.atleast_1d(true_prevalences[i]) for i, name in enumerate(method_names) if name == method
    ])
    estim_pool = np.concatenate([
      np.atleast_1d(estim_prevalences[i]) for i, name in enumerate(method_names) if name == method
    ])

    bin_idx = np.clip(np.digitize(true_pool, bin_edges[1:-1]), 0, n_bins - 1)
    means = np.full(n_bins, np.nan)
    stds = np.full(n_bins, np.nan)
    for b in range(n_bins):
      values = estim_pool[bin_idx == b]
      if values.size > 0:
        means[b] = values.mean()
        stds[b] = values.std() if values.size > 1 else 0.0

    valid = ~np.isnan(means)
    color = colors[idx]
    ax.plot(bin_centers[valid], means[valid], label=method, color=color,
            marker=markers[idx], markersize=marker_size, linewidth=line_width,
            zorder=3)

    if show_std:
      ax.fill_between(bin_centers[valid], np.clip(means[valid] - stds[valid], 0, 1),
                       np.clip(means[valid] + stds[valid], 0, 1), color=color,
                       alpha=band_alpha, zorder=2, linewidth=0)

  if train_prevalence is not None:
    train_prevalence = (train_prevalence if isinstance(train_prevalence, (list, tuple, np.ndarray))
                        else [train_prevalence])
    for p in train_prevalence:
      ax.scatter(p, p, s=(marker_size * 9), color='red', edgecolors=REFERENCE_COLOR,
                 marker='*', linewidth=1.2, zorder=5,
                 label=f'Training prevalence (p={p:.2f})')

  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.set_xlabel(f"True prevalence ({class_name})", fontsize=font_size)
  ax.set_ylabel(f"Estimated prevalence ({class_name})", fontsize=font_size)
  if title:
    ax.set_title(title, fontsize=font_size + 2)
  if grid:
    ax.grid(alpha=0.3)

  handles, labels = ax.get_legend_handles_labels()
  by_label = dict(zip(labels, handles))  # de-duplicate repeated train-prevalence labels
  ax.legend(by_label.values(), by_label.keys(), fontsize=legend_font_size,
            loc='upper center', bbox_to_anchor=(0.5, -0.15),
            ncol=min(3, len(by_label)), frameon=False)

  if own_axes:
    fig.tight_layout()

  return fig
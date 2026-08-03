# src/quack/visualization/_bias.py
"""Bias plot.

Shows, via box plots, the signed error `estimated - true` prevalence for
the target class, evincing any systematic tendency of a quantifier to
over or under-estimate it. An unbiased quantifier has a box centered at 0.
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


def bias_plot(
  method_names: str | Sequence[str],
  true_prevalences: np.ndarray | Sequence[np.ndarray],
  estim_prevalences: np.ndarray | Sequence[np.ndarray],
  class_name: str = "positive class",
  n_bins: int = 1,
  colors: Sequence = None,
  fig_size: tuple[float, float] = (8, 6),
  font_size: int = 11,
  legend_font_size: int = None,
  title: str = "Bias Plot",
  box_width: float = 0.6,
  grid: bool = True,
  ax: matplotlib.axes.Axes = None,
) -> matplotlib.figure.Figure:
  """Plot the distribution of signed prevalence errors per method as box plots.

  The bias for a single test bag is defined as
  `bias = estimated_prevalence - true_prevalence` for `class_name`. A value
  of 0 indicates a perfectly unbiased estimate; positive values indicate a
  tendency to overestimate the class, negative values a tendency to
  underestimate it.

  When `n_bins > 1`, the true test prevalence range `[0, 1]` is split into
  `n_bins` equal-width intervals and one group of boxes (one box per
  method) is drawn per interval, mirroring QuaPy's `binary_bias_bins`, so
  that bias-vs-prevalence patterns invisible in the global view can be
  detected.

  Parameters
  ----------
  method_names: str | Sequence[str]
    Name of the method for each experiment (can repeat across datasets/folds;
    matching experiments are pooled together).
  true_prevalences: np.ndarray | Sequence[np.ndarray]
    True prevalence of `class_name` per test bag, one 1D array per experiment.
  estim_prevalences: np.ndarray | Sequence[np.ndarray]
    Predicted prevalence of `class_name`, same shape as `true_prevalences`.
  class_name: str, default = "positive class"
    Target class label used in axis/legend text. Defaults to "positive class".
  n_bins: int, default = 1
    Number of equal-width true-prevalence bins. Use `1` for a single global box
    per method; use `>1` to break it down by true-prevalence range.
    Defaults to 1.
  colors: Sequence, default = colorblind-safe palette
    Custom colors, one per unique method. Defaults to a colorblind-safe palette,
    auto-extended as needed.
  fig_size: tuple[float, float], default = (8, 6)
    Figure size in inches. Defaults to (8, 6).
  font_size: int, default = 11
    Base font size for axis labels/title. Defaults to 11.
  legend_font_size: int, default = None
    Legend font size. Defaults to `font_size - 1` when None.
  title: str, default = "Bias Plot"
    Plot title. Defaults to "Bias Plot".
  box_width: float, default = 0.6
    Width of each individual box. Defaults to 0.6.
  grid: bool, default = True
    Whether to draw a background grid. Defaults to True.
  ax: matplotlib.axes.Axes, default = None
    Existing axes to draw on. A new figure/axes pair is created
    when None. Defaults to None.

  Returns
  -------
  fig: matplotlib.figure.Figure
    The generated figure. Call `fig.savefig(path)` to persist it in any
    Matplotlib-supported format.

  Examples
  --------
  >>> import numpy as np
  >>> from quack.visualization import quantification_bias_plot
  >>> rng = np.random.default_rng(0)
  >>> true_prev = rng.uniform(0, 1, 200)
  >>> estim_prev = np.clip(true_prev + 0.1 + rng.normal(0, 0.05, 200), 0, 1)
  >>> fig = quantification_bias_plot("CC", true_prev, estim_prev, n_bins=3)
  """
  method_names = [method_names] if isinstance(method_names, str) else list(method_names)
  true_prevalences = _normalize_experiments(true_prevalences)
  estim_prevalences = _normalize_experiments(estim_prevalences)

  if not (len(method_names) == len(true_prevalences) == len(estim_prevalences)):
    raise ValueError(
      "method_names, true_prevalences and estim_prevalences must have the "
      "same length (one entry per experiment)."
    )

  unique_methods = list(dict.fromkeys(method_names))
  n_methods = len(unique_methods)
  colors = get_color_palette(n_methods, palette=colors)
  legend_font_size = legend_font_size if legend_font_size is not None else max(font_size - 1, 6)

  pooled_true, pooled_bias = {}, {}
  for method in unique_methods:
    true_pool = np.concatenate([
      np.atleast_1d(true_prevalences[i]) for i, name in enumerate(method_names) if name == method
    ])
    estim_pool = np.concatenate([
      np.atleast_1d(estim_prevalences[i]) for i, name in enumerate(method_names) if name == method
    ])
    pooled_true[method] = true_pool
    pooled_bias[method] = estim_pool - true_pool

  own_axes = ax is None
  if own_axes:
    fig, ax = plt.subplots(figsize=fig_size)
  else:
    fig = ax.get_figure()

  bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
  intra_gap = box_width * 1.15  # spacing between method boxes within a prevalence-bin group

  legend_handles = []
  for m_idx, method in enumerate(unique_methods):
    offset = (m_idx - (n_methods - 1) / 2) * intra_gap
    box_data, positions = [], []

    for b in range(n_bins):
      lo, hi = bin_edges[b], bin_edges[b + 1]
      in_bin = ((pooled_true[method] >= lo) &
                (pooled_true[method] <= hi if b == n_bins - 1 else pooled_true[method] < hi))
      values = pooled_bias[method][in_bin]
      if values.size > 0:
        box_data.append(values)
        positions.append(b + offset)

    if box_data:
      bp = ax.boxplot(box_data, positions=positions, widths=box_width,
                       patch_artist=True, showfliers=False, manage_ticks=False)
      for patch in bp['boxes']:
        patch.set_facecolor(colors[m_idx])
        patch.set_alpha(0.75)
        patch.set_edgecolor(REFERENCE_COLOR)
      for element in ('whiskers', 'caps', 'medians'):
        for artist in bp[element]:
          artist.set_color(REFERENCE_COLOR)
      legend_handles.append(plt.Line2D([0], [0], marker='s', linestyle='',
                                        markerfacecolor=colors[m_idx],
                                        markeredgecolor=REFERENCE_COLOR,
                                        markersize=10, label=method))

  ax.axhline(0.0, color=REFERENCE_COLOR, linestyle='--', linewidth=1, zorder=0)

  if n_bins == 1:
    ax.set_xticks([0])
    ax.set_xticklabels([""])
    ax.set_xlabel(f"Method (target: {class_name})", fontsize=font_size)
  else:
    tick_positions = list(range(n_bins))
    tick_labels = [f"[{bin_edges[b]:.2f}, {bin_edges[b+1]:.2f}]" for b in range(n_bins)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha='right')
    ax.set_xlabel(f"True prevalence range ({class_name})", fontsize=font_size)

  ax.set_ylabel(f"Bias (estimated - true) for {class_name}", fontsize=font_size)
  if title:
    ax.set_title(title, fontsize=font_size + 2)
  if grid:
    ax.grid(axis='y', alpha=0.3)

  ref_handle = plt.Line2D([0], [0], color=REFERENCE_COLOR, linestyle='--', label='Unbiased (bias = 0)')
  ax.legend(handles=legend_handles + [ref_handle], fontsize=legend_font_size,
            loc='upper center', bbox_to_anchor=(0.5, -0.2),
            ncol=min(4, n_methods + 1), frameon=False)

  if own_axes:
    fig.tight_layout()

  return fig
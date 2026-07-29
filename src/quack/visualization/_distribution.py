# src/quack/visualization/_distribution.py
"""Class distribution plot.

A colorblind-safe bar plot showing the number (or proportion) of
instances per class in an array-like of labels (e.g. a training set or
test bag), useful for sanity-checking the prevalence assumptions behind a
quantification experiment.
"""
from typing import Sequence
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from quack.visualization._colors import get_color_palette, REFERENCE_COLOR


def class_distribution_plot(
  y: np.ndarray,
  normalize: bool = False,
  title: str = "Class Distribution",
  fig_size: tuple[float, float] = (8, 5),
  colors: Sequence = None,
  font_size: int = 11,
  bar_label_font_size: int = None,
  show_bar_labels: bool = True,
  horizontal: bool = False,
  grid: bool = True,
  ax: matplotlib.axes.Axes = None,
) -> matplotlib.figure.Figure:
  """Plot the (optionally normalized) class distribution of a label array.

  Parameters
  ----------
  y: np.ndarray
    Array-like with all labels.
  normalize: bool, default = False
    If True, plot relative frequencies (prevalences summing to 1.0)
    instead of raw counts. Defaults to False.
  title: str, default = "Class Distribution"
    Plot title. Defaults to "Class Distribution".
  fig_size: tuple[float, float], default = (8, 5)
    Figure size in inches. Defaults to (8, 5).
  colors: Sequence, default = colorblind-safe palette
    Custom colors, one per class. Defaults to a colorblind-safe palette,
    auto-extended for many classes.
  font_size: int, default = 11
    Base font size for axis labels/title. Defaults to 11.
  bar_label_font_size: int, default = None
    Font size for the value labels drawn on top of each bar.
    Defaults to `font_size - 1` when None.
  show_bar_labels: bool, default = True
    Whether to annotate each bar with its value. Defaults to True.
  horizontal: bool, default = False
    If True, draws horizontal bars
    (useful for many classes / long class names). Defaults to False.
  grid: bool, default =True
    Whether to draw a background grid. Defaults to True.
  ax: matplotlib.axes.Axes, default = None
    Existing axes to draw on. A new figure/axes pair is
    created when None. Defaults to None.

  Returns
  -------
  fig: matplotlib.figure.Figure
    The generated figure. Call `fig.savefig(path)` to persist it in any
    Matplotlib-supported format.

  Examples
  --------
  >>> import numpy as np
  >>> from quack.visualization import class_distribution_plot
  >>> y = np.random.choice([0, 1, 2], size=500, p=[0.6, 0.3, 0.1])
  >>> fig = class_distribution_plot(y, normalize=True)
  >>> fig.savefig("class_distribution.pdf")
  """
  bar_label_font_size = bar_label_font_size if bar_label_font_size is not None else max(font_size - 1, 6)

  series = pd.Series(np.asarray(y), name="class")
  counts = series.value_counts().sort_index()
  values = (counts / counts.sum()) if normalize else counts

  classes = [str(c) for c in values.index]
  colors = get_color_palette(len(classes), palette=colors)

  own_axes = ax is None
  if own_axes:
    fig, ax = plt.subplots(figsize=fig_size)
  else:
    fig = ax.get_figure()

  if horizontal:
    bars = ax.barh(classes, values.values, color=colors, edgecolor=REFERENCE_COLOR, linewidth=0.5)
    ax.set_xlabel("Proportion" if normalize else "# instances", fontsize=font_size)
    ax.set_ylabel("Class", fontsize=font_size)
  else:
    bars = ax.bar(classes, values.values, color=colors, edgecolor=REFERENCE_COLOR, linewidth=0.5)
    ax.set_ylabel("Proportion" if normalize else "# instances", fontsize=font_size)
    ax.set_xlabel("Class", fontsize=font_size)

  if show_bar_labels:
    fmt = "{:.3f}" if normalize else "{:.0f}"
    ax.bar_label(bars, labels=[fmt.format(v) for v in values.values],
                 fontsize=bar_label_font_size, padding=2)

  if title:
    ax.set_title(title, fontsize=font_size + 2)
  if grid:
    ax.grid(axis='x' if horizontal else 'y', alpha=0.3)

  if own_axes:
    fig.tight_layout()

  return fig
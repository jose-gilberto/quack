import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math


COLORBLIND_COLORMAP = [
  '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
  '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF',
  '#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5',
  '#C49C94', '#F7B6D2', '#C7C7C7', '#DBDB8D', '#9EDAE5', 
]


def binary_prevalence_plot(
  results: dict[str, list[float]],
  true_prevalences: list[float],
  train_prevalence: float | list[float] = None,
  fig_size: tuple[int, int] = (6, 6),
  title: str = None,
  color_map: list[str | tuple[float]] = None,
) -> matplotlib.figure.Figure:
  """
  Plot predicted vs. true class prevalence for one or multiple quantifiers.

  This function visualizes the estimated prevalence of positive samples as predicted
  by one or more quantifiers, compared against the true class prevalence observed in
  test sets. Each model is represented by a line with distinct color, marker, and
  linestyle. The diagonal dashed line (y = x) represents the ideal case where
  the predicted prevalence exactly matches the true one.

  Optionally, one or more training prevalences can be plotted as red dots markers,
  allowing the user to visualize the training points used to train the quantifiers
  in comparison to the test distributions.

  Args:
    results (dict[str, list[float]]): 
      A mapping of model names to their predicted prevalences. Each list must
      correspond to the true prevalences in `true_prevalences` parameter.  
      Example structure:
      {
        'Model A': [0.04, 0.08, 0.12, 0.18, ...],
        'Model B': [0.05, 0.09, 0.11, 0.17, ...]
      }
      
    true_prevalences (list[float]):
      Array like of shape (n_points,). True class prevalences for x-axis.
    train_prevalence (float | list[float], optional): 
      One or more training prevalences (0-1 range) to be shown as red dot markers
      on the plot, marking where quantifiers were trained. Defaults to None.
    fig_size (tuple[int, int], optional): 
      Figure size in inches. Defaults to (6, 6).
    title (str, optional): 
      Title for the plot. Defaults to None.
    color_map (list[str  |  tuple[float]], optional): 
      List of hex colors or RGB tuples to be used as color maps for each quantifier.
      Defaults to None.

  Returns:
    matplotlib.figure.Figure:
      The generated Matplotlib Figure object.

  Notes
  -----
  - The diagonal dashed line `y = x` indicates perfect estimation.
  - The legend position is automatically adjusted to the bottom of the figure,
    with dynamic spacing and column count (reduced to 2 if any label exceeds
    20 characters).

  Examples
  --------
  >>> import numpy as np
  >>> from quack.plots import plot_prevalence_estimation
  >>>
  >>> x_real = np.arange(0.05, 1.0, 0.05)
  >>> results = {
  ...     "CC": np.clip(x_real + np.random.uniform(-0.05, 0.05, len(x_real)), 0, 1),
  ...     "ACC": np.clip(x_real + np.random.uniform(-0.03, 0.03, len(x_real)), 0, 1),
  ... }
  >>>
  >>> fig = plot_prevalence_estimation(
  ...     results,
  ...     x_values=x_real,
  ...     p_train=0.25,
  ...     title="Predicted vs. True Prevalence"
  ... )
  >>> fig.show()
  """
  fig, ax = plt.subplots(figsize=fig_size)
  ax.set_aspect('equal') # Control the aspect ratio between x and y axis

  # Prepare all variations for model lines to avoid repetition
  markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]
  colors = COLORBLIND_COLORMAP if color_map is None else color_map

  ax.grid()

  # We first create the 
  ax.plot(
    [0, 1],
    [0, 1],
    color='black',
    linestyle='--',
    linewidth=1,
    label='Ideal Quantifier (x=y)',
    zorder=1
  )

  for j, (model_name, preds) in enumerate(results.items()):
    marker = markers[j % len(markers)]
    color = colors[j % len(colors)]
    ax.plot(
      true_prevalences,
      preds,
      label=model_name,
      marker=marker,
      color=color,
      linewidth=1,
      markersize=4
    )

  if train_prevalence is not None:
    if not isinstance(train_prevalence, (list, tuple, np.ndarray)):
      train_prevalence = [train_prevalence]
    for p in train_prevalence:
      ax.scatter(p, p, s=45, color='red', marker='o', edgecolors='black',
                 linewidth=0.8, zorder=5, label=f'Train prevalence (p={p})')

  # Set the axis and plot titles
  ax.set_xlabel('True prevalence')
  ax.set_ylabel('Estimated prevalence')
  ax.set_title(title)

  # Set the limits between [0,1]
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)

  handles, labels = ax.get_legend_handles_labels()
  n_labels = len(labels)

  # If we have some label with more than 20 characters
  # the maximum number of cols is 2.
  max_label_len = max(len(lbl) for lbl in labels)
  if max_label_len > 20:
    ncols = 2 
  else:
    ncols = min(3, n_labels)

  n_rows = math.ceil(n_labels / ncols)

  legend_height = 0.10 + 0.05 * (n_rows - 1)
  legend_height = min(legend_height, 0.35)

  ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.25 - (n_rows - 1) * 0.05),
    ncol=ncols,
    frameon=False
  )

  plt.tight_layout()
  plt.subplots_adjust(bottom=0.25)

  return fig
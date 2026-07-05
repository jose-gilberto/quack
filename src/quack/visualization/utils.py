import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_class_distribution(
    y: np.ndarray,
    title: str = "Class Distribution Plot",
    figsize: tuple[int, int] = (10, 6),
    palette: str = "viridis"
) -> None:
  """
  Plot the class distribution in an array-like.

  Args:
      y: Array with all labels.
      title: Title for the plot.
  """
  sns.set_theme(style="whitegrid")
  
  series = pd.Series(y, name="Classes")
  df = series.value_counts().reset_index()
  df.columns = ["class", "count"]

  plt.figure(figsize=figsize)
  
  ax = sns.barplot(
    data=df, 
    x="class", 
    y="count", 
    palette=palette,
    hue="class",
    legend=False
  )

  plt.title(title, fontsize=16, pad=20)
  plt.xlabel("Class", fontsize=12)
  plt.ylabel("# instances", fontsize=12)

  for container in ax.containers:
      ax.bar_label(container)

  plt.tight_layout()
  plt.show()
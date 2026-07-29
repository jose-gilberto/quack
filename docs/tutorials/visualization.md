title: Visualization

# Visualization

`quack.visualization` provides three colorblind-safe, fully customizable Matplotlib plots to inspect quantification experiments, inspired by [QuaPy's plotting module](https://hlt-isti.github.io/QuaPy/manuals/plotting.html):

- [`prevalence_plot`](../api/visualization.md#prevalence-plot) — the *diagonal plot*: predicted vs. true prevalence for a target class (binary).
- [`bias_plot`](../api/visualization.md#bias-plot) — box plots of signed error (`estimated - true`), global or binned by true prevalence.
- [`class_distribution_plot`](../api/visualization.md#class-distribution-plot) — bar plot of class counts/proportions in a label array.

Every function returns a `matplotlib.figure.Figure` — nothing is shown or saved automatically, so you decide when/how to display or persist it:

```python
fig = prevalence_plot(...)
fig.savefig("prevalence.png", dpi=300)   # or .pdf, .svg, .jpg, ...
```

!!! note
    These plots depend on `matplotlib`, an optional dependency. Install it with:
    ```bash
    pip install quack[viz]
    ```

## Prevalence Plot (diagonal plot)

Plots the predicted prevalence of a target class (y-axis) against its true prevalence (x-axis). Repeated experiments for the same method are pooled and binned, showing the mean trend with an optional +/- 1 std band — analogous to QuaPy's `binary_diagonal`.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

from quack.datasets import load_uci
from quack.bag_generator import PriorShiftBagGenerator  # generate test bags across the prior spectrum
from quack.quantifiers import CC, ACC, PACC
from quack.visualization import prevalence_plot

X, y = load_uci("bc-cont")
# ... split into train/test, fit CC/ACC/PACC, and generate bags with varying priors ...

method_names, true_prevs, estim_prevs = [], [], []
for name, quantifier in [("CC", CC(LogisticRegression())), ("ACC", ACC(LogisticRegression()))]:
    quantifier.fit(X_train, y_train)
    true_bag_prevs, estim_bag_prevs = [], []
    for X_bag, y_bag in bags:  # e.g. produced by PriorShiftBagGenerator
        true_bag_prevs.append(np.mean(y_bag == 1))
        estim_bag_prevs.append(quantifier.predict(X_bag)[1])
    method_names.append(name)
    true_prevs.append(np.array(true_bag_prevs))
    estim_prevs.append(np.array(estim_bag_prevs))

fig = prevalence_plot(
    method_names, true_prevs, estim_prevs,
    class_name="malignant",
    train_prevalence=np.mean(y_train == 1),
    n_bins=21,
)
fig.savefig("bc_cont_diagonal.png", dpi=300)
```

A method name can repeat across several experiments (e.g. one array per fold or dataset); matching entries are merged before binning, exactly like QuaPy's convention of allowing repeated `method_names`.

### Key parameters

| Parameter | Purpose |
|---|---|
| `class_name` | Label used on axes/legend for the target class. |
| `train_prevalence` | Marks one or more training priors on the diagonal with a star marker. |
| `n_bins` | Number of true-prevalence bins used to aggregate repeated experiments. |
| `show_std` | Toggle the +/- 1 std shaded band. |
| `colors`, `markers` | Override the default colorblind-safe palette / marker cycle. |
| `fig_size`, `font_size`, `legend_font_size` | Layout and typography controls. |
| `ax` | Draw into an existing `Axes` to compose custom multi-panel figures. |

## Bias Plot

Shows, via box plots, the signed error `estimated - true` prevalence. A box centered at 0 indicates an unbiased quantifier; positive boxes indicate systematic overestimation, negative boxes underestimation.

```python
from quack.visualization import bias_plot

# Global bias per method (QuaPy's binary_bias_global)
fig = bias_plot(method_names, true_prevs, estim_prevs, class_name="malignant")
fig.savefig("bc_cont_bias_global.png")

# Bias broken down by true-prevalence range (QuaPy's binary_bias_bins)
fig = bias_plot(
    method_names, true_prevs, estim_prevs,
    class_name="malignant", n_bins=3,
)
fig.savefig("bc_cont_bias_bins.png")
```

This is particularly useful to spot patterns invisible in the global view — e.g. a method trained at a balanced prior can still show positive bias for low-prevalence bags and negative bias for high-prevalence ones.

## Class Distribution Plot

A quick sanity check for the label distribution of a training set or test bag:

```python
from quack.visualization import class_distribution_plot

fig = class_distribution_plot(y_train, normalize=True, title="Training set class distribution")
fig.savefig("train_distribution.png")
```

Set `horizontal=True` for datasets with many classes or long class-name labels.

## Combining plots in a single figure

Every function accepts an `ax` argument, so you can compose a multi-panel report using standard Matplotlib subplots:

```python
import matplotlib.pyplot as plt
from quack.visualization import prevalence_plot, bias_plot, class_distribution_plot

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

prevalence_plot(method_names, true_prevs, estim_prevs, ax=axes[0], title="Diagonal Plot")
bias_plot(method_names, true_prevs, estim_prevs, ax=axes[1], title="Bias")
class_distribution_plot(y_train, ax=axes[2], title="Train Distribution")

fig.tight_layout()
fig.savefig("report.pdf")
```

## Color palette

All plots share a colorblind-safe palette ([Okabe & Ito, 2008](https://jfly.uni-koeln.de/color/)) via `quack.visualization.get_color_palette`, which automatically extends to any number of methods/classes by sampling a perceptually-uniform colormap once the base palette is exhausted — no two series ever share the same color, regardless of how many quantifiers are being compared.

```python
from quack.visualization import get_color_palette

colors = get_color_palette(12)  # 8 base colorblind-safe colors + 4 extended
fig = prevalence_plot(method_names, true_prevs, estim_prevs, colors=colors)
```
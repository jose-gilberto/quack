# Quack 🦆

**Quack** is a lightweight, clean, and extensible Python library dedicated to **Quantification** (also known as supervised prevalence estimation).

Unlike standard classification tasks that predict instance-level labels, quantification algorithms estimate population-level class prevalence over a set of target samples (a "bag"). This is particularly useful in scenarios subject to dataset shift, such as Prior Probability Shift and Covariate Shift.

> **Status**: `quack` is in **alpha**. APIs may change between releases. If you use it in production, pin to a specific commit/tag on the `main` branch.

---

## Why quack?

- **Pure Python & SciPy Stack** — built on top of NumPy, SciPy, and Scikit-Learn, no heavy external dependencies.
- **Unified Interface** — every quantifier implements a clean, Scikit-Learn-compatible `.fit(X, y)` / `.predict(X)` API.
- **Shift Simulation** — built-in bag generators to simulate and test algorithms against controlled dataset shifts.
- **Core Metrics** — standard error metrics designed specifically to measure quantification performance.
- **Dataset Loaders** — ready-to-use loaders for ~30 UCI datasets plus review-sentiment and Forman (2008) benchmarks.

---

## Installation

`quack` is currently only available via git (no PyPI release yet):

```bash
pip install git+https://github.com/jose-gilberto/quack@main
```

For development (editable install with test/docs extras):

```bash
git clone https://github.com/jose-gilberto/quack.git
cd quack
pip install -e ".[test,docs]"
```

**Requirements**: Python >= 3.8, `scikit-learn>=1.9.0`, `scipy>=1.17.0`, `pandas>=3.0.3`, `numpy>=2.5.0`, `cvxpy>=1.9.2`.

---

## Quickstart

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

from quack.datasets import load_uci
from quack.quantifiers import CC
from quack.metrics import ae

# 1. Load a dataset and split into train/test
X, y = load_uci("bc-cont")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. True prevalence on the test bag
labels, counts = np.unique(y_test, return_counts=True)
true_prev = counts / y_test.shape[0]

# 3. Fit a Classify & Count (CC) quantifier
quantifier = CC(classifier=LogisticRegression(max_iter=1000))
quantifier.fit(X_train, y_train)

predicted_prev = quantifier.predict(X_test)

# 4. Evaluate using Absolute Error
error = ae(true_prev, predicted_prev)
print(f"True: {true_prev[1]:.2f} | Pred: {predicted_prev[1]:.2f} | AE: {error:.4f}")
```

---

## Features

### Quantification methods

| Category | Methods |
|---|---|
| Classify, Count & Correct | `CC`, `ACC`, `PCC`, `PACC`, `GAC`, `GPAC`, `FM`, threshold selectors `X`, `Max`, `T50`, `MedianSweep` |
| Distribution Matching — Feature Based | `HDx`, `ReadMe`, `ED` |
| Distribution Matching — Label Based | `HDy`, `DyS`, `FormanMM` |
| Distribution Matching — Iterators | `EM` (Expectation Maximization), `CDE` |

```python
from quack.quantifiers import CC, ACC, PCC, PACC, HDy, DyS, EM, CDE, HDx, ReadMe, ED
```

### Datasets

```python
from quack.datasets import load_uci, UCI_DATASETS, load_forman, FORMAN_DATASETS
```

- `load_uci(name)` — ~30 UCI Machine Learning Repository datasets (see `UCI_DATASETS` for the full list of keys).
- `load_forman(name)` — datasets from Forman (2008): `fbis`, `la1`, `la2`, `ohscal`.

### Metrics

```python
from quack.metrics import ae, rae, kld, nkld
```

- `ae` — Absolute Error
- `rae` — Relative Absolute Error
- `kld` — Kullback-Leibler Divergence
- `nkld` — Normalized Kullback-Leibler Divergence

### Visualization

```python
from quack.visualization.binary import binary_prevalence_plot
from quack.visualization.utils import plot_class_distribution
```

---

## Project Structure

```
src/quack/
├── quantifiers/     # Quantification algorithms (BaseQuantifier subclasses)
├── datasets/        # Dataset loaders (UCI, Forman, Reviews)
├── metrics/         # Quantification-specific error metrics
├── ensembles/        # Ensemble strategies (WIP)
└── visualization/   # Plotting utilities
```

All quantifiers inherit from `quack.quantifiers.base.BaseQuantifier` and implement:

```python
def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseQuantifier': ...
def predict(self, X: np.ndarray) -> np.ndarray:  # normalized prevalence vector
```

---

## Documentation

Full documentation (getting started guide, tutorials, and API reference) is built with `mkdocs`:

```bash
pip install -e ".[docs]"
mkdocs serve
```

---

## Testing

```bash
pip install -e ".[test]"
pytest
```

---

## Roadmap

- [ ] Bag generation and shift-simulation test pipelines
- [ ] Ensemble strategies (e.g., `EoQ`)
- [ ] Visualization module improvements

---

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines before opening a pull request.

---

## Citing

If you use `quack` in your research, please consider citing the repository:

```bibtex
@misc{quack,
  author       = {de Medeiros Júnior, J.G.B.},
  title        = {Quack: A Quantification Kit Library for Python},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/jose-gilberto/quack}},
}
```

# Quantification Methods

The methods are organized according to the taxonomy proposed by [1] and used in [2], resulting in 3 groups:

- **Classify, Count and Correct**: this category uses classifiers to individually classify each sample and count them grouping by the class labels. Some methods may include a correction step to the counts obtained by the classifier.
- **Distribution Matching**: in this category, the methods model the training distribution and search for the parameters that provide the best match with the test distribution.
- **Adaptation of Classification Algorithms**: this category adapts existing classification algorithms to work under quantification tasks.

All quantification methods inherit from `BaseQuantifier` and `BaseEstimator`, guaranteeing compatibility with scikit-learn:

```python
  @abstractmethod
  def fit(self, X: np.ndarray, y: np.ndarray) -> T:
    pass

  @abstractmethod
  def predict(self, X: np.ndarray) -> np.ndarray:
    pass
```

In order to import them, use:

```python
from quack.quantifiers import CC, ACC, PCC, PACC, GAC, GPAC, FM, X, Max, T50, MedianSweep, HDx, ReadMe, HDy, DyS, FormanMM, ED, EM, CDE
```

Every example below uses the same synthetic setup:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

X, y = make_classification(n_samples=2000, n_classes=2, weights=[0.65, 0.35], random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
```

## Parallelism (`n_jobs` / `parallel_backend`)

Every quantifier built on `BaseCalibratedQuantifier` (`ACC`, `PACC`, `X`, `Max`, `T50`, `MedianSweep`, `DyS`, `HDy`, `FormanMM`, `GAC`, `GPAC`, `FM`, `EM`, `CDE`) accepts `n_jobs` and `parallel_backend`, dispatching every cross-validation fold — plus the final full-data refit — as independent `joblib` jobs:

```python
quantifier = ACC(LogisticRegression(max_iter=1000), cv=10, n_jobs=-1, parallel_backend="loky")
```

- `n_jobs=None` (default): sequential, identical to running without parallelism.
- `n_jobs=-1`: uses all available CPU cores.
- `parallel_backend="threading"`: preferable when the base classifier releases the GIL during `fit`/`predict` (most scikit-learn estimators backed by Cython/BLAS, e.g. `LogisticRegression`, `SVC`).

`ReadMe` and `ED` expose the same `n_jobs`/`parallel_backend` parameters, but parallelize their own independent units of work (random subspaces, and class-pair distance sums, respectively) instead of CV folds — see their sections below.

---

## 1. Classify, Count and Correct

### Classify & Count (CC)

The simplest baseline: classifies every test instance and counts the relative frequency of each predicted class.

```python
from quack.quantifiers import CC

quantifier = CC(classifier=LogisticRegression(max_iter=1000))
quantifier.fit(X_train, y_train)
prevalences = quantifier.predict(X_test)
print(prevalences)  # array([p_class_0, p_class_1])
```

### Adjusted Classify & Count (ACC)

Corrects CC's counts using the classifier's True Positive Rate / False Positive Rate, estimated out-of-fold. Binary only.

```python
from quack.quantifiers import ACC

quantifier = ACC(classifier=LogisticRegression(max_iter=1000), cv=10, n_jobs=-1)
quantifier.fit(X_train, y_train)
prevalences = quantifier.predict(X_test)
```

### Probabilistic Classify & Count (PCC)

Uses soft `predict_proba` scores instead of hard predictions, averaging the posterior probabilities directly.

```python
from quack.quantifiers import PCC

quantifier = PCC(classifier=LogisticRegression(max_iter=1000))
quantifier.fit(X_train, y_train)
prevalences = quantifier.predict(X_test)
```

### Probabilistic Adjusted Classify & Count (PACC)

The probabilistic counterpart of ACC: corrects PCC's averaged probabilities using out-of-fold expected scores. Binary only.

```python
from quack.quantifiers import PACC

quantifier = PACC(classifier=LogisticRegression(max_iter=1000), cv=10, n_jobs=-1)
quantifier.fit(X_train, y_train)
prevalences = quantifier.predict(X_test)
```

### Generalized Adjusted Classify & Count (GAC)

Multiclass generalization of ACC: builds a full confusion matrix out-of-fold and solves a distance-minimization problem (via CVXPY, with a Golden Section Search fallback for the binary case).

```python
from quack.quantifiers import GAC

quantifier = GAC(classifier=LogisticRegression(max_iter=1000), distance_metric="L2", cv=10, n_jobs=-1)
quantifier.fit(X_train, y_train)  # also works with 3+ classes
prevalences = quantifier.predict(X_test)
```

### Generalized Probabilistic Adjusted Classify & Count (GPAC)

Same idea as GAC, but using soft `predict_proba` scores instead of hard confusion-matrix counts.

```python
from quack.quantifiers import GPAC

quantifier = GPAC(classifier=LogisticRegression(max_iter=1000), distance_metric="L2", cv=10, n_jobs=-1)
quantifier.fit(X_train, y_train)
prevalences = quantifier.predict(X_test)
```

### Friedman's Method (FM)

Compares soft classifier probabilities against the training prevalence, thresholding them into a binary indicator matrix before solving the mixture.

```python
from quack.quantifiers import FM

quantifier = FM(classifier=LogisticRegression(max_iter=1000), distance_metric="L2", cv=10, n_jobs=-1)
quantifier.fit(X_train, y_train)
prevalences = quantifier.predict(X_test)
```

### Threshold Selectors: X, Max, T50, MedianSweep

Binary methods that adjust the decision threshold dynamically over an out-of-fold TPR/FPR grid, each with a different selection criterion (see each class's docstring for the exact formula). All share the same interface:

```python
from quack.quantifiers import X, Max, T50, MedianSweep

for QuantifierCls in (X, Max, T50):
    quantifier = QuantifierCls(classifier=LogisticRegression(max_iter=1000), cv=10, n_jobs=-1)
    quantifier.fit(X_train, y_train)
    print(QuantifierCls.__name__, quantifier.predict(X_test))

# MedianSweep also exposes delta_min, filtering out unstable thresholds
quantifier = MedianSweep(classifier=LogisticRegression(max_iter=1000), cv=10, delta_min=0.25, n_jobs=-1)
quantifier.fit(X_train, y_train)
prevalences = quantifier.predict(X_test)
```

---

## 2. Distribution Matching

### Feature Based

#### Hellinger Distance x (HDx)

Operates directly on (categorical/discretized) features, without a classifier, minimizing the Hellinger Distance between per-feature marginal distributions.

```python
from quack.quantifiers import HDx

quantifier = HDx(use_convex_solver=True)
quantifier.fit(X_train, y_train)
prevalences = quantifier.predict(X_test)
```

#### ReadMe

Ensemble of mixture models trained on random feature subspaces — useful for high-dimensional categorical/bag-of-words data. `n_jobs` parallelizes fitting/predicting the `n_subsets` independent sub-quantifiers; `random_state` makes the random subspaces reproducible.

```python
from quack.quantifiers import ReadMe

quantifier = ReadMe(n_subsets=100, n_features=None, random_state=0, n_jobs=-1)
quantifier.fit(X_train, y_train)
prevalences = quantifier.predict(X_test)
```

#### Energy Distance (ED)

Minimizes the Energy Distance between training class distributions and the test bag, with an exact analytical solution for binary problems and quadratic programming for multiclass. `n_jobs` parallelizes the per-class-pair pairwise-distance computations (most beneficial with large per-class sample counts).

```python
from quack.quantifiers import ED

quantifier = ED(n_jobs=-1)
quantifier.fit(X_train, y_train)
prevalences = quantifier.predict(X_test)
```

### Label Based

#### Hellinger Distance y (HDy)

Bins classifier scores into histograms and minimizes the Hellinger Distance between the training and test score distributions. Binary only.

```python
from quack.quantifiers import HDy

quantifier = HDy(classifier=LogisticRegression(max_iter=1000), n_bins=10, cv=10, n_jobs=-1)
quantifier.fit(X_train, y_train)
prevalences = quantifier.predict(X_test)
```

#### Distribution y-Similarity (DyS)

Generalizes HDy to any supported distance metric (`'L1'`, `'L2'`, `'HD'`, `'TS'`). Binary only.

```python
from sklearn.svm import SVC
from quack.quantifiers import DyS

quantifier = DyS(classifier=SVC(), distance_metric="TS", n_bins=10, cv=10, n_jobs=-1)
quantifier.fit(X_train, y_train)
prevalences = quantifier.predict(X_test)
```

#### Forman's Mixture Model (FormanMM)

Minimizes the L1 distance over the Cumulative Distribution Function (CDF) of classifier scores instead of a fixed-bin histogram. Binary only.

```python
from quack.quantifiers import FormanMM

quantifier = FormanMM(classifier=SVC(), cv=10, n_jobs=-1)
quantifier.fit(X_train, y_train)
prevalences = quantifier.predict(X_test)
```

### Iterator Based

#### Expectation Maximization Quantifier (EM)

Iteratively adjusts posterior probabilities and prior estimates until convergence, maximizing the likelihood of the test data. Multiclass-capable.

```python
from quack.quantifiers import EM

quantifier = EM(classifier=LogisticRegression(max_iter=1000), cv=10, epsilon=1e-6, max_iter=1000, n_jobs=-1)
quantifier.fit(X_train, y_train)
prevalences = quantifier.predict(X_test)
```

---

## 3. Adaptation of Classification Algorithms

### Class Distribution Estimation (CDE)

Iteratively adjusts the decision threshold based on directional weights derived from the training prior. Binary only.

```python
from quack.quantifiers import CDE

quantifier = CDE(classifier=LogisticRegression(max_iter=1000), cv=10, epsilon=1e-6, max_iter=1000, n_jobs=-1)
quantifier.fit(X_train, y_train)
prevalences = quantifier.predict(X_test)
```

---

## Choosing a method: quick reference

| Method | Binary only? | Needs `predict_proba`? | Notes |
|---|---|---|---|
| `CC` | No | No | Fastest, least accurate baseline |
| `PCC` | No | Yes | Sensitive to classifier calibration |
| `ACC` | Yes | No | Classic correction, denominator instability possible |
| `PACC` | Yes | Yes | Probabilistic counterpart of ACC |
| `GAC` | No | No | Multiclass generalization of ACC |
| `GPAC` | No | Yes | Multiclass generalization of PACC |
| `FM` | No | Yes | Threshold vs. training prior |
| `X`, `Max`, `T50` | Yes | Yes | Threshold-selection strategies (Forman, 2008) |
| `MedianSweep` | Yes | Yes | Robust median across many thresholds |
| `HDx` | No | N/A (no classifier) | Categorical/discretized features only |
| `ReadMe` | No | N/A (no classifier) | High-dimensional / bag-of-words data |
| `ED` | No | N/A (no classifier) | Exact binary solution, QP for multiclass |
| `HDy` | Yes | Yes | Hellinger distance on score histograms |
| `DyS` | Yes | Yes | Configurable distance metric |
| `FormanMM` | Yes | Yes | CDF-based, no fixed bin count |
| `EM` | No | Yes | Iterative, theoretically well-grounded |
| `CDE` | Yes | Yes | Iterative threshold adjustment |

## References

[1] González, P., Castaño, A., Chawla, N. V., & Coz, J. J. D. (2017). A review on quantification learning. ACM Computing Surveys (CSUR), 50(5), 1-40.

[2] Donyavi, Z., Serapião, A. B., & Batista, G. (2024). MC-SQ and MC-MQ: Ensembles for multi-class quantification. IEEE Transactions on Knowledge and Data Engineering, 36(8), 4007-4019.
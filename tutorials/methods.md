title: Quantification Methods

The methods are organized according to the taxonomy proposed by [1] and used in [2], resulting in 3 groups:

- **Classify, Count and Correct**: this category uses classifiers to individually classify each sample and count them grouping by the class labels. Some methods may include a correction step to the counts obtained by the classifier. They are:
    1. Classify & Count (CC)
    2. Adjusted Classify & Count (ACC)
    3. Probabilistic Classify & Count (PCC)
    4. Probabilistic Adjusted Classify & Count (PACC)
    5. Generalized Adjusted Classify & Count (GAC)
    6. Generalized Probabilistic Adjusted Classify & Count (GPAC)
    7. Friedman's Method (FM)
    8. Threshold X (X)
    9. Threshold Max (MAX)
    10. Threshold 50 (T50)
    11. MedianSweep (MS)
- **Distribution Matching**: in this category, the methods model the training distribution and search for the parameters that provides the best match or fit with test distribution. We also split then in 3 subcategories:
    - Feature Based Models (those who works on feature space, do not require a classifier)
        1. Hellinger Distance over features (HDx)
        2. ReadMe
        3. Energy Distance Minimization (ED)
    - Label Based Models (those who works with the label)
        1. Hellinger Distance over class histograms (HDy)
        2. Forman's Mixture Model (FMM)
        3. Distribution y-Similarity (DyS)
    - Iterator Based (those who use an optimization process based on iterators)
        1. Expectation Maximization Quantifier
- **Adaptation of Classification Algorithms**: this category adapts existing classification algorithms to work under quantification tasks. For now we have:
    1. Class Distribution Estimation (CDE)


All quantification methods inherits from the `BaseQuantifier` and `BaseEstimator` in order to be compatible with scikit-learn. Method names like `fit` and `predict` are used to provide an easy integration with other modules from scikit. This way all quantifiers must implement the following methods:

```python
  @abstractmethod
  def fit(self, X: np.ndarray, y: np.ndarray) -> T:
    pass

  @abstractmethod
  def predict(self, X: np.ndarray) -> np.ndarray:
    pass
```

In order to import then we only need to use:

```python
from quack.quantifiers import CC, ACC, CDE, HDy, ...
```
# Getting Started

This guide walks you through a instalation, training and testing a classifier-based quantifier, and evaluating the predictions using a `quack`'s built-in metrics.

**Note**: this page may be adjusted in the feature as we are in a `alpha` version of the package, if you use it in any of you projects be careful to clone, install and use a 'stable' version (main branch). If you want to keep up with the updates you can use the 'develop' branch or some specific branch for other features.

## Instalation

For now, to install quack you need to rely on the git pip feature. Just need to run the following command:

```bash
pip install git+https://github.com/jose-gilberto/quack@main
```

## Quickstart Workflow

Considering that you already have a dataset loaded and splitted into training and testing, the following script will create a quantifier based on logistic regression to estimate the prevalence of your set.

Any data compatible with Scikit-learn or Numpy can be used, whether `np.ndarray` or `sparse_matrix` objects.

To evaluate the method, we used the Absolute Error (AE) between the predicted and real prevalences.

```python
from sklearn.linear_model import LogisticRegression
# Import from quack
from quack.quantifiers import CC
from quack.metrics import absolute_error

# 1. Load the dataset
X_train, y_train = ...
X_test, y_test = ...

# 2. Initialize and fit an Adjusted Classify & Count (ACC) quantifier
quantifier = CC(classifier=LogisticRegression())
quantifier.fit(X, y)

predicted_prev = quantifier.predict(X_test)
# Measure performance using Absolute Error
error = absolute_error(true_prev, predicted_prev)

print(f"True: {true_prev[1]:.2f} | Pred: {predicted_prev[1]:.2f} | AE: {error:.4f}")
```

Quantification is very useful in scenarios where prior probability shifts occur.
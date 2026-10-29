title: Bag Generator

# Bag Generator

`quack.bag_generator` simulates **dataset shift** by drawing repeated "bags" (labeled subsets) from a labeled dataset, following the standard *Artificial Prevalence Protocol* (APP) used to benchmark quantifiers across the full spectrum of possible test-time distributions — rather than relying on a single, fixed train/test split.

Two complementary shift generators are provided:

| Generator | Shift type | What's preserved | What changes |
|---|---|---|---|
| [`PriorShiftBagGenerator`](../api/bag_generator.md#prior-shift-bag-generator) | Prior Probability Shift | `P(X\|y)` | `P(y)` |
| [`CovariateShiftBagGenerator`](../api/bag_generator.md#covariate-shift-bag-generator) | Covariate Shift | `P(y\|X)` | `P(X)` (and, indirectly, `P(y)`) |

Both inherit from [`BaseBagGenerator`](../api/bag_generator.md#base-class) and expose the same interface:

```python
generator.generate(X, y)   # lazy generator of (X_bag, y_bag) pairs
generator.to_list(X, y)    # eagerly materializes all bags into a list
```

---

## Prior Shift

Resamples bags across the class-prevalence simplex while keeping each class's feature distribution untouched. Useful for testing whether a quantifier correctly tracks class prevalence changes independent of any change in `P(X|y)`.

```python
from sklearn.datasets import make_classification
from quack.bag_generator import PriorShiftBagGenerator

X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=0)

generator = PriorShiftBagGenerator(
    n_bags=100,
    bag_size=200,
    sampling_strategy="uniform",   # or "dirichlet"
    random_state=42,
)

bags = generator.to_list(X, y)
print(len(bags))                          # 100
print(generator.sampled_prevalences_[:5])  # realized [P(y=0), P(y=1)] per bag
```

### Sampling strategies

- **`"uniform"`** — samples prevalence vectors uniformly over the full simplex (Kraemer algorithm). Good default for a broad, unbiased sweep across possible test priors.
- **`"dirichlet"`** — samples from `Dirichlet(dirichlet_alpha)`, giving control over how extreme the shifts are:

```python
# concentrate prevalences near a single dominant class (extreme shifts)
generator = PriorShiftBagGenerator(
    n_bags=100, bag_size=200,
    sampling_strategy="dirichlet", dirichlet_alpha=0.1,
    random_state=42,
)

# concentrate prevalences near uniform/balanced (mild shifts)
generator = PriorShiftBagGenerator(
    n_bags=100, bag_size=200,
    sampling_strategy="dirichlet", dirichlet_alpha=10.0,
    random_state=42,
)
```

`dirichlet_alpha=1.0` for every class is mathematically equivalent to `"uniform"`.

---

## Covariate Shift

Resamples bags biased towards random regions of the feature space via RBF-kernel similarity to a randomly chosen pivot instance, leaving `P(y|X)` untouched.

```python
from quack.bag_generator import CovariateShiftBagGenerator

generator = CovariateShiftBagGenerator(
    n_bags=100,
    bag_size=200,
    gamma=0.5,       # None defaults to sklearn's 1 / n_features
    random_state=42,
)

bags = generator.to_list(X, y)
print(generator.pivot_indices_[:5])        # pivot instance used per bag
print(generator.sampled_prevalences_[:5])  # resulting class prevalence per bag (side effect)
```

`gamma` controls how tightly each bag clusters around its pivot: higher values produce bags concentrated in a small region of feature space (stronger shift); lower values approach the original, unshifted distribution.

---

## Reproducibility

Pass an `int` to `random_state` to get the exact same sequence of bags across repeated calls — essential when comparing several quantifiers on identical test conditions:

```python
generator = PriorShiftBagGenerator(n_bags=50, bag_size=200, random_state=123)

bags_a = generator.to_list(X, y)
bags_b = generator.to_list(X, y)
# bags_a and bags_b are identical, bag-by-bag
```

---

## End-to-end example: benchmarking quantifiers under Prior Shift

Combine `bag_generator` with `quack.quantifiers` and `quack.visualization` to reproduce a full Artificial Prevalence Protocol experiment:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from quack.datasets import load_uci
from quack.bag_generator import PriorShiftBagGenerator
from quack.quantifiers import CC, ACC, PACC
from quack.metrics import ae
from quack.visualization import prevalence_plot, quantification_bias_plot

X, y = load_uci("bc-cont")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)

generator = PriorShiftBagGenerator(n_bags=200, bag_size=250, random_state=0)
bags = generator.to_list(X_test, y_test)  # reused across every quantifier below

method_names, true_prevs, estim_prevs = [], [], []
for name, quantifier in [
    ("CC", CC(LogisticRegression(max_iter=1000))),
    ("ACC", ACC(LogisticRegression(max_iter=1000))),
    ("PACC", PACC(LogisticRegression(max_iter=1000))),
]:
    quantifier.fit(X_train, y_train)

    bag_true, bag_estim = [], []
    for X_bag, y_bag in bags:
        bag_true.append(np.mean(y_bag == 1))
        bag_estim.append(quantifier.predict(X_bag)[1])

    method_names.append(name)
    true_prevs.append(np.array(bag_true))
    estim_prevs.append(np.array(bag_estim))

    errors = [ae(np.array([1 - t, t]), np.array([1 - p, p])) for t, p in zip(bag_true, bag_estim)]
    print(f"{name}: mean AE = {np.mean(errors):.4f}")

fig = prevalence_plot(method_names, true_prevs, estim_prevs,
                       class_name="malignant", train_prevalence=np.mean(y_train == 1))
fig.savefig("bc_cont_prior_shift_diagonal.png", dpi=300)

fig = quantification_bias_plot(method_names, true_prevs, estim_prevs,
                                class_name="malignant", n_bins=3)
fig.savefig("bc_cont_prior_shift_bias.png", dpi=300)
```

This is exactly the pattern used throughout `quack`'s evaluation tooling: generate bags once with a `BagGenerator`, evaluate every quantifier over the same bags, then feed the collected `(true_prevalences, estim_prevalences)` pairs into `quack.metrics` and `quack.visualization`.
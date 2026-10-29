# Ensembles

`quack.ensembles` provides meta-quantifiers that combine several base quantifiers into a single, typically more robust, prevalence estimate. Currently it ships one implementation:

- [`EoQ`](../api/ensembles.md#ensemble-of-quantifiers-eoq) — Ensemble of Quantifiers, based on Pérez-Gállego et al. (2017, 2019).

`EoQ` trains `n_estimators` independent copies of any `quack` quantifier, each on a bag with an artificially shifted class prevalence drawn via [`quack.bag_generator`](./bag-generator.md), then aggregates their individual predictions. Reusing `bag_generator` means `EoQ` benefits from every bag generator already available — `PriorShiftBagGenerator` (default), `CovariateShiftBagGenerator`, or a custom one you write yourself.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from quack.quantifiers import CC
from quack.ensembles import EoQ

X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.65, 0.35], random_state=0)

ensemble = EoQ(
    base_quantifier=CC(LogisticRegression(max_iter=1000)),
    n_estimators=30,
    selection_policy="average",
    random_state=0,
)
ensemble.fit(X, y)

X_test, _ = make_classification(n_samples=300, n_classes=2, random_state=7)
prevalences = ensemble.predict(X_test)
print(prevalences)  # array([p_class_0, p_class_1])
```

`base_quantifier` can be **any** `quack` quantifier — `CC`, `ACC`, `DyS`, `EM`, another `EoQ`, and so on — since `EoQ` only relies on the shared `.fit(X, y)` / `.predict(X)` contract from `BaseQuantifier`.

---

## Selection policies

`EoQ` supports three ways of turning `n_estimators` individual predictions into one final estimate, controlled by `selection_policy`:

| Policy | Type | What it does |
|---|---|---|
| `"average"` (default) | none | Simple average of every member's prediction. |
| `"performance"` | static | Keeps only the `red_size` members with the lowest quantification error on a held-out validation split, decided once during `fit`. |
| `"ptr"` | dynamic | Keeps only the `red_size` members whose *own training bag* prevalence is closest to an estimate of the current test bag's prevalence — recomputed for every `predict` call. |

### Average (baseline)

No selection — every member contributes equally to the final average. This is `EoQ`'s default and matches the base ensemble from Pérez-Gállego et al. (2017).

```python
ensemble = EoQ(CC(LogisticRegression(max_iter=1000)), n_estimators=30, selection_policy="average")
ensemble.fit(X_train, y_train)
```

### Performance (static selection)

Reserves a fraction (`val_split`) of the training data to generate validation bags with *known* prevalence, measures every member's mean quantification error on them (via any metric in [`quack.metrics`](../api/metrics.md)), and permanently keeps only the `red_size` best members.

```python
ensemble = EoQ(
    CC(LogisticRegression(max_iter=1000)),
    n_estimators=30,
    selection_policy="performance",
    red_size=10,        # keep the 10 best members
    metric="ae",         # any key from quack.metrics.MetricRegistry, or a QuantificationMetric instance
    val_split=0.4,
    random_state=0,
)
ensemble.fit(X_train, y_train)

print(ensemble.oob_scores_)        # mean validation error per member
print(ensemble.selected_indices_)  # indices of the red_size retained members
```

You can also pass a `QuantificationMetric` instance directly instead of a string key:

```python
from quack.metrics import KullbackLeiblerDivergence

ensemble = EoQ(
    CC(LogisticRegression(max_iter=1000)),
    n_estimators=30, selection_policy="performance",
    red_size=10, metric=KullbackLeiblerDivergence(),
)
```

### Training Prevalence / `ptr` (dynamic selection)

For every test bag passed to `predict`, `EoQ` first estimates its prevalence (the average of all `n_estimators` members), then re-averages only the `red_size` members whose own training-bag prevalence sits closest to that estimate. Different test bags can therefore select different subsets of members.

```python
ensemble = EoQ(
    CC(LogisticRegression(max_iter=1000)),
    n_estimators=30,
    selection_policy="ptr",
    red_size=10,
    random_state=0,
)
ensemble.fit(X_train, y_train)

prevalences = ensemble.predict(X_test)  # selection recomputed for this specific bag
```

---

## Choosing a bag generator

By default, `EoQ` uses `PriorShiftBagGenerator(sampling_strategy="uniform")` — a broad, unbiased sweep across the prevalence simplex. Pass your own `bag_generator` to change the shift protocol used to train (and, for `selection_policy="performance"`, validate) the ensemble members:

```python
from quack.bag_generator import PriorShiftBagGenerator, CovariateShiftBagGenerator

# concentrate members' training bags on more extreme priors
ensemble = EoQ(
    CC(LogisticRegression(max_iter=1000)),
    n_estimators=30,
    bag_generator=PriorShiftBagGenerator(sampling_strategy="dirichlet", dirichlet_alpha=0.3, bag_size=200),
    random_state=0,
)

# train members under covariate shift instead of prior shift
ensemble = EoQ(
    CC(LogisticRegression(max_iter=1000)),
    n_estimators=30,
    bag_generator=CovariateShiftBagGenerator(gamma=0.5, bag_size=200),
    random_state=0,
)
```

Only `n_bags` (and, internally, `random_state`) are overridden on the generator you pass in — every other parameter (`bag_size`, `sampling_strategy`, `dirichlet_alpha`, `gamma`, ...) is respected as-is.

!!! note "Every training bag is guaranteed to contain all classes"
    Extreme shift configurations (e.g. a very small `dirichlet_alpha`) can legitimately draw a bag missing one or more classes entirely — most base quantifiers cannot be `.fit()` on such data. `EoQ` deterministically tops up any missing class with one instance swapped in from the bag's currently largest class, so this never causes a training failure; `member_train_prevalences_` always reflects the bag's true final composition.

---

## Parallelism

`EoQ` accepts the same `n_jobs`/`parallel_backend` parameters used throughout `quack.quantifiers` (see [Methods — Parallelism](./methods.md#parallelism-n_jobs--parallel_backend)), parallelizing three independent stages: fitting the `n_estimators` members, scoring them for `selection_policy="performance"`, and predicting with every member.

```python
ensemble = EoQ(
    CC(LogisticRegression(max_iter=1000)),
    n_estimators=100, n_jobs=-1, parallel_backend="loky", random_state=0,
)
ensemble.fit(X_train, y_train)
```

---

## Reproducibility

Pass an `int` to `random_state` to get the exact same sequence of member training bags (and validation bags, for `selection_policy="performance"`) across repeated `fit` calls:

```python
ens_a = EoQ(CC(LogisticRegression(max_iter=1000)), n_estimators=20, random_state=42).fit(X_train, y_train)
ens_b = EoQ(CC(LogisticRegression(max_iter=1000)), n_estimators=20, random_state=42).fit(X_train, y_train)
# ens_a and ens_b produce identical predictions
```

---

## Inspecting the ensemble

After `fit`, a few attributes are available for diagnostics:

```python
ensemble.estimators_                  # list of n_estimators fitted BaseQuantifier instances
ensemble.member_train_prevalences_    # ndarray (n_estimators, n_classes): realized prevalence per member
ensemble.oob_scores_                  # ndarray (n_estimators,) or None: validation error (only 'performance')
ensemble.selected_indices_            # indices retained ('performance'/'average'); unused for 'ptr'
```

Combine `member_train_prevalences_` with [`prevalence_coverage_plot`](./visualization.md#prevalence-coverage-plot) to check how well the ensemble's own training bags span the prevalence range for a given class:

```python
from quack.visualization import prevalence_coverage_plot

fig = prevalence_coverage_plot(
    ensemble.member_train_prevalences_[:, 1],
    labels=["EoQ members"],
    class_name="positive class",
    train_prevalence=ensemble.train_prevalence_[1],
)
fig.savefig("eoq_member_coverage.png", dpi=300)
```

## References

Pérez-Gállego, P., Quevedo, J. R., & del Coz, J. J. (2017). Using ensembles for problems with characterizable changes in data distribution: A case study on quantification. *Information Fusion*, 34, 87-100.

Pérez-Gállego, P., Castaño, A., Quevedo, J. R., & del Coz, J. J. (2019). Dynamic ensemble selection for quantification tasks. *Information Fusion*, 45, 1-15.

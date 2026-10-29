# UCI Datasets

`quack` bundles loaders for ~30 datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/), following the same selection used in [1]. Each dataset is wrapped by a `BaseDatasetLoader` subclass, registered under a short string key in `UCILoaderFactory`, and exposed through the single `load_uci` function.

## Quickstart

```python
from quack.datasets import load_uci

X, y = load_uci("bc-cont")

print(X.shape, y.shape)
```

`load_uci` downloads the raw source file(s) (cached via `urllib`), applies the dataset-specific preprocessing (encoding categoricals, binning continuous targets, dropping constant/leaky columns, etc.), and returns `X` as `float32` and `y` as the label array — ready to be split and fed into any quantifier's `.fit(X, y)`.

## Listing available datasets

```python
from quack.datasets import UCI_DATASETS

print(UCI_DATASETS)
# ['adult', 'avila', 'bike', 'blog', 'bc-cont', 'bc-int', 'cars', 'conc',
#  'contra', 'cappl', 'ccard', 'dota', 'drug', 'ener', 'flare', 'grid',
#  'ads', 'magic', 'boone', 'mush', 'music', 'news', 'nurse', 'occup',
#  'spam', 'cond', 'turk', 'wine', 'yeast']
```

## Dataset reference

| Key      | Dataset                                   | Task type / target binning                |
|----------|--------------------------------------------|---------------------------------------------|
| `adult`  | Adult (Census Income)                       | Binary (`<=50K` / `>50K`)                    |
| `avila`  | Avila                                       | Binary (class A vs. rest)                    |
| `bike`   | Bike Sharing                                | 4-class (`cnt` binned)                       |
| `blog`   | BlogFeedback                                | 4-class (`att280` binned)                    |
| `bc-cont`| Breast Cancer Wisconsin (Original)          | Binary (`diagnosis`)                         |
| `bc-int` | Breast Cancer Wisconsin (Diagnostic)        | Binary (`Class`)                             |
| `cars`   | Car Evaluation                              | 4-class                                      |
| `conc`   | Concrete Compressive Strength               | 3-class (`strength` binned)                  |
| `contra` | Contraceptive Method Choice                 | 3-class                                      |
| `cappl`  | Credit Approval                             | Binary                                       |
| `ccard`  | Default of Credit Card Clients              | Binary                                       |
| `dota`   | Dota2 Games Results                         | Binary (`Winner`)                            |
| `drug`   | Drug Consumption                            | 3-class (grouped `CL0-CL6`)                  |
| `ener`   | Appliances Energy Prediction                | 3-class (`Appliances` binned)                |
| `flare`  | Solar Flare                                 | Binary (`C` binned)                          |
| `grid`   | Electrical Grid Stability Simulated         | Binary (`stabf`)                             |
| `ads`    | Internet Advertisements                     | Binary (`class`)                             |
| `magic`  | MAGIC Gamma Telescope                       | Binary (`target`)                            |
| `boone`  | MiniBooNE Particle Identification           | Binary (`signal`)                            |
| `mush`   | Mushroom                                    | Binary (`result`)                            |
| `music`  | Geographical Original of Music              | Binary (`att117` binned)                     |
| `news`   | News Popularity                             | 4-class (`shares` binned)                    |
| `nurse`  | Nursery                                     | 3-class                                      |
| `occup`  | Occupancy Detection                         | Binary (`Occupancy`)                         |
| `spam`   | Spambase                                    | Binary (`spam`)                              |
| `cond`   | Superconductivity                           | 4-class (`critical_temp` binned)             |
| `turk`   | Turkiye Student Evaluation                  | Multi-class (`instr`)                        |
| `wine`   | Wine Quality (red + white)                  | 4-class (`quality` binned)                   |
| `yeast`  | Yeast                                       | 4-class (grouped)                            |

!!! info 
    Some loaders (`ccard`, `dota`, `boone`, `avila`, `music`, `bike`, `blog`, `ener`, `news`, `cond`) download `.zip`/`.xls` archives on demand and may take a while on first use, since no local cache directory is used by these UCI loaders (unlike `load_forman`, which caches under `~/.quack_data/`).

## Combining with a quantifier

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from quack.datasets import load_uci
from quack.quantifiers import ACC
from quack.metrics import ae
import numpy as np

X, y = load_uci("magic")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

quantifier = ACC(classifier=LogisticRegression(max_iter=1000))
quantifier.fit(X_train, y_train)

labels, counts = np.unique(y_test, return_counts=True)
true_prev = counts / y_test.shape[0]
pred_prev = quantifier.predict(X_test)

print(f"AE: {ae(true_prev, pred_prev):.4f}")
```

## References

[1] Schumacher, T., Strohmaier, M., & Lemmerich, F. (2025). *A comparative evaluation of quantification methods*. Journal of Machine Learning Research, 26(55), 1-54.
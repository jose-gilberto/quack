# Multi-label, Hierarchical and Regression Datasets

Besides the single-label benchmarks of `load_uci` and `load_forman`, `quack` loads datasets for three more complex quantification scenarios:

| Scenario | Function | Returns | Keys |
|----------|----------|---------|------|
| Multi-label quantification | `load_multilabel` | `X, Y` | `MULTILABEL_DATASETS` |
| Hierarchical quantification | `load_hierarchical` | `X, Y, hierarchy` | `HIERARCHICAL_DATASETS` |
| Quantification for regression / ordinal quantification | `load_regression` | `X, y` | `REGRESSION_DATASETS` |

The loaders follow the same design as the UCI ones: each dataset is a loader class (subclass of `BaseMultilabelDatasetLoader`, `BaseHierarchicalDatasetLoader` or `BaseRegressionDatasetLoader`), registered under a short key in a factory (`MultilabelLoaderFactory`, `HierarchicalLoaderFactory`, `RegressionLoaderFactory`) and exposed through one `load_*` function. Unlike the UCI loaders, the files are cached under `~/.quack_data/<dataset>/` (or `data_home`) and verified against a checksum when one is known, and every loader records the license of its data.

## Multi-label datasets

```python
from quack.datasets import load_multilabel

X, Y, meta = load_multilabel("goemotions", return_metadata=True)

print(X.shape, Y.shape)          # (54263,) (54263, 28)
print(meta.label_names[:3])      # ['admiration', 'amusement', 'anger']
print(Y.mean(axis=0).round(3))   # prevalence of each label
```

`Y` is a binary indicator matrix (`int8`) with one column per label. Text datasets return `X` as an object array of documents, so that any vectorizer can be used; the others return `float32` features. `meta.split` holds the official split when the source has one:

```python
train = meta.split == "train"
X_train, Y_train = X[train], Y[train]
```

Some loaders take options, passed as keyword arguments:

```python
# ToLD-Br: 3 annotators per tweet; a label is positive with at least `min_votes` votes
X, Y, meta = load_multilabel("told-br", min_votes=2, return_metadata=True)
votes = meta.extras["votes"]     # raw vote counts, for soft labels

# Reuters-21578: keep the topics with at least 5 training documents, as in [1]
X, Y = load_multilabel("reuters21578", accept_terms=True, min_train_positives=5)
```

## Hierarchical datasets

```python
from quack.datasets import load_hierarchical

X, Y, hierarchy = load_hierarchical("20newsgroups")

print(hierarchy)                 # Hierarchy(tree, n=27, top=7, leaves=20, depth=2)
print(hierarchy.level(1))        # ['alt', 'comp', 'misc', 'rec', 'sci', 'soc', 'talk']
print(hierarchy.children("sci"))
```

`Y` has one column per class of the taxonomy, in `hierarchy.nodes` order, and is closed upward: an instance labelled `sci.space` is also labelled `sci`. Prevalences at any level are column means:

```python
top = [hierarchy.index[c] for c in hierarchy.level(1)]
print(Y[:, top].mean(axis=0))
```

`hierarchy.constraint(Y)` tells which consistency constraint holds between levels. Single-path trees (20 Newsgroups, DBPedia, Amazon, Web of Science) give `"equality"`: the prevalence of a class is the sum of its children's, and `hierarchy.sum_matrix()` maps leaf prevalences to all levels. Multi-label taxonomies (GoEmotions, RCV1, MultiEURLEX, arXiv, Clus-HMC) give `"inequality"`: a child's prevalence never exceeds its parent's. `hierarchy.violations(p)` counts the parent/child pairs that an estimate `p` gets wrong.

`Hierarchy` can also be built directly, for instance for a taxonomy of your own:

```python
from quack.datasets import Hierarchy

h = Hierarchy.from_paths([("clothing", "shirt"), ("clothing", "trousers"), ("shoes",)])
Y = h.label_matrix([["clothing/shirt"], ["shoes"]])
```

## Regression datasets

```python
from quack.datasets import load_regression, make_ordinal

X, y = load_regression("fried")
classes, keep, edges = make_ordinal(y, n_bins=10, min_count=1000)
X_ordinal = X[keep]
```

`y` is the continuous target (`float64`). `make_ordinal` follows the protocol of [2] to turn a regression dataset into an ordinal quantification one: equal-width bins with at least `min_count` items each, dropping the items outside the bins.

## Natural bags

Some datasets come with a natural grouping of their instances: the year of a law (MultiEURLEX) or paper (arXiv), the month of a weather record (Szeged), the state of a person (ACS Income). It is returned in `meta.groups`, and each group is a bag with a naturally shifted distribution, a complement to the artificial bags of `quack.bag_generator`:

```python
import numpy as np
from quack.datasets import load_regression

X, y, meta = load_regression("acs-income", states=["CA", "TX", "NY"], return_metadata=True)

for state in np.unique(meta.groups):
  bag = meta.groups == state
  print(state, bag.sum(), np.median(y[bag]))
```

## Licenses

Every loader declares the license of its data, as read on the official page (`loader.license`, or `meta.license`). The table of all of them is available as a data frame:

```python
from quack.datasets import get_dataset_licenses

table = get_dataset_licenses()
print(table[["task", "spdx", "redistribution", "access", "requires_acceptance"]])
```

Datasets whose terms limit their use (research only, restricted, or a site that requires an account) raise `TermsNotAcceptedError` with a summary of the terms until `accept_terms=True` is passed:

```python
load_multilabel("reuters21578")                     # TermsNotAcceptedError
load_multilabel("reuters21578", accept_terms=True)  # ok
```

| Key | Dataset | License | Redistribution | Access |
|-----|---------|---------|----------------|--------|
| `told-br` | ToLD-Br | CC BY-SA 4.0 | allowed (share-alike) | open |
| `goemotions` | GoEmotions (multi-label and hierarchical) | Apache-2.0 | allowed | open |
| `reuters21578`, `reuters21578-modapte` | Reuters-21578 | Reuters terms | research only | open / manual |
| `jigsaw-toxic` | Jigsaw Toxic Comments | CC BY-SA 3.0 | allowed (share-alike) | Kaggle login |
| `emotions`, `scene`, `yeast`, ... | Mulan collection (12 datasets) | not declared | download from the source | manual |
| `20newsgroups` | 20 Newsgroups | not declared | download from the source | open |
| `rcv1` | RCV1-v2 | Reuters/NIST agreements | restricted | open |
| `dbpedia` | DBPedia Classes | CC BY-SA 3.0 | allowed (share-alike) | open |
| `amazon-hier` | Amazon reviews (3 levels) | CC BY 1.0 | allowed | open |
| `wos` | Web of Science | CC BY 4.0 | allowed | open |
| `multieurlex` | MultiEURLEX | CC BY-SA 4.0 | allowed (share-alike) | open |
| `arxiv` | arXiv metadata | CC0 | allowed | Kaggle login |
| `<name>-fun`, `<name>-go` | Clus-HMC yeast datasets (12 x 2) | not declared | download from the source | manual |
| `blog` | BlogFeedback | CC BY 4.0 | allowed | open |
| `news` | Online News Popularity | not confirmed | download from the source | open |
| `yolanda`, `fried`, `szeged` | OpenML copies | not confirmed | download from the source | open |
| `acs-income` | ACS PUMS (US Census) | US government work | allowed | open |

!!! warning
    The license column records what the sources state, checked on the date in `license.checked`. It is not legal advice; read the terms (`license.source`) before publishing data or models derived from these datasets.

## Manual downloads

Datasets whose source declares no license, or that sit behind a login, are not downloaded automatically. The loader raises `FileNotFoundError` with instructions until the files are in place:

| Key | Get | Place in |
|-----|-----|----------|
| `emotions`, `scene`, ... | ARFF + XML files from the [Mulan page](https://mulan.sourceforge.net/datasets-mlc.html) | `~/.quack_data/mulan/<name>/` (`<name>-train.arff`, `<name>-test.arff`, `<name>.xml`) |
| `<name>-fun`, `<name>-go` | FunCat and GO archives from the [Clus-HMC page](https://dtai.cs.kuleuven.be/clus/hmcdatasets) | `~/.quack_data/clus_hmc/` (zips or extracted `.arff` files, any subfolder) |
| `reuters21578-modapte` | `reuters21578.tar.gz` from the [UCI KDD archive](https://kdd.ics.uci.edu/databases/reuters21578/) | `~/.quack_data/reuters21578/` |
| `jigsaw-toxic` | `kaggle competitions download -c jigsaw-toxic-comment-classification-challenge` | `~/.quack_data/jigsaw_toxic/` |
| `arxiv` | `arxiv-metadata-oai-snapshot.json` from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) | `~/.quack_data/arxiv/` |

## Adding a dataset

A new dataset is a loader class registered in the factory of its scenario. For a multi-label dataset:

```python
import pandas as pd
from quack.datasets._licenses import DatasetLicense
from quack.datasets.utils import BaseMultilabelDatasetLoader


class MyLoader(BaseMultilabelDatasetLoader):
  dataset_name = "My dataset"
  cache_dir = "my_dataset"
  is_text = True
  labels = ["a", "b", "c"]
  license = DatasetLicense(
    spdx="CC-BY-4.0",
    redistribution="allowed",
    source="https://example.org/my-dataset",
    checked="2026-10-28",
    citation="Author, A. My dataset. 2026.",
  )

  def _download_and_load(self) -> pd.DataFrame:
    path = self._fetch("https://example.org/my-dataset.csv", "my-dataset.csv", checksum="sha256:...")
    return pd.read_csv(path)

  def _preprocess(self, raw_data):
    return raw_data

  def _split_features_and_target(self, data):
    return data["text"].tolist(), data[self.labels].to_numpy()

  def _label_names(self, data):
    return list(self.labels)
```

Then add it to `MultilabelLoaderFactory._registry` and write a test that serves the file through a monkeypatched `download_file`, as in `tests/test_multilabel_datasets.py`.

## References

[1] Moreo, A., Francisco, M., & Sebastiani, F. (2023). *Multi-Label Quantification*. ACM Transactions on Knowledge Discovery from Data, 18(1).

[2] Bunse, M., Moreo, A., Sebastiani, F., & Senz, M. (2024). *Regularization-based methods for ordinal quantification*. Data Mining and Knowledge Discovery, 38, 4076-4121.

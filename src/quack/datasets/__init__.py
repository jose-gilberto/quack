from quack.datasets._forman_datasets import load_forman, FORMAN_DATASETS
from quack.datasets._uci_datasets import load_uci, UCI_DATASETS
from quack.datasets._multilabel_datasets import load_multilabel, MULTILABEL_DATASETS, MultilabelLoaderFactory
from quack.datasets._hierarchical_datasets import load_hierarchical, HIERARCHICAL_DATASETS, HierarchicalLoaderFactory
from quack.datasets._regression_datasets import load_regression, REGRESSION_DATASETS, RegressionLoaderFactory
from quack.datasets._hierarchy import Hierarchy, ROOT
from quack.datasets._licenses import DatasetLicense, TermsNotAcceptedError, REDISTRIBUTION, ACCESS
from quack.datasets.utils import DatasetMetadata, make_ordinal


def get_dataset_licenses():
  """License terms of every multi-label, hierarchical and regression dataset.

  Returns
  -------
  licenses : pandas.DataFrame
    One row per dataset key, with its task, SPDX identifier, redistribution
    category, access mode, whether `accept_terms=True` is required, whether
    a processed copy may be published, the page where the license was read
    and the date of that reading.

  Examples
  --------
  >>> from quack.datasets import get_dataset_licenses
  >>> table = get_dataset_licenses()
  >>> table[table.requires_acceptance].index.tolist()  # doctest: +SKIP
  """
  import pandas as pd

  rows = []
  for task, factory in (("multilabel", MultilabelLoaderFactory),
                        ("hierarchical", HierarchicalLoaderFactory),
                        ("regression", RegressionLoaderFactory)):
    for key, loader in factory._registry.items():
      lic = loader.license
      rows.append({
        "dataset": key,
        "task": task,
        "name": loader.dataset_name,
        "spdx": lic.spdx,
        "redistribution": lic.redistribution,
        "access": lic.access,
        "requires_acceptance": lic.requires_acceptance,
        "allows_mirror": lic.allows_mirror,
        "source": lic.source,
        "checked": lic.checked,
      })
  return pd.DataFrame(rows).set_index("dataset")


__all__ = [
    'load_forman',
    'FORMAN_DATASETS',
    'load_uci',
    'UCI_DATASETS',
    'load_multilabel',
    'MULTILABEL_DATASETS',
    'load_hierarchical',
    'HIERARCHICAL_DATASETS',
    'load_regression',
    'REGRESSION_DATASETS',
    'Hierarchy',
    'ROOT',
    'DatasetLicense',
    'DatasetMetadata',
    'TermsNotAcceptedError',
    'REDISTRIBUTION',
    'ACCESS',
    'make_ordinal',
    'get_dataset_licenses',
]

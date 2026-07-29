# tests/test_uci_datasets.py
import numpy as np
import pandas as pd
import pytest

from quack.datasets._uci_datasets import UCILoaderFactory, UCI_DATASETS, load_uci
from quack.datasets.utils import BaseDatasetLoader


class DummyLoader(BaseDatasetLoader):
  """Minimal concrete loader used to test the template method pipeline
  without touching the network."""

  def _download_and_load(self) -> pd.DataFrame:
    return pd.DataFrame({
      "f1": [1.0, 2.0, 3.0, 4.0],
      "f2": [10, 20, 30, 40],
      "target": [0, 1, 0, 1],
    })

  def _preprocess(self, raw_data: pd.DataFrame) -> pd.DataFrame:
    return raw_data

  def _split_features_and_target(self, data: pd.DataFrame):
    X = data.drop(columns=["target"]).values
    y = data["target"].values
    return X, y


class TestUCILoaderFactory:
  def test_uci_datasets_matches_registry_keys(self):
    assert set(UCI_DATASETS) == set(UCILoaderFactory._registry.keys())

  def test_get_loader_returns_instance_for_valid_key(self):
    loader = UCILoaderFactory.get_loader("bc-cont")
    assert isinstance(loader, BaseDatasetLoader)
    assert loader.__class__.__name__ == "BreastCancerContLoader"

  def test_get_loader_is_case_insensitive(self):
    loader_lower = UCILoaderFactory.get_loader("bc-cont")
    loader_upper = UCILoaderFactory.get_loader("BC-CONT")
    assert type(loader_lower) is type(loader_upper)

  def test_get_loader_raises_for_invalid_key(self):
    with pytest.raises(ValueError, match="not supported"):
      UCILoaderFactory.get_loader("not-a-real-dataset")


class TestBaseDatasetLoaderPipeline:
  def test_load_dataset_returns_correct_dtypes_and_shapes(self):
    X, y = DummyLoader().load_dataset()

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.dtype == np.float32
    assert X.shape == (4, 2)
    assert y.shape == (4,)
    np.testing.assert_array_equal(y, [0, 1, 0, 1])


class TestLoadUCI:
  def test_load_uci_delegates_to_factory(self, monkeypatch):
    dummy = DummyLoader()
    monkeypatch.setattr(UCILoaderFactory, "get_loader", classmethod(lambda cls, name: dummy))

    X, y = load_uci("anything")

    assert X.shape == (4, 2)
    assert y.shape == (4,)
    assert X.dtype == np.float32

  def test_load_uci_raises_for_invalid_dataset(self):
    with pytest.raises(ValueError, match="not supported"):
      load_uci("not-a-real-dataset")

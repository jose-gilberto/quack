# tests/test_regression_datasets.py
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.utils import Bunch

from quack.datasets import REGRESSION_DATASETS, load_regression, make_ordinal
from quack.datasets import utils as dataset_utils
from quack.datasets._regression_datasets import ACSIncomeLoader, RegressionLoaderFactory, SzegedWeatherLoader


def zip_bytes(members: dict) -> bytes:
  buffer = io.BytesIO()
  with zipfile.ZipFile(buffer, "w") as zf:
    for name, content in members.items():
      zf.writestr(name, content)
  return buffer.getvalue()


class FakeDownloads:
  """Replaces `download_file` with a function that writes local fixtures."""

  def __init__(self):
    self.files: dict[str, bytes] = {}
    self.calls: list[str] = []

  def __call__(self, url, filename, exist_ok=False, checksum=None):
    self.calls.append(url)
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(self.files[path.name])


@pytest.fixture
def downloads(monkeypatch):
  fake = FakeDownloads()
  monkeypatch.setattr(dataset_utils, "download_file", fake)
  return fake


@pytest.fixture
def openml(monkeypatch):
  """Replaces `sklearn.datasets.fetch_openml` and records the requested ids."""
  import sklearn.datasets

  served, calls = {}, []

  def fake_fetch(data_id, data_home, as_frame, parser):
    calls.append((data_id, data_home))
    frame, target = served[data_id]
    return Bunch(frame=frame, target_names=[target] if target else [])

  monkeypatch.setattr(sklearn.datasets, "fetch_openml", fake_fetch)
  return served, calls


class TestRegressionLoaderFactory:
  def test_regression_datasets_matches_registry_keys(self):
    assert REGRESSION_DATASETS == list(RegressionLoaderFactory._registry.keys())

  def test_get_loader_is_case_insensitive_and_passes_options(self, tmp_path):
    loader = RegressionLoaderFactory.get_loader("ACS-Income", data_home=tmp_path, states=["tx", "ca"], year=2019)
    assert isinstance(loader, ACSIncomeLoader)
    assert loader.states == ["TX", "CA"] and loader.year == 2019

  def test_get_loader_raises_for_invalid_key(self):
    with pytest.raises(ValueError, match="not supported"):
      RegressionLoaderFactory.get_loader("not-a-real-dataset")


class TestUCIRegression:
  def test_blog_feedback_keeps_continuous_target(self, tmp_path, downloads):
    rows = []
    for i in range(3):
      values = [float(i)] * 281
      values[5] = 7.0  # constant column, dropped
      values[280] = [0.0, 3.0, 250.0][i]
      rows.append(",".join(str(v) for v in values))
    downloads.files["BlogFeedback.zip"] = zip_bytes({"blogData_test-2012.03.31.01_00.csv": rows[0],
                                                     "blogData_train.csv": "\n".join(rows)})
    X, y = load_regression("blog", data_home=tmp_path)
    np.testing.assert_array_equal(y, [0.0, 3.0, 250.0])
    assert y.dtype == np.float64 and X.dtype == np.float32
    assert X.shape == (3, 279)

  def test_online_news_drops_url_and_timedelta(self, tmp_path, downloads):
    csv = ("url, timedelta, n_tokens_title, shares\n"
           "http://a, 731.0, 12.0, 593\n"
           "http://b, 731.0, 9.0, 1200\n")
    downloads.files["OnlineNewsPopularity.zip"] = zip_bytes({
      "OnlineNewsPopularity/OnlineNewsPopularity.names": "description",
      "OnlineNewsPopularity/OnlineNewsPopularity.csv": csv})
    X, y = load_regression("news", data_home=tmp_path)
    np.testing.assert_array_equal(X, [[12.0], [9.0]])
    np.testing.assert_array_equal(y, [593, 1200])


class TestOpenMLRegression:
  def test_fried_uses_openml_id_and_default_target(self, tmp_path, openml):
    served, calls = openml
    served[564] = (pd.DataFrame({"x1": [0.1, 0.2], "x2": [0.3, 0.4], "y": [1.5, 2.5]}), "y")
    X, y = load_regression("fried", data_home=tmp_path)
    np.testing.assert_array_equal(y, [1.5, 2.5])
    assert X.shape == (2, 2)
    assert calls == [(564, str(tmp_path / "openml"))]

  def test_categorical_features_are_one_hot_encoded(self, tmp_path, openml):
    served, _ = openml
    served[42705] = (pd.DataFrame({"a": ["u", "v", "u"], "b": [1.0, 2.0, 3.0], "t": [0.0, 1.0, 2.0]}), "t")
    X, _ = load_regression("yolanda", data_home=tmp_path)
    assert X.shape == (3, 2)

  def test_dataset_without_default_target_raises(self, tmp_path, openml):
    served, _ = openml
    served[564] = (pd.DataFrame({"x1": [0.1]}), None)
    with pytest.raises(ValueError, match="no default target"):
      load_regression("fried", data_home=tmp_path)


class TestSzegedWeather:
  FRAME = pd.DataFrame({
    "Formatted Date": ["2006-04-01 00:00:00.000 +0200", "2006-04-01 01:00:00.000 +0200",
                       "2007-11-15 00:00:00.000 +0100"],
    "Summary": ["Clear", "Cloudy", "Foggy"],
    "Temperature (C)": [9.5, 9.4, 1.0],
    "Apparent Temperature (C)": [7.4, 7.2, -2.0],
    "Humidity": [0.89, 0.86, 0.95],
  })

  def test_temperature_target_without_leak_and_monthly_bags(self, tmp_path, openml):
    served, _ = openml
    served[43713] = (self.FRAME.copy(), None)
    X, y, meta = load_regression("szeged", data_home=tmp_path, return_metadata=True)
    np.testing.assert_allclose(y, [9.5, 9.4, 1.0])
    np.testing.assert_allclose(X.ravel(), [0.89, 0.86, 0.95], rtol=1e-6)  # only humidity is left
    np.testing.assert_array_equal(meta.groups, ["2006-04", "2006-04", "2007-11"])

  def test_yearly_bags(self, tmp_path, openml):
    served, _ = openml
    served[43713] = (self.FRAME.copy(), None)
    _, _, meta = load_regression("szeged", data_home=tmp_path, return_metadata=True, bag="year")
    np.testing.assert_array_equal(meta.groups, ["2006", "2006", "2007"])

  def test_rejects_unknown_bag(self, tmp_path):
    with pytest.raises(ValueError, match="bag"):
      SzegedWeatherLoader(data_home=tmp_path, bag="week")


class TestACSIncome:
  CSV = ("RT,ST,AGEP,COW,SCHL,MAR,OCCP,POBP,RELP,WKHP,SEX,RAC1P,PINCP,PWGTP,OTHER\n"
         "P,6,35,1,21,1,1010,6,0,40,1,1,85000,20,9\n"
         "P,6,15,1,10,5,,6,2,10,2,1,2000,15,9\n"       # too young
         "P,6,40,1,19,1,4720,36,1,0,2,6,30000,12,9\n"  # no hours worked
         "P,6,52,3,22,3,2205,6,0,50,2,2,120000,30,9\n")

  def test_selection_target_groups_and_weights(self, tmp_path, downloads):
    downloads.files["csv_pca.zip"] = zip_bytes({"psam_p06.csv": self.CSV, "ACS2018_PUMS_README.pdf": "x"})
    X, y, meta = load_regression("acs-income", data_home=tmp_path, return_metadata=True)
    np.testing.assert_array_equal(y, [85000, 120000])
    assert X.shape == (2, len(ACSIncomeLoader.features))
    np.testing.assert_array_equal(meta.groups, [6, 6])
    np.testing.assert_array_equal(meta.extras["weights"], [20, 30])
    assert downloads.calls == [
      "https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pca.zip"]

  def test_several_states(self, tmp_path, downloads):
    texas = self.CSV.replace("P,6,", "P,48,")
    downloads.files["csv_pca.zip"] = zip_bytes({"psam_p06.csv": self.CSV})
    downloads.files["csv_ptx.zip"] = zip_bytes({"psam_p48.csv": texas})
    _, y, meta = load_regression("acs-income", data_home=tmp_path, return_metadata=True, states=["CA", "TX"])
    assert len(y) == 4
    np.testing.assert_array_equal(meta.groups, [6, 6, 48, 48])

  def test_rejects_years_before_2014(self, tmp_path):
    with pytest.raises(ValueError, match="2014"):
      ACSIncomeLoader(data_home=tmp_path, year=2010)


class TestMakeOrdinal:
  def test_equal_width_bins_with_minimum_count(self):
    rng = np.random.default_rng(0)
    y = rng.normal(size=20000)
    classes, keep, edges = make_ordinal(y, n_bins=5, min_count=500)
    assert len(edges) == 6
    np.testing.assert_allclose(np.diff(edges), np.diff(edges)[0])
    counts = np.bincount(classes, minlength=5)
    assert counts.min() >= 500
    assert len(classes) == keep.sum()
    assert np.all(y[keep] >= edges[0]) and np.all(y[keep] <= edges[-1])

  def test_uniform_target_keeps_almost_everything(self):
    y = np.linspace(0, 1, 10000)
    classes, keep, _ = make_ordinal(y, n_bins=10, min_count=900)
    assert keep.mean() > 0.95
    assert set(classes) == set(range(10))

  def test_ignores_missing_values(self):
    y = np.r_[np.linspace(0, 1, 5000), [np.nan] * 10]
    _, keep, _ = make_ordinal(y, n_bins=5, min_count=500)
    assert not keep[-10:].any()

  def test_too_few_values_raises(self):
    with pytest.raises(ValueError, match="need at least"):
      make_ordinal(np.arange(100.0), n_bins=10, min_count=20)

  def test_impossible_minimum_raises(self):
    y = np.r_[np.zeros(5000), np.ones(5000), 1000.0]
    with pytest.raises(ValueError, match="no range"):
      make_ordinal(y, n_bins=10, min_count=900)


class TestLoadRegression:
  def test_raises_for_invalid_dataset(self):
    with pytest.raises(ValueError, match="not supported"):
      load_regression("not-a-real-dataset")

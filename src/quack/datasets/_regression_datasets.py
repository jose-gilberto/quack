import re
import zipfile
from pathlib import Path
from typing import Sequence

import pandas as pd

from quack.datasets._licenses import DatasetLicense, check_terms
from quack.datasets.utils import BaseRegressionDatasetLoader

class BlogFeedbackRegressionLoader(BaseRegressionDatasetLoader):
  """BlogFeedback with its continuous target: comments a post receives in the next 24 hours.

  Same source as `load_uci("blog")`, which bins the target into 4 classes.
  Reads the training file (`blogData_train.csv`, 52,397 posts) and drops the
  constant columns. One of the four regression datasets that [1] turns into ordinal
  quantification problems (see `make_ordinal`).

  References
  ----------
  [1] Bunse, M., Moreo, A., Sebastiani, F., & Senz, M. (2024). Regularization-based
      methods for ordinal quantification. Data Mining and Knowledge Discovery, 38.
  """
  dataset_name = "BlogFeedback (regression)"
  cache_dir = "blog_feedback"
  target = "att280"
  license = DatasetLicense(
    spdx="CC-BY-4.0",
    redistribution="allowed",
    source="https://archive.ics.uci.edu/dataset/304/blogfeedback",
    checked="2026-10-27",
    citation="Buza, K. Feedback Prediction for Blogs. Data Analysis, Machine Learning and Knowledge Discovery, 2014.",
    conditions=("attribution",),
  )

  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00304/BlogFeedback.zip"

  def _download_and_load(self) -> pd.DataFrame:
    path = self._fetch(self.url, "BlogFeedback.zip")
    with zipfile.ZipFile(path) as zf:
      names = zf.namelist()
      member = next((m for m in names if m.endswith("blogData_train.csv")), names[-1])
      with zf.open(member) as fh:
        return pd.read_csv(fh, header=None, names=[f"att{i}" for i in range(281)], skipinitialspace=True)

  def _preprocess(self, raw_data: pd.DataFrame) -> pd.DataFrame:
    return raw_data.drop(columns=raw_data.columns[raw_data.nunique() == 1])

  def _split_features_and_target(self, data):
    return data.drop(columns=[self.target]).values, data[self.target].values


class OnlineNewsRegressionLoader(BaseRegressionDatasetLoader):
  """Online News Popularity with its continuous target: number of shares.

  Same source as `load_uci("news")`, which bins the target into 4 classes.
  The article URL and the days between publication and collection
  (`timedelta`) are dropped as non-predictive, as the dataset page advises.
  One of the four regression datasets used in [1].

  References
  ----------
  [1] Bunse, M., Moreo, A., Sebastiani, F., & Senz, M. (2024). Regularization-based
      methods for ordinal quantification. Data Mining and Knowledge Discovery, 38.
  """
  dataset_name = "Online News Popularity (regression)"
  cache_dir = "online_news"
  target = "shares"
  license = DatasetLicense(
    spdx="NOASSERTION",
    redistribution="not-declared",
    source="https://archive.ics.uci.edu/dataset/332/online+news+popularity",
    checked="2026-10-27",
    citation=("Fernandes, K., Vinagre, P., & Cortez, P. A Proactive Intelligent Decision Support System "
              "for Predicting the Popularity of Online News. EPIA 2015."),
    conditions=("cite",),
    notes="UCI lists most datasets under CC BY 4.0, but this one was not confirmed on its page.",
  )

  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip"

  def _download_and_load(self) -> pd.DataFrame:
    path = self._fetch(self.url, "OnlineNewsPopularity.zip")
    with zipfile.ZipFile(path) as zf:
      member = next(m for m in zf.namelist() if m.endswith(".csv"))
      with zf.open(member) as fh:
        return pd.read_csv(fh, skipinitialspace=True)

  def _preprocess(self, raw_data: pd.DataFrame) -> pd.DataFrame:
    data = raw_data.rename(columns=lambda c: c.strip())
    return data.drop(columns=[c for c in ("url", "timedelta") if c in data])

  def _split_features_and_target(self, data):
    return data.drop(columns=[self.target]).values, data[self.target].values


class _OpenMLRegressionLoader(BaseRegressionDatasetLoader):
  """Base for regression datasets fetched with `sklearn.datasets.fetch_openml`."""
  openml_id: int = 0

  def _download_and_load(self) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=self.openml_id, data_home=str(self.dataset_dir), as_frame=True, parser="auto")
    self._target = bunch.target_names[0] if bunch.target_names else None
    return bunch.frame.copy()

  def _preprocess(self, raw_data: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(raw_data, drop_first=True)

  def _split_features_and_target(self, data):
    if self._target is None:
      raise ValueError(f"OpenML dataset {self.openml_id} declares no default target")
    return data.drop(columns=[self._target]).values, data[self._target].values


class YolandaLoader(_OpenMLRegressionLoader):
  """yolanda (OpenML 42705): 400,000 instances, 100 numeric features.

  One of the four regression datasets that [1] turns into ordinal
  quantification problems (see `make_ordinal`).

  References
  ----------
  [1] Bunse, M., Moreo, A., Sebastiani, F., & Senz, M. (2024). Regularization-based
      methods for ordinal quantification. Data Mining and Knowledge Discovery, 38.
  """
  dataset_name = "yolanda"
  cache_dir = "openml"
  openml_id = 42705
  license = DatasetLicense(
    spdx="NOASSERTION",
    redistribution="not-declared",
    source="https://www.openml.org/d/42705",
    checked="2026-10-27",
    citation="Guyon, I. et al. Analysis of the AutoML Challenge series 2015-2018. AutoML, Springer, 2019.",
    conditions=("cite",),
    notes="OpenML licence field not confirmed; check `fetch_openml(data_id=42705).details['licence']`.",
  )


class FriedLoader(_OpenMLRegressionLoader):
  """fried (OpenML 564): Friedman's synthetic regression benchmark, 40,768 instances.

  One of the four regression datasets that [1] turns into ordinal
  quantification problems (see `make_ordinal`).

  References
  ----------
  [1] Bunse, M., Moreo, A., Sebastiani, F., & Senz, M. (2024). Regularization-based
      methods for ordinal quantification. Data Mining and Knowledge Discovery, 38.
  """
  dataset_name = "fried"
  cache_dir = "openml"
  openml_id = 564
  license = DatasetLicense(
    spdx="NOASSERTION",
    redistribution="not-declared",
    source="https://www.openml.org/d/564",
    checked="2026-10-27",
    citation="Friedman, J. H. Multivariate Adaptive Regression Splines. The Annals of Statistics 19(1), 1991.",
    conditions=("cite",),
    notes="OpenML licence field not confirmed; check `fetch_openml(data_id=564).details['licence']`.",
  )


class SzegedWeatherLoader(_OpenMLRegressionLoader):
  """Hourly weather in Szeged, 2006-2016, with temperature as the target.

  Used in [1] for continuous target shift: November and December are colder
  than the rest of the year. The natural group of each record is its
  year-month (`bag="month"`) or year (`bag="year"`). The apparent temperature
  is dropped to avoid target leakage, and text columns are dropped.

  Parameters
  ----------
  bag : {"month", "year"}, default = "month"
    Granularity of the natural groups.

  References
  ----------
  [1] Kim, H., Zhang, X., Zhao, J., & Tian, Q. (2024). ReTaSA: A Nonparametric
      Functional Estimation Approach for Addressing Continuous Target Shift. ICLR 2024.
  """
  dataset_name = "Szeged weather"
  cache_dir = "openml"
  openml_id = 43713
  license = DatasetLicense(
    spdx="NOASSERTION",
    redistribution="not-declared",
    source="https://www.openml.org/d/43713",
    checked="2026-10-27",
    citation="Budincsevity, N. Weather in Szeged 2006-2016. Kaggle.",
    conditions=("cite",),
    notes="OpenML copy of a Kaggle dataset; the license of the original was not confirmed.",
  )

  def __init__(self, data_home=None, force_load: bool = False, bag: str = "month"):
    super().__init__(data_home, force_load)
    if bag not in {"month", "year"}:
      raise ValueError("bag must be 'month' or 'year'")
    self.bag = bag

  def _preprocess(self, raw_data: pd.DataFrame) -> pd.DataFrame:
    keys = {c: re.sub(r"[^a-z]", "", c.lower()) for c in raw_data.columns}
    temperature = next(c for c, k in keys.items() if k.startswith("temperature"))
    date = next(c for c, k in keys.items() if "date" in k)
    leaks = [c for c, k in keys.items() if k.startswith("apparenttemperature")]
    when = pd.to_datetime(raw_data[date].astype(str).str.slice(0, 19), errors="coerce")
    self._bags = when.dt.strftime("%Y-%m" if self.bag == "month" else "%Y").to_numpy()
    self._target = temperature
    numeric = raw_data.drop(columns=[date, *leaks]).select_dtypes("number")
    return numeric

  def _groups(self, data):
    return self._bags


class ACSIncomeLoader(BaseRegressionDatasetLoader):
  """ACS PUMS person records with the ACSIncome features and income as target.

  Downloads the American Community Survey microdata of the chosen states from
  the US Census Bureau and applies the ACSIncome selection of [1] (age over
  16, income over 100, positive hours and weight), keeping total personal
  income (`PINCP`) as a continuous target. The state of each person is the
  natural group, so each state is a bag with its own income distribution.
  `OCCP` (occupation code) is one of the features.

  Parameters
  ----------
  states : sequence of str, default = ("CA",)
    Two-letter state codes.
  year : int, default = 2018
    Survey year (2014 or later).
  horizon : {"1-Year", "5-Year"}, default = "1-Year"
    ACS sample.

  References
  ----------
  [1] Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021). Retiring Adult:
      New Datasets for Fair Machine Learning. NeurIPS 2021.
  """
  dataset_name = "ACS Income (PUMS)"
  cache_dir = "acs_pums"
  features = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]
  target = "PINCP"
  license = DatasetLicense(
    spdx="LicenseRef-US-Gov-Public-Domain",
    redistribution="allowed",
    source="https://catalog.data.gov/dataset/public-use-microdata-samples-pums",
    checked="2026-10-27",
    citation="Ding, F. et al. Retiring Adult: New Datasets for Fair Machine Learning. NeurIPS 2021.",
    conditions=("cite",),
    notes="US Census data are works of the US government (17 U.S.C. 105). De-identified microdata.",
  )

  def __init__(self, data_home=None, force_load: bool = False, states: Sequence[str] = ("CA",),
               year: int = 2018, horizon: str = "1-Year"):
    super().__init__(data_home, force_load)
    if int(year) < 2014:
      raise ValueError("year must be 2014 or later")
    self.states = [s.upper() for s in states]
    self.year = int(year)
    self.horizon = horizon

  def _read_state(self, state: str) -> pd.DataFrame:
    name = f"csv_p{state.lower()}.zip"
    url = f"https://www2.census.gov/programs-surveys/acs/data/pums/{self.year}/{self.horizon}/{name}"
    path = self._fetch(url, f"{self.year}/{self.horizon}/{name}")
    columns = set(self.features + [self.target, "PWGTP", "ST"])
    with zipfile.ZipFile(path) as zf:
      member = next(m for m in zf.namelist() if m.lower().endswith(".csv"))
      with zf.open(member) as fh:
        return pd.read_csv(fh, usecols=lambda c: c in columns)

  def _download_and_load(self) -> pd.DataFrame:
    return pd.concat([self._read_state(s) for s in self.states], ignore_index=True)

  def _preprocess(self, raw_data: pd.DataFrame) -> pd.DataFrame:
    data = raw_data[(raw_data["AGEP"] > 16) & (raw_data["PINCP"] > 100) &
                    (raw_data["WKHP"] > 0) & (raw_data["PWGTP"] >= 1)].reset_index(drop=True)
    self.extras_["weights"] = data["PWGTP"].to_numpy()
    return data

  def _split_features_and_target(self, data):
    return data[self.features].fillna(-1).values, data[self.target].values

  def _groups(self, data):
    return data["ST"].to_numpy()


class RegressionLoaderFactory:
  """Factory used to instantiate the correct regression loader by dataset key.

  Datasets with a continuous target, for quantification for regression
  (estimating the distribution or a summary of the target in a bag) and,
  through `make_ordinal`, for ordinal quantification as in [1].

  Refs
  [1] Bunse, M., Moreo, A., Sebastiani, F., & Senz, M. (2024). Regularization-based
      methods for ordinal quantification. Data Mining and Knowledge Discovery, 38.
  """
  _registry = {
    "blog": BlogFeedbackRegressionLoader,
    "news": OnlineNewsRegressionLoader,
    "yolanda": YolandaLoader,
    "fried": FriedLoader,
    "szeged": SzegedWeatherLoader,
    "acs-income": ACSIncomeLoader,
  }

  @classmethod
  def get_loader(cls, dataset_name: str, **kwargs) -> BaseRegressionDatasetLoader:
    pipeline_class = cls._registry.get(dataset_name.lower())
    if not pipeline_class:
      raise ValueError(f"Dataset '{dataset_name}' not supported on regression datasets.")
    return pipeline_class(**kwargs)


REGRESSION_DATASETS: list[str] = list(RegressionLoaderFactory._registry.keys())


def load_regression(dataset: str,
                    data_home: str | Path | None = None,
                    force_load: bool = False,
                    accept_terms: bool = False,
                    return_metadata: bool = False,
                    **kwargs) -> tuple:
  """Load a dataset with a continuous target by name using the registered loader factory.

  Parameters
  ----------
  dataset : str
    Dataset key. Must be one of `quack.datasets.REGRESSION_DATASETS`.
  data_home : str or Path, default = None
    Root folder of the cached files. Defaults to `~/.quack_data/`.
  force_load : bool, default = False
    Download the files again even if they are cached.
  accept_terms : bool, default = False
    Required for datasets whose license limits use.
  return_metadata : bool, default = False
    Also return a `DatasetMetadata` with natural groups and license.
  **kwargs
    Dataset-specific options, passed to the loader (e.g. `states` and `year`
    for `"acs-income"`, `bag` for `"szeged"`).

  Returns
  -------
  X : ndarray of shape (n_samples, n_features)
    Features as float32.
  y : ndarray of shape (n_samples,)
    Continuous target as float64.
  metadata : DatasetMetadata
    Only when `return_metadata=True`.

  Raises
  ------
  ValueError
    If `dataset` is not a registered key.
  TermsNotAcceptedError
    If the dataset has use conditions and `accept_terms` is False.

  Examples
  --------
  >>> from quack.datasets import load_regression, make_ordinal
  >>> X, y, meta = load_regression("acs-income", states=["CA", "TX"], return_metadata=True)
  >>> classes, keep, edges = make_ordinal(y, n_bins=10, min_count=1000)
  """
  loader = RegressionLoaderFactory.get_loader(dataset, data_home=data_home, force_load=force_load, **kwargs)
  check_terms(loader.license, accept_terms, dataset)
  X, y = loader.load_dataset()
  if return_metadata:
    return X, y, loader.metadata()
  return X, y

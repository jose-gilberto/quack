from io import BytesIO
import pandas as pd
import numpy as np
from zipfile import ZipFile
import urllib
from quack.datasets.utils import BaseDatasetLoader


class AdultLoader(BaseDatasetLoader):
  dataset_name = "Adult"
  col_names = ["Age", "Workclass", "Census_Weight", "Education", "Education_Numeric",
              "Marital_Status", "Occupation", "Relationship", "Race", "Sex", "Gain",
              "Loss", "Hours", "Country", "Income"]

  feat_cols = ["Age", "Workclass", "Education_Numeric", "Marital_Status", "Occupation",
               "Relationship", "Race", "Sex", "Gain", "Loss", "Hours", "Country", "Income"]
  target = "Income"

  def _download_and_load(self) -> pd.DataFrame:
    urls = ["https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"]

    train = pd.read_csv(urls[0], names=self.col_names, na_values=['?'], index_col=False, skipinitialspace=True)
    test = pd.read_csv(urls[1], names=self.col_names, na_values=['?'], index_col=False, skipinitialspace=True)
    
    return pd.concat([train, test], ignore_index=True)

  def _preprocess(self, raw_data: pd.DataFrame) -> pd.DataFrame:
    clean_data = raw_data.loc[:, self.feat_cols].dropna().reset_index(drop=True)
    clean_data["Age"] = pd.to_numeric(clean_data["Age"], errors='coerce')
    
    target_series = clean_data.pop(self.target).replace({
      "<=50K": 0, "<=50K.": 0, 
      ">50K.": 1, ">50K": 1
    }).astype(int)  
    
    clean_data = pd.get_dummies(clean_data, drop_first=True)
    clean_data[self.target] = target_series

    return clean_data

  def _split_features_and_target(self, data):
    X = data.loc[:, ~data.columns.isin([self.target])].values
    y = data[self.target].values

    return X, y
    

class AvilaLoader(BaseDatasetLoader):
  dataset_name = "Avila"
  target = "class"

  def _download_and_load(self) -> pd.DataFrame:
    url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip")
    zip_file = ZipFile(BytesIO(url.read()))
    
    train_file = zip_file.namelist()[1]
    test_file = zip_file.namelist()[2]
    
    col_names = ["feat" + str(i + 1) for i in range(10)] + ["class"]
    
    train = pd.read_csv(zip_file.open(train_file), names=col_names, skipinitialspace=True)
    test = pd.read_csv(zip_file.open(test_file), names=col_names, skipinitialspace=True)
    
    return pd.concat([train, test], ignore_index=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.copy()
    clean_data[self.target] = (clean_data[self.target] != "A").astype(int)
    # clean_data = raw_data.loc[clean_data["class"] != "A", "class"] = "B"
    # clean_data["class"] = clean_data["class"].replace({"A": 0, "B": 1})
    
    return clean_data
  
  def _split_features_and_target(self, data):
    X = data.drop(columns=[self.target]).values
    # data.loc[:, ~data.columns.isin(["class"])].values
    y = data[self.target].values
    return X, y
  

class BikeLoader(BaseDatasetLoader):
  dataset_name = "Bike"
  target = "cnt"
  
  def _download_and_load(self) -> pd.DataFrame:
    url = urllib.request.urlopen(
      "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip")
    zip_file = ZipFile(BytesIO(url.read()))
    filename = zip_file.namelist()[2]
    
    full = pd.read_csv(zip_file.open(filename), header=0, skipinitialspace=True)
    return full
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.drop(["instant", "casual", "registered", "dteday"], axis=1)

    clean_data = pd.get_dummies(clean_data, columns=["season", "yr", "mnth", "hr", "weekday", "weathersit"])
    
    bins = [0, 100, 200, 300, 1000]
    labels = [1, 2, 3, 4]
    clean_data['cnt'] = pd.cut(clean_data['cnt'], bins=bins, labels=labels)
    clean_data['cnt'] = clean_data['cnt'].astype("int64")

    return clean_data
  
  def _split_features_and_target(self, data):
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class BlogFeedbackLoader(BaseDatasetLoader):
  dataset_name = "BlogFeedback"
  target = "att280"

  def _download_and_load(self) -> pd.DataFrame:
    url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00304/BlogFeedback.zip")
    zip_file = ZipFile(BytesIO(url.read()))
    filename = zip_file.namelist()[-1]
    
    full = pd.read_csv(zip_file.open(filename),
                      header=None,
                      names=["att" + str(i) for i in range(281)],
                      skipinitialspace=True)
    return full
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    constcols = raw_data.columns[raw_data.nunique() == 1]
    clean_data = raw_data.drop(constcols, axis=1)
    
    bins = [-1, 0, 1, 10, 2000]
    labels = [0, 1, 2, 3]
    
    clean_data['att280'] = pd.cut(clean_data['att280'], bins=bins, labels=labels)
    clean_data['att280'] = clean_data['att280'].astype("int64")
    return clean_data

  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class BreastCancerContLoader(BaseDatasetLoader):
  dataset_name = "Breast Cancer Wisconsis (Original)"
  col_names = ["ID", "diagnosis", "C1_radius", "C1_texture", "C1_perimeter", "C1_area",
               "C1_smoothness", "C1_compactness", "C1_concavity", "C1_concave_points",
               "C1_symmetry", "C1_fractal_dimension", "C2_radius", "C2_texture", "C2_perimeter",
               "C2_area", "C2_smoothness", "C2_compactness", "C2_concavity", "C2_concave_points",
               "C2_symmetry", "C2_fractal_dimension", "C3_radius", "C3_texture", "C3_perimeter",
               "C3_area", "C3_smoothness", "C3_compactness", "C3_concavity", "C3_concave_points",
               "C3_symmetry", "C3_fractal_dimension"]
  
  target = "diagnosis"
  
  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    return pd.read_csv(url, names=self.col_names, index_col="ID", skipinitialspace=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.copy()
    clean_data["diagnosis"] = clean_data["diagnosis"].replace({"B": 0, "M": 1})
    return clean_data
  
  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class BreastCancerIntLoader(BaseDatasetLoader):
  dataset_name = "Breast Cancer Wisconsis (Diagnostic)"
  target = "Class"
  col_names = ["ID", "Clump_Thickness", "Uniformity_of_Cell_Size", "Uniformity_of_Cell_Shape",
              "Marginal_Adhesion", "Single_Epithelial_Cell_Size", "Bare_Nuclei", "Bland_Chromatin",
              "Normal_Nucleoli", "Mitoses", "Class"]
  
  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin" \
          ".data "

    return pd.read_csv(url, names=self.col_names, na_values=['?'], 
                       index_col="ID", skipinitialspace=True)

  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.dropna()
    clean_data.Class = clean_data.Class.replace({2: 0, 4: 1})
    return clean_data
  
  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class CarsLoader(BaseDatasetLoader):
  dataset_name = "Car Evaluation"
  target = "acc_class"
  
  col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "acc_class"]
  
  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    return pd.read_csv(url, names=self.col_names, index_col=False, skipinitialspace=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.copy()
    target_series= clean_data.pop(self.target).replace(
      {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}).astype(int)
        
    clean_data = pd.get_dummies(clean_data, drop_first=True)
    clean_data[self.target] = target_series

    return clean_data
  
  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class ConcreteLoader(BaseDatasetLoader):
  
  dataset_name = "Concrete Compressive Strength"
  target = "strength"
  
  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    col_names = ["comp" + str(i + 1) for i in range(8)] + ["strength"]
    
    return pd.read_excel(url, header=0, names=col_names, skipinitialspace=True)    
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.copy()
    bins = [0, 20, 35, 100]
    labels = [1, 2, 3]
    clean_data['strength'] = pd.cut(clean_data['strength'], bins=bins, labels=labels)
    clean_data['strength'] = clean_data['strength'].astype("int64")
    
    return clean_data
  
  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class ContraceptiveLoader(BaseDatasetLoader):
  dataset_name = "Contraceptive Method Choice"
  col_names = ["Age", "Education", "HusbandEducation", "NumberChildren",
               "Islamic", "Work", "HusbandJob", 
               "LivingStandard", "MediaExposure", "Contraceptive"]
  target = "Contraceptive"

  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data"
    return pd.read_csv(url, names=self.col_names, skipinitialspace=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = pd.get_dummies(raw_data, columns=["HusbandJob"])
    return clean_data
  
  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class CreditApprovalLoader(BaseDatasetLoader):
  dataset_name = "Credit Approval"
  target = "A16"
  col_names = ["A1", "A2", "A3", "A4",
               "A5", "A6", "A7", "A8",
               "A9", "A10", "A11", "A12",
               "A13", "A14", "A15", "A16"]
  
  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
    return pd.read_csv(url, names=self.col_names, na_values=['?'], skipinitialspace=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.dropna()
    clean_data = clean_data.reset_index(drop=True)
    clean_data.A1 = clean_data.A1.replace({"b": 0, "a": 1})
    clean_data.A9 = clean_data.A9.replace({"f": 0, "t": 1})
    clean_data.A10 = clean_data.A10.replace({"f": 0, "t": 1})
    clean_data.A12 = clean_data.A12.replace({"f": 0, "t": 1})
    clean_data.A16 = clean_data.A16.replace({"-": 0, "+": 1})
    clean_data = pd.get_dummies(clean_data, columns=["A4", "A5", "A6", "A7", "A13"])
    return clean_data
  
  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class CreditCardsLoader(BaseDatasetLoader):
  dataset_name = "Default of Credit Card Clients"
  target = "DEFAULT_PAYMENT"
  # col_names = ""
  
  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    return pd.read_excel(url, header=1, index_col="ID")
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.copy()
    clean_data = clean_data.rename(index=str, columns={"default payment next month": "DEFAULT_PAYMENT"})
    clean_data = pd.get_dummies(clean_data, columns=['SEX', 'EDUCATION', 'MARRIAGE', ])
    return 
  
  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


# TODO: this dataset comes from the Kaggle, so we need to adapt the code to download it
# class DiamondsLoader(BaseDatasetLoader): 
#   dataset_name = ""
#   target = ""
#   def _download_and_load(self):
#     return super()._download_and_load()
#   def _preprocess(self, raw_data):
#     return super()._preprocess(raw_data)
#   def _split_features_and_target(self, data):
#     return super()._split_features_and_target(data)

class DotaLoader(BaseDatasetLoader):
  dataset_name = "Dota2 Games Results"
  target = "Winner"

  def _download_and_load(self) -> pd.DataFrame:
    url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00367/dota2Dataset.zip")
    zip_file = ZipFile(BytesIO(url.read()))
    train_file = zip_file.namelist()[0]
    test_file = zip_file.namelist()[1]
    
    hero_columns = ["H" + str(i + 1) for i in range(113)]
    col_names = ["Winner", "LocID", "GameMode", "GameType"] + hero_columns
    train = pd.read_csv(zip_file.open(train_file), names=col_names, skipinitialspace=True)
    test = pd.read_csv(zip_file.open(test_file), names=col_names, skipinitialspace=True)

    return pd.concat([train, test], ignore_index=True)

  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.copy()
    clean_data = clean_data.drop(columns="LocID")
    clean_data = pd.get_dummies(clean_data, columns=["GameMode", "GameType"])

    return clean_data

  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class DrugLoader(BaseDatasetLoader):
  
  dataset_name = "Drug Consumption"
  target = "att28"
  
  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data"
    col_names = ["att" + str(i) for i in range(32)]
    return pd.read_csv(url, header=None, names=col_names, skipinitialspace=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.drop(["att0"], axis=1)
    clean_data.att28 = clean_data.att28.replace(
      {"CL0": 1, "CL1": 2, "CL2": 2, "CL3": 3,
       "CL4": 3, "CL5": 3, "CL6": 3})
    clean_data = pd.get_dummies(clean_data)
    
    return clean_data

  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class EnergyLoader(BaseDatasetLoader):
  dataset_name = "Appliances Energy Prediction"
  target = "Appliances"
  
  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
    return pd.read_csv(url, header=0, skipinitialspace=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.drop(["date", "rv1", "rv2"], axis=1)
    bins = [0, 50, 100, 2000]
    labels = [1, 2, 3]
    clean_data['Appliances'] = pd.cut(clean_data['Appliances'], bins=bins, labels=labels)
    clean_data['Appliances'] = clean_data['Appliances'].astype("int64")
    return clean_data
  
  def _split_features_and_target(self, data):
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class FlareLoader(BaseDatasetLoader):
  dataset_name = "Solar Flare"
  target = "C"
  
  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/solar-flare/flare.data2"
    col_names = ["att" + str(i + 1) for i in range(10)] + ["C", "M", "X"]
    return pd.read_csv(url, header=None, names=col_names, sep=' ', skiprows=1, skipinitialspace=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.drop(["M", "X", "att10"], axis=1)
    
    clean_data = pd.get_dummies(clean_data, columns=["att1", "att2", "att3", "att4", "att5", "att9", "att6"])
    clean_data.loc[clean_data['att7'] > 1, 'att7'] = 0
    clean_data.loc[clean_data['att8'] > 1, 'att8'] = 0
    
    bins = [-1, 0, 10]
    labels = [0, 1]
    clean_data['C'] = pd.cut(clean_data['C'], bins=bins, labels=labels)
    clean_data['C'] = clean_data['C'].astype("int64")

    return clean_data
  
  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class GridStabilityLoader(BaseDatasetLoader):
  dataset_name = "Electrical Grid Stability Simulated Data"
  target = "stabf"
  
  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv"
    return pd.read_csv(url, header=0, skipinitialspace=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.drop(['p1', 'stab'], axis=1)
    clean_data.stabf = clean_data.stabf.replace({"unstable": 0, "stable": 1})
    return clean_data
  
  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class InternetAdsLoader(BaseDatasetLoader):
  dataset_name = "Internet Advertisements"
  target = "class"
  col_names = [f"attr-{i}" for i in range(1558)] + ["class"]
  
  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.data"
    return pd.read_csv(url, names=self.col_names, na_values=['?'], index_col=False, skipinitialspace=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.dropna()
    clean_data = clean_data.reset_index(drop=True)
    clean_data['class'] = clean_data['class'].replace({"nonad.": 0, "ad.": 1})
    return clean_data
  
  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class MagicLoader(BaseDatasetLoader):
  dataset_name = "MAGIC Gamma Telescope"
  target = "target"
  col_names = ["fLength", "fWidth", "fSize", "fConc", "fConc1",
               "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "target"]
  
  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
    return pd.read_csv(url, header=None, names=self.col_names, skipinitialspace=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.copy()
    clean_data.target = clean_data.target.replace({"h": 0, "g": 1})
    return clean_data
  
  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y
  

class BooneLoader(BaseDatasetLoader):
  dataset_name = "MiniBooNE Particle Identification"
  target = "signal"
  col_names = ["att" + str(i + 1) for i in range(50)]
  
  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt"
    return pd.read_csv(url, header=None, names=self.col_names, skiprows=1, sep=' ',
                       na_values="-0.999000E+03", skipinitialspace=True)

  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.copy()
    clean_data["signal"] = 0
    clean_data.iloc[:36499, -1] = 1
    clean_data = clean_data.dropna()
    return clean_data
  
  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y
  

class MushroomLoader(BaseDatasetLoader):
  dataset_name = "Mushroom"
  target = "result"
  col_names = ['result', 'ap-shape', 'ap-surface', 'ap-color', 'ruises?', 'dor',
              'ill-attachment', 'ill-spacing', 'ill-size', 'ill-color', 'stalk-shape',
              'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
              'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 
              'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
  
  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    return pd.read_csv(url, names=self.col_names, index_col=False, skipinitialspace=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.copy()
    clean_data.result = clean_data.result.replace({"p": 0, "e": 1})
    clean_data['ring-number'] = clean_data['ring-number'].replace({"n": 0, "o": 1, "t": 2})
    clean_data = clean_data.drop(columns=['stalk-root'])
    clean_data = pd.get_dummies(clean_data)
    return clean_data
  
  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class MusicLoader(BaseDatasetLoader):
  dataset_name = "Geographical Original of Music"
  target = "att117"
  
  def _download_and_load(self) -> pd.DataFrame:
    url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00315/Geographical%20Original%20of%20Music.zip")
    zip_file = ZipFile(BytesIO(url.read()))
    filename = zip_file.namelist()[6]

    return pd.read_csv(zip_file.open(filename), header=None, names=["att" + str(i + 1) for i in range(118)], skipinitialspace=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.copy()
    clean_data = clean_data.drop("att118", axis=1)

    bins = [-40, 35, 60]
    labels = [0, 1]
    clean_data['att117'] = pd.cut(clean_data['att117'], bins=bins, labels=labels)
    clean_data['att117'] = clean_data['att117'].astype("int64")
    
  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


# TODO: this dataset have a local file, we need to adapt this to download it
# class Musk2Loader(BaseDatasetLoader):
#   def _download_and_load(self):
#     return super()._download_and_load()
#   def _preprocess(self, raw_data):
#     return super()._preprocess(raw_data)
#   def _split_features_and_target(self, data):
#     return super()._split_features_and_target(data)


class NewsPopularityLoader(BaseDatasetLoader):
  dataset_name = "News Popularity in Multiple Social Media Platforms"
  target = "shares"

  def _download_and_load(self) -> pd.DataFrame:
    url = urllib.request.urlopen(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip")
    zip_file = ZipFile(BytesIO(url.read()))
    filename = zip_file.namelist()[2]
    
    return pd.read_csv(zip_file.open(filename), skipinitialspace=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.drop(columns="url")
    
    bins = [0, 1000, 2500, 5000, 1000000]
    labels = [1, 2, 3, 4]
    clean_data['shares'] = pd.cut(clean_data['shares'], bins=bins, labels=labels)
    clean_data['shares'] = clean_data['shares'].astype("int64")

    return clean_data

  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class NurseryLoader(BaseDatasetLoader):
  dataset_name = "Nursery"
  target = "att9"
  col_names = ["att" + str(i + 1) for i in range(9)]
  
  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
    return pd.read_csv(url, header=None, names=self.col_names, skipinitialspace=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.copy()
    clean_data.att9 = clean_data.att9.replace({"not_recom": 0, "recommend": 1, "very_recom": 1, "priority": 1, "spec_prior": 2})
    clean_data = pd.get_dummies(clean_data)
    return clean_data

  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class OccupancyLoader(BaseDatasetLoader):
  dataset_name = "Occupancy Detection"
  target = "Occupancy"
  
  def _download_and_load(self) -> pd.DataFrame:
    url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip")
    zip_file = ZipFile(BytesIO(url.read()))
    train_file = zip_file.namelist()[2]
    test_file1 = zip_file.namelist()[1]
    test_file2 = zip_file.namelist()[0]
    
    train = pd.read_csv(zip_file.open(train_file), header=0, skipinitialspace=True)
    test1 = pd.read_csv(zip_file.open(test_file1), header=0, skipinitialspace=True)
    test2 = pd.read_csv(zip_file.open(test_file2), header=0, skipinitialspace=True)
    
    return pd.concat([train, test1, test2], ignore_index=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.drop("date", axis=1)
    return clean_data

  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


# TODO: this dataset use a local version of a CSV file, we need to adapt this
# to load from internet instead
# class PhishingLoader(BaseDatasetLoader):
#   def _download_and_load(self):
#     return super()._download_and_load()
#   def _preprocess(self, raw_data):
#     return super()._preprocess(raw_data)
#   def _split_features_and_target(self, data):
#     return super()._split_features_and_target(data)

class SpamBaseLoader(BaseDatasetLoader):
  col_names = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over',
               'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive',
               'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free', 
               'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your', 
               'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 
               'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data',
               'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm',
               'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
               'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(', 'char_freq_[',
               'char_freq_!', 'char_freq_$','char_freq_#', 'capital_run_length_average', 'capital_run_length_longest',
               'capital_run_length_total', 'spam']
  dataset_name = "Spambase"
  target = "spam"
  
  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    return pd.read_csv(url, names=self.col_names, index_col=False, skipinitialspace=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    return raw_data

  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y

# TODO: implement a online version of this dataset, not only local
# class StudentAlcoholLoader(BaseDatasetLoader):
#   def _download_and_load(self):
#     return super()._download_and_load()
#   def _preprocess(self, raw_data):
#     return super()._preprocess(raw_data)
#   def _split_features_and_target(self, data):
#     return super()._split_features_and_target(data)

# TODO: implement a online version of this dataset, not only local
# class StudentPerformanceLoader(BaseDatasetLoader):
#   def _download_and_load(self):
#     return super()._download_and_load()
#   def _preprocess(self, raw_data):
#     return super()._preprocess(raw_data)
#   def _split_features_and_target(self, data):
#     return super()._split_features_and_target(data)


class SuperConductorLoader(BaseDatasetLoader):
  dataset_name = "Superconductivity Data"
  target = "critical_temp"
  
  def _download_and_load(self) -> pd.DataFrame:
    url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip")

    zip_file = ZipFile(BytesIO(url.read()))
    filename = zip_file.namelist()[1]

    return pd.read_csv(zip_file.open(filename), header=0, skipinitialspace=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.copy()
    bins = [0, 5,20, 60, 2000]
    labels = [0, 1, 2, 3]
    clean_data['critical_temp'] = pd.cut(clean_data['critical_temp'], bins=bins, labels=labels)
    clean_data['critical_temp'] = clean_data['critical_temp'].astype("int64")
    clean_data = pd.get_dummies(clean_data, columns = ["number_of_elements"])
    
    return clean_data

  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class TurkStudentEvalLoader(BaseDatasetLoader):
  dataset_name = "Turkiye Student Evaluation"
  target = "instr"
  
  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00262/turkiye-student-evaluation_generic.csv"
    return pd.read_csv(url, index_col=False, sep=',', skipinitialspace=True)
  
  def _preprocess(self, raw_data) -> pd.DataFrame:
    return raw_data.drop(columns=['class', 'nb.repeat'])

  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class WineLoader(BaseDatasetLoader):
  dataset_name = "Wine Quality"
  target = "quality"
  
  def _download_and_load(self) -> pd.DataFrame:
    white = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
                       index_col=False, sep=';', skipinitialspace=True)
    white.insert(0, 'type', 'white')
    red = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                      index_col=False, sep=';', skipinitialspace=True)
    red.insert(0, 'type', 'red')
    return pd.concat([white, red], ignore_index=True)

  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.copy()
    clean_data = pd.get_dummies(clean_data)
    
    bins = [0, 4, 5, 6, 10]
    labels = [1, 2, 3, 4]
    clean_data['quality'] = pd.cut(clean_data['quality'], bins=bins, labels=labels)
    clean_data['quality'] = clean_data['quality'].astype("int64")
    
    return clean_data
  
  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class YeastLoader(BaseDatasetLoader):
  dataset_name = "Yeast"
  target = "class"
  col_names = ['name', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'class']

  def _download_and_load(self) -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
    return pd.read_csv(url, names=self.col_names, index_col=False, sep=' ', skipinitialspace=True)

  def _preprocess(self, raw_data) -> pd.DataFrame:
    clean_data = raw_data.copy()
    clean_data = clean_data.drop(columns=['name'])
    clean_data['class'] = clean_data['class'].replace(
      {'CYT': 0, 'NUC': 1, 'MIT': 2, 'ME3': 3, 'ME2': 4,
       'ME1': 4, 'EXC': 4, 'VAC': 4, 'POX': 4, 'ERL': 4})
    clean_data = clean_data.loc[clean_data["class"] < 4]
    return clean_data

  def _split_features_and_target(self, data) -> tuple[np.ndarray, np.ndarray]:
    X = data.drop(columns=[self.target]).values
    y = data[self.target].values
    return X, y


class UCILoaderFactory:
  """Factory used to instanciate the correct Loader based on the dataset name.
  
  Some of these datasets are build with one-versus-all strategy and others are
  naturally binary problems. These are the datasets used in [1].

  Refs
  [1] Schumacher, Tobias, Markus Strohmaier, and Florian Lemmerich.
      "A comparative evaluation of quantification methods."
      Journal of Machine Learning Research 26.55 (2025): 1-54.
  """
  _registry = {
    "adult": AdultLoader,
    "avila": AvilaLoader,
    "bike": BikeLoader,
    "blog": BlogFeedbackLoader,
    "bc-cont": BreastCancerContLoader,
    "bc-int": BreastCancerIntLoader,
    "cars": CarsLoader,
    "conc": ConcreteLoader,
    "contra": ContraceptiveLoader,
    "cappl": CreditApprovalLoader,
    "ccard": CreditCardsLoader,
    "dota": DotaLoader,
    "drug": DrugLoader,
    "ener": EnergyLoader,
    "flare": FlareLoader,
    "grid": GridStabilityLoader,
    "ads": InternetAdsLoader,
    "magic": MagicLoader,
    "boone": BooneLoader,
    "mush": MushroomLoader,
    "music": MusicLoader,
    "news": NewsPopularityLoader,
    "nurse": NurseryLoader,
    "occup": OccupancyLoader,
    "spam": SpamBaseLoader,
    "cond": SuperConductorLoader,
    "turk": TurkStudentEvalLoader,
    "wine": WineLoader,
    "yeast": YeastLoader,
  }
  
  @classmethod
  def get_loader(cls, dataset_name: str) -> BaseDatasetLoader:
    pipeline_class = cls._registry.get(dataset_name.lower())
    if not pipeline_class:
      raise ValueError(f"Dataset '{dataset_name}' not supported on UCI datasets.")
    return pipeline_class()
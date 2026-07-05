import os
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
from quack.utils import get_quack_home, download_file
import scipy.sparse as sp
from abc import ABC, abstractmethod


FORMAN_DATASETS: list[str] = ['fbis', 'la1', 'la2', 'ohscal']

# TODO: these options will be implemented in a new method to
# return the same experiment config from the Forman (2005) paper.

# AVAILABLE_P_TRAIN = ['all', 10, 20, 50, 100, 200]
# AVAILABLE_CLASSES: dict[str, list[str]] = {
#   'fbis': ['111', '142', '189'],
#   'ohscal': ['Antibodies', 'Carcinoma', 'DNA', 'In-Vitro', 'Molecular-Sequence-Data', 'Pregnancy', 'Prognosis', 'Receptors', 'Risk-Factors', 'Tomography'],
#   'la1': ['Entertainment', 'Financial', 'Metro', 'Sports'],
#   'la2': ['Entertainment', 'Financial', 'Metro', 'Sports'],
# }


def load_forman(dataset: str,
                data_home: Path = None,
                force_load: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """ Load the datasets from [1] following scikit patterns. If there is no tmp dataset
  locally available, this method will download a tmp version from zenodo and save locally.

  Note that the splits are not performed, once the user may want to perform a binary or
  a multiclass quantification over these datasets.

  Refs:
    [1] Forman, G. Quantifying counts and costs via classification.
        Data Min Knowl Disc 17, 164–206 (2008). https://doi.org/10.1007/s10618-008-0097-y

  Args:
      dataset (str): Dataset name. The list of valid dataset name is available
        at quack.datasets.FORMAN_DATASETS.
      data_home (Path, optional): Directory that the tmp dataset file will be saved
        locally. If its None, the default dataset will be obtained from utils. Defaults to None.
      force_load (bool, optional): Force the download of the files even if there is a tmp file
        available locally. Defaults to False.

  Returns:
      tuple[np.ndarray, np.ndarray]: Tuple containing the features and labels
        of the choosen dataset.
  """

  assert dataset in FORMAN_DATASETS

  if data_home is None:
    data_home = get_quack_home()

  URL = f'https://zenodo.org/records/20707970/files/{dataset}.zip'
  unzipped_path = os.path.join(data_home, dataset)

  if not os.path.exists(unzipped_path):
    downloaded_path = os.path.join(data_home, f'{dataset}.zip')
    download_file(URL, downloaded_path, exist_ok=force_load)

    with zipfile.ZipFile(downloaded_path) as file:
      file.extractall(data_home)
    os.remove(downloaded_path)

  # Load all data
  X = sp.load_npz(os.path.join(unzipped_path, f'{dataset}_X.npz'))
  y = np.load(os.path.join(unzipped_path, f'{dataset}_y.npy'), allow_pickle=True)

  return X.toarray(), y


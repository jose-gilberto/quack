import os
import pandas as pd
import numpy as np
from pathlib import Path
from quack.utils import get_quack_home, download_file


REVIEWS_SENTIMENT_DATASETS: list[str] = ['hp', 'kindle', 'imdb']


def load_reviews(dataset: str,
                 data_home: Path = None,
                 force_load: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """ Load the review dataset [1] following scikit patterns. If there is no tmp dataset
  locally available, this method will download a tmp version from zenodo and save locally.

  Note that the texts remains in plain format, the embedding or other processing methods may be
  performed locally based on requirements to run the datasets.

  Refs:
    [1] Esuli, A., Moreo, A., and Sebastiani, F. "A recurrent neural network for sentiment quantification."
        Proceedings of the 27th ACM International Conference on Information and Knowledge Management.
        2018. <https://dl.acm.org/doi/abs/10.1145/3269206.3269287>`.

  Args:
      dataset (str): Dataset name. The list of valid dataset name is available
        at quack.datasets.REVIEW_SENTIMENT_DATASETS.
      data_home (Path, optional): Directory that the tmp dataset file will be saved
        locally. If its None, the default dataset will be obtained from utils. Defaults to None.
      force_load (bool, optional): Force the download of the files even if there is a tmp file
        available locally. Defaults to False.

  Returns:
      tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: _description_
  """
  assert dataset in REVIEWS_SENTIMENT_DATASETS

  if data_home is None:
    data_home = get_quack_home()

  URL_TRAIN = f'https://zenodo.org/record/4117827/files/{dataset}_train.txt'
  URL_TEST = f'https://zenodo.org/record/4117827/files/{dataset}_test.txt'

  os.makedirs(os.path.join(data_home, 'reviews'), exist_ok=True)

  train_path = os.path.join(data_home, 'reviews', dataset, 'train.txt')
  test_path = os.path.join(data_home, 'reviews', dataset, 'test.txt')
  
  download_file(URL_TRAIN, train_path, exist_ok=force_load)
  download_file(URL_TEST, test_path, exist_ok=force_load)

  train_set = pd.read_csv(train_path, sep='\t', header=None, names=['label', 'text'])
  X_train, y_train = train_set['text'], train_set['label']

  test_set = pd.read_csv(test_path, sep='\t', header=None, names=['label', 'text'])
  X_test, y_test = test_set['text'], test_set['label']

  return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
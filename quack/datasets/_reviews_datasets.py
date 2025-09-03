import os
from pathlib import Path
from quack.utils import get_quack_home, download_file


REVIEWS_SENTIMENT_DATASETS: list[str] = ['hp', 'kindle', 'imdb']

def load_reviews(dataset: str, data_home: Path = None):
  assert dataset in REVIEWS_SENTIMENT_DATASETS

  if data_home is None:
    data_home = get_quack_home()

  URL_TRAIN = f'https://zenodo.org/record/4117827/files/{dataset}_train.txt'
  URL_TEST = f'https://zenodo.org/record/4117827/files/{dataset}_test.txt'

  os.makedirs(os.path.join(data_home, 'reviews'), exist_ok=True)

  train_path = os.path.join(data_home, 'reviews', dataset, 'train.txt')
  test_path = os.path.join(data_home, 'reviews', dataset, 'test.txt')
  
  download_file(URL_TRAIN, train_path)
  download_file(URL_TEST, test_path)


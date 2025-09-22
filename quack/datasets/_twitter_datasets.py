import os
import pandas as pd
import zipfile
import numpy as np
from pathlib import Path
from quack.utils import get_quack_home, download_file


TWITTER_SENTIMENT_DATASETS: list[str] = [
  'gasp', 'hcr', 'omd', 'sanders',
  'semeval13', 'semeval14', 'semeval16',
  'sst', 'wa', 'wb',
]


def load_twitter(dataset: str,
                 data_home: Path = None,
                 force_load: bool = False) -> any:
  assert dataset in TWITTER_SENTIMENT_DATASETS

  if data_home is None:
    data_home = get_quack_home()

  URL = 'https://zenodo.org/record/4255764/files/tweet_sentiment_quantification_snam.zip'
  unzipped_path = os.path.join(data_home, 'tweet_sentiment_quantification_snam')

  if not os.path.exists(unzipped_path):
    downloaded_path = os.path.join(data_home, 'tweet_sentiment_quantification_snam.zip')
    download_file(URL, downloaded_path, exist_ok=force_load)

    with zipfile.ZipFile(downloaded_path) as file:
      file.extractall(data_home)
    os.remove(downloaded_path)

    if dataset in {}
  
import os
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
from quack.utils import get_quack_home, download_file
import scipy.sparse as sp
from abc import ABC, abstractmethod


FORMAN_DATASETS: list[str] = ['fbis', 'la1', 'la2', 'ohscal']

AVAILABLE_P_TRAIN = ['all', 10, 20, 50, 100, 200]

AVAILABLE_CLASSES: dict[str, list[str]] = {
  'fbis': ['111', '142', '189'],
  'ohscal': ['Antibodies', 'Carcinoma', 'DNA', 'In-Vitro', 'Molecular-Sequence-Data', 'Pregnancy', 'Prognosis', 'Receptors', 'Risk-Factors', 'Tomography'],
  'la1': ['Entertainment', 'Financial', 'Metro', 'Sports'],
  'la2': ['Entertainment', 'Financial', 'Metro', 'Sports'],
}


class FormanTestSplit:

  def __init__(self, p: int, X_test, y_test):
    self.p = p
    self.X = X_test
    self.y = y_test

  def __repr__(self):
    return f'<FormanTestSplit p={self.p} n={self.X.shape[0]}>'


class FormanTrainSplit:

  def __init__(self, n_pos: int, X_train, y_train, test_splits) -> None:
    self.n_pos = n_pos
    self.X = X_train
    self.y = y_train
    # Create the test splits
    self.tests: dict[int, FormanTestSplit] = {
      p: FormanTestSplit(p=p, X_test=X_test, y_test=y_test) for p, (X_test, y_test) in test_splits.items()
    }

  def available_tests(self):
    """ Returns all values of p available in test splits.
    TODO: refactor docstring
    Returns:
        _type_: _description_
    """
    return sorted(self.tests.keys())
  
  def get_test(self, p: int) -> FormanTestSplit:
    return self.tests.get(p, None)
  
  def __repr__(self) -> str:
    return f'<FormanTrainSplit n_pos={self.n_pos} n={self.X.shape[0]} tests={len(self.tests)}>'

class FormanDataset:

  def __init__(self, dataset: str, class_name: str, train_splits: dict[str, FormanTrainSplit]):
    self.class_name = class_name
    self.dataset = dataset
    self.trains = train_splits # dict n_pos -> train_split

  def available_trains(self):
    return sorted(self.trains.keys())
  
  def get_train(self, n_pos):
    return self.trains.get(n_pos, None)
  
  def __repr__(self) -> str:
    return f'<FormanDataset dataset={self.dataset} class={self.class_name} trains={len(self.trains)}>'


def build_forman_dataset(dataset: str, structure):
  class_name = structure['class']
  train_splits = {}
  for t in structure['train_splits']:
    train_splits[t['n_pos']] = FormanTrainSplit(
      n_pos=t['n_pos'],
      X_train=t['X_train'],
      y_train=t['y_train'],
      test_splits=t['test_splits']
    )
  return FormanDataset(dataset=dataset, class_name=class_name, train_splits=train_splits)


# TODO: create a class structure to handle the FormanDatasets
def load_forman(dataset: str,
                positive_class: str,
                data_home: Path = None,
                force_load: bool = False,
                p_train: str | int = 'all') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  
  assert dataset in FORMAN_DATASETS
  assert positive_class in AVAILABLE_CLASSES[dataset]
  assert p_train in AVAILABLE_P_TRAIN

  if data_home is None:
    data_home = get_quack_home()

  URL = f'https://zenodo.org/records/17238986/files/{dataset}.zip'
  unzipped_path = os.path.join(data_home, dataset)

  if not os.path.exists(unzipped_path):
    downloaded_path = os.path.join(data_home, f'{dataset}.zip')
    download_file(URL, downloaded_path, exist_ok=force_load)

    with zipfile.ZipFile(downloaded_path) as file:
      file.extractall(data_home)
    os.remove(downloaded_path)

  # Load all training splits at once
  if p_train == 'all':
    train_dirs = [d for d in os.listdir(os.path.join(unzipped_path, positive_class)) if d.startswith('train_p')]
  else:
    train_dirs = [f'train_p{p_train}']

  _splits = {
    'class': positive_class,
    'train_splits': []
  }

  for train_dir in train_dirs:
    full_train_dir = os.path.join(unzipped_path, positive_class, train_dir)
    if not os.path.exists(full_train_dir):
      continue

    # Load training data
    X_train = sp.load_npz(os.path.join(full_train_dir, 'X_train.npz'))
    y_train = np.load(os.path.join(full_train_dir, 'y_train.npy'), allow_pickle=True)

    # Load all included test splits
    test_splits: dict[int, tuple] = {}
    for d in os.listdir(full_train_dir):
      if d.startswith('test_p'):
        p_str = d.replace('test_p', '')
        p = int(p_str)

        test_dir = os.path.join(full_train_dir, d)

        X_test = sp.load_npz(os.path.join(test_dir, 'X_test.npz'))
        y_test = np.load(os.path.join(test_dir, 'y_test.npy'), allow_pickle=True)

        test_splits[p] = (X_test, y_test) # Store features, labels for splits
  
    _splits['train_splits'].append({
      'n_pos': int(train_dir.replace('train_p', '')),
      'X_train': X_train,
      'y_train': y_train,
      'test_splits': test_splits
    })

  return build_forman_dataset(dataset, _splits)


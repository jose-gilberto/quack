import csv
import re
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from quack.datasets._arff import read_multilabel
from quack.datasets._licenses import DatasetLicense, check_terms
from quack.datasets.utils import BaseMultilabelDatasetLoader

_GOEMOTIONS_COMMIT = "db336cdca06c1bed170c672c20f00a1559e6242b"
_GOEMOTIONS_URL = (f"https://raw.githubusercontent.com/google-research/google-research/"
                   f"{_GOEMOTIONS_COMMIT}/goemotions/data/")
GOEMOTIONS_FILES: dict[str, str] = {
  "train.tsv": "sha256:1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5",
  "dev.tsv": "sha256:575489c079c9de1097062a01738f998590d6b7ead66dd1c9fd1d2ba01fd8bc62",
  "test.tsv": "sha256:0587b2dd8b27b97352adbfc3fb083d46005c8946657fdc2b1ca8b1cc7f1f8be4",
  "emotions.txt": "sha256:45c3ef86782d2a4d7fedcd6d8c111aa0d0e94720689bd164fac94fefb4495a89",
  "ekman_mapping.json": "sha256:d6b1fea382917c1685c42e1324a0344106be9390f52a3850ce032656d103a328",
  "sentiment_mapping.json": "sha256:801f6f028082141bd38541a4b14d65b33bdc55d29c7b2aaa029fefecf206f76b",
}

GOEMOTIONS_LICENSE = DatasetLicense(
  spdx="Apache-2.0",
  redistribution="allowed",
  source="https://github.com/google-research/google-research/tree/master/goemotions",
  checked="2026-10-27",
  citation="Demszky, D. et al. GoEmotions: A Dataset of Fine-Grained Emotions. ACL 2020.",
  conditions=("attribution",),
  notes="Reddit comments; the raw version (not used here) carries usernames.",
)


def read_goemotions(folder: Path) -> tuple[pd.DataFrame, list[str]]:
  """Read the simplified GoEmotions files (train/dev/test TSV and the emotion list).

  Returns a frame with `text`, `labels` (list of emotion names), `id` and
  `split`, and the 28 emotion names.
  """
  emotions = (folder / "emotions.txt").read_text(encoding="utf-8").split()
  parts = []
  for split, filename in (("train", "train.tsv"), ("val", "dev.tsv"), ("test", "test.tsv")):
    df = pd.read_csv(folder / filename, sep="\t", header=None, names=["text", "labels", "id"],
                     quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    df["split"] = split
    parts.append(df)
  data = pd.concat(parts, ignore_index=True)
  data["labels"] = [[emotions[int(i)] for i in s.split(",")] for s in data["labels"]]
  return data, emotions


def _indicator(label_sets, names: list[str]) -> np.ndarray:
  index = {n: i for i, n in enumerate(names)}
  Y = np.zeros((len(label_sets), len(names)), dtype=np.int8)
  for k, labels in enumerate(label_sets):
    for label in labels:
      Y[k, index[label]] = 1
  return Y


class ToLDBrLoader(BaseMultilabelDatasetLoader):
  """ToLD-Br: Brazilian Portuguese tweets with six toxicity categories.

  Each tweet was labelled by three annotators. A category is positive when at
  least `min_votes` annotators chose it; the vote counts are kept in the
  metadata (`extras["votes"]`), so soft labels are available.

  Parameters
  ----------
  min_votes : int, default = 2
    Votes needed for a positive label (2 = majority of the 3 annotators).

  References
  ----------
  [1] Leite, J. A., Silva, D., Bontcheva, K., & Scarton, C. (2020). Toxic
      Language Detection in Social Media for Brazilian Portuguese: New Dataset
      and Multilingual Analysis. AACL-IJCNLP 2020.
  """
  dataset_name = "ToLD-Br"
  cache_dir = "told_br"
  is_text = True
  labels = ["homophobia", "obscene", "insult", "racism", "misogyny", "xenophobia"]
  url = ("https://raw.githubusercontent.com/JAugusto97/ToLD-Br/"
         "6b325d26a9d25b321a3e9ba98ef98832b56729f5/ToLD-BR.csv")
  checksum = "sha256:a905b90d5886b9c80d737aa26c41dd29077155f23b9b857080634a78f8203c90"
  license = DatasetLicense(
    spdx="CC-BY-SA-4.0",
    redistribution="allowed-sharealike",
    source="https://github.com/JAugusto97/ToLD-Br/blob/main/LICENSE_ToLD-Br.txt",
    checked="2026-10-27",
    citation=("Leite, J. A. et al. Toxic Language Detection in Social Media for Brazilian "
              "Portuguese: New Dataset and Multilingual Analysis. AACL-IJCNLP 2020."),
    conditions=("attribution", "share-alike"),
    notes="Mentions replaced by @user; tweet content is also subject to the platform's terms.",
  )

  def __init__(self, data_home=None, force_load: bool = False, min_votes: int = 2):
    super().__init__(data_home, force_load)
    self.min_votes = min_votes

  def _download_and_load(self) -> pd.DataFrame:
    return pd.read_csv(self._fetch(self.url, "ToLD-BR.csv", self.checksum))

  def _preprocess(self, raw_data: pd.DataFrame) -> pd.DataFrame:
    self.extras_["votes"] = raw_data[self.labels].to_numpy(dtype=float)
    return raw_data

  def _split_features_and_target(self, data):
    return data["text"].tolist(), (data[self.labels].to_numpy() >= self.min_votes)

  def _label_names(self, data) -> list[str]:
    return list(self.labels)


class GoEmotionsLoader(BaseMultilabelDatasetLoader):
  """GoEmotions (simplified version): Reddit comments with 27 emotions plus neutral.

  The 54,263 comments with rater agreement, split into train / val / test as
  published by the authors. `load_hierarchical("goemotions")` returns the same
  data with the sentiment > Ekman group > emotion hierarchy.

  References
  ----------
  [1] Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., &
      Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. ACL 2020.
  """
  dataset_name = "GoEmotions"
  cache_dir = "goemotions"
  is_text = True
  license = GOEMOTIONS_LICENSE

  def _download_and_load(self) -> pd.DataFrame:
    for filename, checksum in GOEMOTIONS_FILES.items():
      self._fetch(_GOEMOTIONS_URL + filename, filename, checksum)
    data, self._emotions = read_goemotions(self.dataset_dir)
    return data

  def _preprocess(self, raw_data):
    return raw_data

  def _split_features_and_target(self, data):
    return data["text"].tolist(), _indicator(data["labels"], self._emotions)

  def _label_names(self, data) -> list[str]:
    return list(self._emotions)

  def _split(self, data):
    return data["split"].to_numpy()


_SGML_DOC = re.compile(r"<REUTERS([^>]*)>(.*?)</REUTERS>", re.S)
_SGML_ATTR = re.compile(r'(\w+)="([^"]*)"')
_SGML_TOPICS = re.compile(r"<TOPICS>(.*?)</TOPICS>", re.S)
_SGML_D = re.compile(r"<D>(.*?)</D>", re.S)
_SGML_TITLE = re.compile(r"<TITLE>(.*?)</TITLE>", re.S)
_SGML_BODY = re.compile(r"<BODY>(.*?)</BODY>", re.S)


def parse_reuters_sgml(text: str):
  """Yield `(split, topics, text)` for the ModApte documents of one Reuters-21578 `.sgm` file."""
  for match in _SGML_DOC.finditer(text):
    attrs = dict(_SGML_ATTR.findall(match.group(1)))
    if attrs.get("TOPICS") != "YES" or attrs.get("LEWISSPLIT") not in {"TRAIN", "TEST"}:
      continue
    body = match.group(2)
    topics = _SGML_TOPICS.search(body)
    labels = _SGML_D.findall(topics.group(1)) if topics else []
    parts = [m.group(1).strip() for m in (_SGML_TITLE.search(body), _SGML_BODY.search(body)) if m]
    yield ("train" if attrs["LEWISSPLIT"] == "TRAIN" else "test"), labels, " ".join(parts)


REUTERS_LICENSE = DatasetLicense(
  spdx="LicenseRef-Reuters-21578",
  redistribution="allowed-research-only",
  source="https://ics.uci.edu/~kdd/databases/reuters21578/README.txt",
  checked="2026-10-27",
  citation="Lewis, D. D. Reuters-21578 text categorization test collection, Distribution 1.0.",
  conditions=("research-only", "cite"),
  notes=("Copyright Reuters Ltd.; free distribution allowed for research purposes only. "
         "Acknowledge 'Reuters-21578, Distribution 1.0'."),
)


class _ReutersBase(BaseMultilabelDatasetLoader):
  is_text = True
  cache_dir = "reuters21578"
  license = REUTERS_LICENSE

  def __init__(self, data_home=None, force_load: bool = False, min_train_positives: int = 0):
    super().__init__(data_home, force_load)
    self.min_train_positives = min_train_positives

  def _preprocess(self, raw_data: pd.DataFrame) -> pd.DataFrame:
    names = sorted({t for topics in raw_data["topics"] for t in topics})
    if self.min_train_positives > 0:
      train = raw_data.loc[raw_data["split"] == "train", "topics"]
      counts = pd.Series([t for topics in train for t in topics]).value_counts()
      names = [n for n in names if counts.get(n, 0) >= self.min_train_positives]
    self._names = names
    return raw_data

  def _split_features_and_target(self, data):
    keep = set(self._names)
    return data["text"].tolist(), _indicator([[t for t in ts if t in keep] for ts in data["topics"]], self._names)

  def _label_names(self, data) -> list[str]:
    return list(self._names)

  def _split(self, data):
    return data["split"].to_numpy()


class Reuters21578Loader(_ReutersBase):
  """Reuters-21578, ApteMod version (NLTK copy): 10,788 newswire stories, 90 topics.

  Only documents with at least one topic; official split of 7,769 training and
  3,019 test documents. For the full ModApte split used in [2] (9,603 / 3,299,
  including documents without topics) use `"reuters21578-modapte"`.

  Parameters
  ----------
  min_train_positives : int, default = 0
    Drop topics with fewer training positives ([2] used 5).

  References
  ----------
  [1] Lewis, D. D. Reuters-21578 text categorization test collection, Distribution 1.0.
  [2] Moreo, A., Francisco, M., & Sebastiani, F. (2023). Multi-Label Quantification.
      ACM Transactions on Knowledge Discovery from Data, 18(1).
  """
  dataset_name = "Reuters-21578 (ApteMod)"
  url = ("https://raw.githubusercontent.com/nltk/nltk_data/"
         "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/corpora/reuters.zip")
  checksum = "sha256:9a59a43823f02a6e2777075c989a3dc454e4b6f68e0332ee3c0e8264075b62f5"

  def _download_and_load(self) -> pd.DataFrame:
    path = self._fetch(self.url, "reuters.zip", self.checksum)
    rows = []
    with zipfile.ZipFile(path) as zf:
      for line in zf.read("reuters/cats.txt").decode("latin-1").splitlines():
        if line.strip():
          doc, *topics = line.split()
          rows.append({"text": zf.read(f"reuters/{doc}").decode("latin-1"), "topics": topics,
                       "split": "train" if doc.startswith("training/") else "test"})
    return pd.DataFrame(rows)


class Reuters21578ModApteLoader(_ReutersBase):
  """Reuters-21578, full ModApte split read from the original SGML files.

  Download `reuters21578.tar.gz` from the UCI KDD archive
  (https://kdd.ics.uci.edu/databases/reuters21578/) and place it in
  `~/.quack_data/reuters21578/`.

  Parameters
  ----------
  min_train_positives : int, default = 0
    Drop topics with fewer training positives (Moreo et al., 2023, used 5).
  """
  dataset_name = "Reuters-21578 (ModApte)"

  def _download_and_load(self) -> pd.DataFrame:
    path = self._require_file("reuters21578.tar.gz",
                              "Download it from https://kdd.ics.uci.edu/databases/reuters21578/")
    rows = []
    with tarfile.open(path) as tf:
      for member in sorted(tf.getmembers(), key=lambda m: m.name):
        if member.name.endswith(".sgm"):
          for split, topics, text in parse_reuters_sgml(tf.extractfile(member).read().decode("latin-1")):
            rows.append({"text": text, "topics": topics, "split": split})
    return pd.DataFrame(rows)


MULAN_LICENSE = DatasetLicense(
  spdx="NOASSERTION",
  redistribution="not-declared",
  source="https://mulan.sourceforge.net/datasets-mlc.html",
  checked="2026-10-27",
  citation=("Tsoumakas, G. et al. Mulan: A Java Library for Multi-Label Learning. JMLR 12, 2011. "
            "Cite also the dataset's original paper, listed on the Mulan page."),
  access="manual",
  conditions=("cite",),
  notes="Mulan, MEKA and the UCO KDIS repository declare no license for these files.",
)


class MulanLoader(BaseMultilabelDatasetLoader):
  """A dataset of the Mulan/MEKA multi-label collection, read from local ARFF files.

  The sources declare no license, so the files are not downloaded
  automatically: get them from https://mulan.sourceforge.net/datasets-mlc.html
  (ARFF + XML) or from MEKA, and extract them into
  `~/.quack_data/mulan/<name>/` as `<name>-train.arff`, `<name>-test.arff`
  (or a single `<name>.arff`) plus `<name>.xml` for Mulan files.

  These are 12 of the 15 datasets of [1].

  Parameters
  ----------
  min_train_positives : int, default = 0
    Drop labels with fewer training positives ([1] used 5).

  References
  ----------
  [1] Moreo, A., Francisco, M., & Sebastiani, F. (2023). Multi-Label Quantification.
      ACM Transactions on Knowledge Discovery from Data, 18(1).
  """
  subset: str = ""
  license = MULAN_LICENSE

  def __init__(self, data_home=None, force_load: bool = False, min_train_positives: int = 0):
    super().__init__(data_home, force_load)
    self.min_train_positives = min_train_positives

  @property
  def cache_dir(self) -> str:
    return f"mulan/{self.subset}"

  def _find(self, *patterns):
    for pattern in patterns:
      hits = sorted(self.dataset_dir.glob(pattern))
      if hits:
        return hits[0]
    return None

  def _download_and_load(self) -> dict:
    s = self.subset
    train = self._find(f"{s}-train.arff", f"{s}_train.arff", f"{s}*train*.arff")
    test = self._find(f"{s}-test.arff", f"{s}_test.arff", f"{s}*test*.arff")
    full = self._find(f"{s}.arff")
    xml = self._find(f"{s}.xml", "*.xml")
    if not (train and test) and not full:
      raise FileNotFoundError(
        f"Mulan dataset '{s}' not found in {self.dataset_dir}.\n"
        f"Download it from https://mulan.sourceforge.net/datasets-mlc.html and extract the "
        f"ARFF (and XML) files into that folder.")
    if train and test:
      Xa, Ya, _, names = read_multilabel(train, xml)
      Xb, Yb, _, names_b = read_multilabel(test, xml)
      if names != names_b:
        raise ValueError(f"{s}: train and test files have different labels")
      X = sp.vstack([Xa, Xb]) if sp.issparse(Xa) else np.vstack([Xa, Xb])
      return {"X": X, "Y": sp.vstack([Ya, Yb]).tocsr(), "names": names,
              "split": np.array(["train"] * Ya.shape[0] + ["test"] * Yb.shape[0])}
    X, Y, _, names = read_multilabel(full, xml)
    return {"X": X, "Y": Y, "names": names, "split": None}

  def _preprocess(self, raw_data: dict) -> dict:
    if self.min_train_positives > 0:
      rows = raw_data["split"] == "train" if raw_data["split"] is not None else slice(None)
      counts = np.asarray(raw_data["Y"][rows].sum(axis=0)).ravel()
      keep = np.flatnonzero(counts >= self.min_train_positives)
      raw_data = {**raw_data, "Y": raw_data["Y"][:, keep], "names": [raw_data["names"][i] for i in keep]}
    return raw_data

  def _split_features_and_target(self, data: dict):
    X = data["X"].toarray() if sp.issparse(data["X"]) else data["X"]
    return X, data["Y"]

  def _label_names(self, data: dict) -> list[str]:
    return list(data["names"])

  def _split(self, data: dict):
    return data["split"]


MULAN_DATASETS: dict[str, str] = {
  "emotions": "Emotions (music)",
  "scene": "Scene (images)",
  "yeast": "Yeast (gene functions)",
  "birds": "Birds (audio)",
  "genbase": "Genbase (proteins)",
  "medical": "Medical (clinical text)",
  "tmc2007_500": "TMC2007-500 (aviation reports)",
  "enron": "Enron (e-mail)",
  "mediamill": "Mediamill (video)",
  "bibtex": "Bibtex (tags)",
  "corel5k": "Corel5k (images)",
  "delicious": "Delicious (bookmarks)",
}


def _mulan_loader(subset: str, name: str) -> type:
  cls_name = "".join(c for c in subset.title() if c.isalnum()) + "MulanLoader"
  return type(cls_name, (MulanLoader,), {"subset": subset, "dataset_name": f"{name} [Mulan]",
                                         "__doc__": f"{name}, from the Mulan collection. See `MulanLoader`."})


class JigsawToxicLoader(BaseMultilabelDatasetLoader):
  """Jigsaw Toxic Comment Classification: Wikipedia comments with six toxicity types.

  The data is distributed through a Kaggle competition, which requires
  accepting its rules on the website. Download it with
  `kaggle competitions download -c jigsaw-toxic-comment-classification-challenge`
  (or from the competition page) and place the zip, or `train.csv`,
  `test.csv` and `test_labels.csv`, in `~/.quack_data/jigsaw_toxic/`.
  Test rows that were not scored (labels -1) are dropped.

  References
  ----------
  [1] Jigsaw/Conversation AI. Toxic Comment Classification Challenge. Kaggle, 2018.
  """
  dataset_name = "Jigsaw Toxic Comment Classification"
  cache_dir = "jigsaw_toxic"
  is_text = True
  labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
  license = DatasetLicense(
    spdx="CC-BY-SA-3.0",
    redistribution="allowed-sharealike",
    source="https://www.tensorflow.org/datasets/catalog/wikipedia_toxicity_subtypes",
    checked="2026-10-27",
    citation="Jigsaw/Conversation AI. Toxic Comment Classification Challenge. Kaggle, 2018.",
    access="login",
    conditions=("attribution", "share-alike", "accept-kaggle-rules"),
    notes=("Labels released under CC0; comment text from Wikipedia talk pages. Sources disagree "
           "on the text license (CC0 vs CC BY-SA 3.0), so the stricter one is kept."),
  )

  def _read(self, name: str) -> pd.DataFrame:
    for archive in self.dataset_dir.glob("*.zip"):
      with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
          if Path(member).name in {f"{name}.csv", f"{name}.csv.zip"}:
            zf.extract(member, self.dataset_dir)
    hits = sorted(self.dataset_dir.rglob(f"{name}.csv")) or sorted(self.dataset_dir.rglob(f"{name}.csv.zip"))
    if not hits:
      raise FileNotFoundError(
        f"{name}.csv not found in {self.dataset_dir}. Accept the rules of the Kaggle competition "
        f"'jigsaw-toxic-comment-classification-challenge' and download its files there.")
    return pd.read_csv(hits[0])

  def _download_and_load(self) -> pd.DataFrame:
    train = self._read("train").assign(split="train")
    test = self._read("test").merge(self._read("test_labels"), on="id").assign(split="test")
    return pd.concat([train, test[test[self.labels[0]] >= 0]], ignore_index=True)

  def _preprocess(self, raw_data):
    return raw_data

  def _split_features_and_target(self, data):
    return data["comment_text"].tolist(), (data[self.labels].to_numpy() > 0)

  def _label_names(self, data) -> list[str]:
    return list(self.labels)

  def _split(self, data):
    return data["split"].to_numpy()


class MultilabelLoaderFactory:
  """Factory used to instantiate the correct multi-label loader by dataset key.

  Includes the datasets used in [1] that can be loaded without new
  dependencies, plus Portuguese and English toxicity datasets. Every loader
  declares its license in the `license` attribute.

  Refs
  [1] Moreo, A., Francisco, M., & Sebastiani, F. (2023). Multi-Label Quantification.
      ACM Transactions on Knowledge Discovery from Data, 18(1).
  """
  _registry = {
    "told-br": ToLDBrLoader,
    "goemotions": GoEmotionsLoader,
    "reuters21578": Reuters21578Loader,
    "reuters21578-modapte": Reuters21578ModApteLoader,
    "jigsaw-toxic": JigsawToxicLoader,
    **{subset: _mulan_loader(subset, name) for subset, name in MULAN_DATASETS.items()},
  }

  @classmethod
  def get_loader(cls, dataset_name: str, **kwargs) -> BaseMultilabelDatasetLoader:
    pipeline_class = cls._registry.get(dataset_name.lower())
    if not pipeline_class:
      raise ValueError(f"Dataset '{dataset_name}' not supported on multi-label datasets.")
    return pipeline_class(**kwargs)


MULTILABEL_DATASETS: list[str] = list(MultilabelLoaderFactory._registry.keys())


def load_multilabel(dataset: str,
                    data_home: str | Path | None = None,
                    force_load: bool = False,
                    accept_terms: bool = False,
                    return_metadata: bool = False,
                    **kwargs) -> tuple:
  """Load a multi-label dataset by name using the registered loader factory.

  Downloads the files (if not cached under `data_home`, verifying their
  checksums), preprocesses them and returns the instances and the binary
  label matrix.

  Parameters
  ----------
  dataset : str
    Dataset key. Must be one of `quack.datasets.MULTILABEL_DATASETS`.
  data_home : str or Path, default = None
    Root folder of the cached files. Defaults to `~/.quack_data/`.
  force_load : bool, default = False
    Download the files again even if they are cached.
  accept_terms : bool, default = False
    Required for datasets whose license limits use (research only,
    non-commercial, login). Passing True states that the conditions shown in
    the error message were read.
  return_metadata : bool, default = False
    Also return a `DatasetMetadata` with label names, official split,
    natural groups and license.
  **kwargs
    Dataset-specific options, passed to the loader (e.g. `min_votes` for
    `"told-br"`, `min_train_positives` for Reuters and Mulan datasets).

  Returns
  -------
  X : ndarray
    Features as float32, or an object array of texts for text datasets.
  Y : ndarray of shape (n_samples, n_labels)
    Binary indicator matrix (int8).
  metadata : DatasetMetadata
    Only when `return_metadata=True`.

  Raises
  ------
  ValueError
    If `dataset` is not a registered key.
  TermsNotAcceptedError
    If the dataset has use conditions and `accept_terms` is False.
  FileNotFoundError
    If the dataset must be downloaded by hand and its files are missing.

  Examples
  --------
  >>> from quack.datasets import load_multilabel
  >>> X, Y, meta = load_multilabel("told-br", return_metadata=True)
  >>> dict(zip(meta.label_names, Y.mean(axis=0)))
  """
  loader = MultilabelLoaderFactory.get_loader(dataset, data_home=data_home, force_load=force_load, **kwargs)
  check_terms(loader.license, accept_terms, dataset)
  X, Y = loader.load_dataset()
  if return_metadata:
    return X, Y, loader.metadata()
  return X, Y

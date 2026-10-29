# tests/test_multilabel_datasets.py
import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quack.datasets import MULTILABEL_DATASETS, DatasetMetadata, TermsNotAcceptedError, load_multilabel
from quack.datasets import utils as dataset_utils
from quack.datasets._multilabel_datasets import (
  GOEMOTIONS_FILES, MULAN_DATASETS, JigsawToxicLoader, MulanLoader, MultilabelLoaderFactory,
  Reuters21578Loader, ToLDBrLoader, parse_reuters_sgml)
from quack.datasets.utils import BaseMultilabelDatasetLoader

GOEMOTIONS_FIXTURE = {
  "emotions.txt": "admiration\nanger\njoy\nsurprise\nneutral\n",
  "train.tsv": "great job\t0,2\tc1\nso mad\t1\tc2\n",
  "dev.tsv": "whatever\t4\tc3\n",
  "test.tsv": "wow\t3\tc4\n",
  "ekman_mapping.json": json.dumps({"anger": ["anger"], "joy": ["admiration", "joy"], "surprise": ["surprise"]}),
  "sentiment_mapping.json": json.dumps({"positive": ["admiration", "joy"], "negative": ["anger"],
                                        "ambiguous": ["surprise"]}),
}

SGML = """<!DOCTYPE lewis SYSTEM "lewis.dtd">
<REUTERS TOPICS="YES" LEWISSPLIT="TRAIN" CGISPLIT="TRAINING-SET" OLDID="5544" NEWID="1">
<TOPICS><D>cocoa</D></TOPICS><TEXT><TITLE>COCOA RAIN</TITLE><BODY>Showers continued.</BODY></TEXT>
</REUTERS>
<REUTERS TOPICS="NO" LEWISSPLIT="TRAIN" CGISPLIT="TRAINING-SET" OLDID="5545" NEWID="2">
<TOPICS></TOPICS><TEXT><BODY>No topics field.</BODY></TEXT>
</REUTERS>
<REUTERS TOPICS="YES" LEWISSPLIT="TEST" CGISPLIT="TRAINING-SET" OLDID="5546" NEWID="3">
<TOPICS></TOPICS><TEXT><TITLE>NOTHING</TITLE><BODY>Topics=YES but empty.</BODY></TEXT>
</REUTERS>
<REUTERS TOPICS="YES" LEWISSPLIT="NOT-USED" CGISPLIT="TRAINING-SET" OLDID="5547" NEWID="4">
<TOPICS><D>earn</D></TOPICS><TEXT><BODY>Not used.</BODY></TEXT>
</REUTERS>
<REUTERS TOPICS="YES" LEWISSPLIT="TEST" CGISPLIT="TRAINING-SET" OLDID="5548" NEWID="5">
<TOPICS><D>earn</D><D>acq</D></TOPICS><TEXT><TITLE>PROFIT</TITLE><BODY>Up.</BODY></TEXT>
</REUTERS>
"""

MULAN_ARFF = """@relation emotions
@attribute f1 numeric
@attribute f2 numeric
@attribute happy {0,1}
@attribute sad {0,1}
@data
"""

MULAN_XML = """<?xml version="1.0" encoding="utf-8"?>
<labels xmlns="http://mulan.sourceforge.net/labels"><label name="happy"></label><label name="sad"></label></labels>
"""


def zip_bytes(members: dict) -> bytes:
  buffer = io.BytesIO()
  with zipfile.ZipFile(buffer, "w") as zf:
    for name, content in members.items():
      zf.writestr(name, content)
  return buffer.getvalue()


def tar_bytes(members: dict) -> bytes:
  buffer = io.BytesIO()
  with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
    for name, content in members.items():
      data = content.encode() if isinstance(content, str) else content
      info = tarfile.TarInfo(name)
      info.size = len(data)
      tf.addfile(info, io.BytesIO(data))
  return buffer.getvalue()


class FakeDownloads:
  """Replaces `download_file` with a function that writes local fixtures."""

  def __init__(self):
    self.files: dict[str, bytes] = {}
    self.calls: list[tuple[str, str | None]] = []

  def __call__(self, url, filename, exist_ok=False, checksum=None):
    self.calls.append((url, checksum))
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = self.files[path.name]
    path.write_bytes(content.encode() if isinstance(content, str) else content)


@pytest.fixture
def downloads(monkeypatch):
  fake = FakeDownloads()
  monkeypatch.setattr(dataset_utils, "download_file", fake)
  return fake


class DummyMultilabelLoader(BaseMultilabelDatasetLoader):
  dataset_name = "dummy"
  cache_dir = "dummy"

  def _download_and_load(self):
    return pd.DataFrame({"f": [1.0, 2.0, 3.0], "a": [1, 0, 1], "b": [0, 0, 1], "g": ["x", "x", "y"]})

  def _preprocess(self, raw_data):
    return raw_data

  def _split_features_and_target(self, data):
    return data[["f"]].values, data[["a", "b"]].values

  def _label_names(self, data):
    return ["a", "b"]

  def _groups(self, data):
    return data["g"].to_numpy()


class TestMultilabelLoaderFactory:
  def test_multilabel_datasets_matches_registry_keys(self):
    assert MULTILABEL_DATASETS == list(MultilabelLoaderFactory._registry.keys())
    assert set(MULAN_DATASETS) <= set(MULTILABEL_DATASETS)

  def test_get_loader_is_case_insensitive_and_passes_options(self, tmp_path):
    loader = MultilabelLoaderFactory.get_loader("ToLD-BR", data_home=tmp_path, min_votes=3)
    assert isinstance(loader, ToLDBrLoader)
    assert loader.min_votes == 3

  def test_get_loader_raises_for_invalid_key(self):
    with pytest.raises(ValueError, match="not supported"):
      MultilabelLoaderFactory.get_loader("not-a-real-dataset")

  def test_mulan_loaders_are_generated_per_subset(self, tmp_path):
    loader = MultilabelLoaderFactory.get_loader("tmc2007_500", data_home=tmp_path)
    assert isinstance(loader, MulanLoader)
    assert type(loader).__name__ == "Tmc2007500MulanLoader"
    assert loader.dataset_dir == tmp_path / "mulan" / "tmc2007_500"


class TestBaseMultilabelPipeline:
  def test_load_dataset_returns_dtypes_and_metadata(self, tmp_path):
    loader = DummyMultilabelLoader(data_home=tmp_path)
    X, Y = loader.load_dataset()
    assert X.dtype == np.float32 and X.shape == (3, 1)
    assert Y.dtype == np.int8 and Y.shape == (3, 2)
    meta = loader.metadata()
    assert isinstance(meta, DatasetMetadata)
    assert meta.label_names == ["a", "b"]
    np.testing.assert_array_equal(meta.groups, ["x", "x", "y"])
    assert meta.split is None

  def test_label_names_must_match_columns(self, tmp_path, monkeypatch):
    monkeypatch.setattr(DummyMultilabelLoader, "_label_names", lambda self, data: ["a"])
    with pytest.raises(ValueError, match="label names"):
      DummyMultilabelLoader(data_home=tmp_path).load_dataset()

  def test_groups_must_have_one_entry_per_instance(self, tmp_path, monkeypatch):
    monkeypatch.setattr(DummyMultilabelLoader, "_groups", lambda self, data: ["x"])
    with pytest.raises(ValueError, match="group"):
      DummyMultilabelLoader(data_home=tmp_path).load_dataset()


class TestToLDBr:
  CSV = ("text,homophobia,obscene,insult,racism,misogyny,xenophobia\n"
         "tweet one,0,2,3,0,0,0\n"
         "tweet two,1,0,0,0,0,0\n")

  def test_majority_vote_and_soft_labels(self, tmp_path, downloads):
    downloads.files["ToLD-BR.csv"] = self.CSV
    X, Y, meta = load_multilabel("told-br", data_home=tmp_path, return_metadata=True)
    assert X.dtype == object and list(X) == ["tweet one", "tweet two"]
    assert meta.label_names == ToLDBrLoader.labels
    np.testing.assert_array_equal(Y, [[0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(meta.extras["votes"][0], [0, 2, 3, 0, 0, 0])
    assert downloads.calls[0][1] == ToLDBrLoader.checksum

  def test_min_votes(self, tmp_path, downloads):
    downloads.files["ToLD-BR.csv"] = self.CSV
    _, Y = load_multilabel("told-br", data_home=tmp_path, min_votes=1)
    assert Y[1, 0] == 1


class TestGoEmotions:
  def test_flat_labels_with_official_split(self, tmp_path, downloads):
    downloads.files.update(GOEMOTIONS_FIXTURE)
    X, Y, meta = load_multilabel("goemotions", data_home=tmp_path, return_metadata=True)
    assert meta.label_names == ["admiration", "anger", "joy", "surprise", "neutral"]
    np.testing.assert_array_equal(Y[0], [1, 0, 1, 0, 0])
    np.testing.assert_array_equal(meta.split, ["train", "train", "val", "test"])
    assert {name for name, _ in GOEMOTIONS_FILES.items()} == set(GOEMOTIONS_FIXTURE)
    assert all(checksum.startswith("sha256:") for _, checksum in downloads.calls)


class TestReuters:
  def test_parse_reuters_sgml_keeps_modapte_documents(self):
    docs = list(parse_reuters_sgml(SGML))
    assert [(split, topics) for split, topics, _ in docs] == [
      ("train", ["cocoa"]), ("test", []), ("test", ["earn", "acq"])]
    assert docs[0][2] == "COCOA RAIN Showers continued."

  def test_requires_accepting_research_only_terms(self, tmp_path):
    with pytest.raises(TermsNotAcceptedError, match="research"):
      load_multilabel("reuters21578", data_home=tmp_path)

  def test_nltk_copy(self, tmp_path, downloads):
    downloads.files["reuters.zip"] = zip_bytes({
      "reuters/cats.txt": "training/1 earn acq\ntraining/2 earn\ntest/3 grain\n",
      "reuters/training/1": "first", "reuters/training/2": "second", "reuters/test/3": "third"})
    X, Y, meta = load_multilabel("reuters21578", data_home=tmp_path, accept_terms=True, return_metadata=True)
    assert meta.label_names == ["acq", "earn", "grain"]
    np.testing.assert_array_equal(Y, [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(meta.split, ["train", "train", "test"])
    assert downloads.calls[0][1] == Reuters21578Loader.checksum

  def test_min_train_positives_drops_rare_topics(self, tmp_path, downloads):
    downloads.files["reuters.zip"] = zip_bytes({
      "reuters/cats.txt": "training/1 earn acq\ntraining/2 earn\ntest/3 grain\n",
      "reuters/training/1": "first", "reuters/training/2": "second", "reuters/test/3": "third"})
    _, Y, meta = load_multilabel("reuters21578", data_home=tmp_path, accept_terms=True,
                                 return_metadata=True, min_train_positives=2)
    assert meta.label_names == ["earn"]
    assert Y.shape == (3, 1)

  def test_modapte_reads_local_sgml_archive(self, tmp_path):
    folder = tmp_path / "reuters21578"
    folder.mkdir()
    (folder / "reuters21578.tar.gz").write_bytes(tar_bytes({"reut2-000.sgm": SGML}))
    X, Y, meta = load_multilabel("reuters21578-modapte", data_home=tmp_path, accept_terms=True,
                                 return_metadata=True)
    assert len(X) == 3
    assert meta.label_names == ["acq", "cocoa", "earn"]
    assert Y[1].sum() == 0

  def test_modapte_without_archive_explains_where_to_get_it(self, tmp_path):
    with pytest.raises(FileNotFoundError, match="kdd.ics.uci.edu"):
      load_multilabel("reuters21578-modapte", data_home=tmp_path, accept_terms=True)


class TestMulan:
  def write(self, folder, name, rows):
    (folder / name).write_text(MULAN_ARFF + "\n".join(rows) + "\n", encoding="utf-8")

  def test_train_and_test_files(self, tmp_path):
    folder = tmp_path / "mulan" / "emotions"
    folder.mkdir(parents=True)
    self.write(folder, "emotions-train.arff", ["0.1,0.2,1,0", "0.3,0.4,1,1"])
    self.write(folder, "emotions-test.arff", ["0.5,0.6,0,1"])
    (folder / "emotions.xml").write_text(MULAN_XML, encoding="utf-8")
    X, Y, meta = load_multilabel("emotions", data_home=tmp_path, return_metadata=True)
    assert X.shape == (3, 2) and X.dtype == np.float32
    np.testing.assert_array_equal(Y, [[1, 0], [1, 1], [0, 1]])
    np.testing.assert_array_equal(meta.split, ["train", "train", "test"])
    assert meta.label_names == ["happy", "sad"]

  def test_min_train_positives(self, tmp_path):
    folder = tmp_path / "mulan" / "emotions"
    folder.mkdir(parents=True)
    self.write(folder, "emotions-train.arff", ["0.1,0.2,1,0", "0.3,0.4,1,1"])
    self.write(folder, "emotions-test.arff", ["0.5,0.6,0,1"])
    (folder / "emotions.xml").write_text(MULAN_XML, encoding="utf-8")
    _, Y, meta = load_multilabel("emotions", data_home=tmp_path, return_metadata=True, min_train_positives=2)
    assert meta.label_names == ["happy"]

  def test_missing_files_explain_where_to_get_them(self, tmp_path):
    with pytest.raises(FileNotFoundError, match="mulan.sourceforge.net"):
      load_multilabel("scene", data_home=tmp_path)


class TestJigsaw:
  def test_requires_accepting_kaggle_terms(self, tmp_path):
    with pytest.raises(TermsNotAcceptedError, match="Kaggle|account"):
      load_multilabel("jigsaw-toxic", data_home=tmp_path)

  def test_reads_competition_zip_and_drops_unscored_test_rows(self, tmp_path):
    labels = ",".join(JigsawToxicLoader.labels)
    folder = tmp_path / "jigsaw_toxic"
    folder.mkdir()
    (folder / "jigsaw.zip").write_bytes(zip_bytes({
      "train.csv": f"id,comment_text,{labels}\na,hello,0,0,0,0,0,0\nb,you idiot,1,0,0,0,1,0\n",
      "test.csv": "id,comment_text\nc,fine\nd,not scored\n",
      "test_labels.csv": f"id,{labels}\nc,0,0,0,0,0,0\nd,-1,-1,-1,-1,-1,-1\n",
    }))
    X, Y, meta = load_multilabel("jigsaw-toxic", data_home=tmp_path, accept_terms=True, return_metadata=True)
    assert list(X) == ["hello", "you idiot", "fine"]
    np.testing.assert_array_equal(Y[1], [1, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(meta.split, ["train", "train", "test"])

  def test_missing_files_explain_how_to_get_them(self, tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
      load_multilabel("jigsaw-toxic", data_home=tmp_path, accept_terms=True)


class TestLoadMultilabel:
  def test_delegates_to_factory(self, tmp_path, monkeypatch):
    dummy = DummyMultilabelLoader(data_home=tmp_path)
    dummy.license = ToLDBrLoader.license
    monkeypatch.setattr(MultilabelLoaderFactory, "get_loader", classmethod(lambda cls, name, **kw: dummy))
    X, Y = load_multilabel("anything")
    assert X.shape == (3, 1) and Y.shape == (3, 2)

  def test_raises_for_invalid_dataset(self):
    with pytest.raises(ValueError, match="not supported"):
      load_multilabel("not-a-real-dataset")

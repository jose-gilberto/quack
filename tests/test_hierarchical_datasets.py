# tests/test_hierarchical_datasets.py
import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.utils import Bunch

from quack.datasets import HIERARCHICAL_DATASETS, ROOT, Hierarchy, TermsNotAcceptedError, load_hierarchical
from quack.datasets import utils as dataset_utils
from quack.datasets._hierarchical_datasets import (
  CLUS_DATASETS, ClusHMCLoader, HierarchicalLoaderFactory, RCV1Loader, _extract_once, arxiv_archive,
  newsgroup_path, parse_rcv1_hierarchy)
from quack.datasets.utils import BaseHierarchicalDatasetLoader

GOEMOTIONS_FIXTURE = {
  "emotions.txt": "admiration\nanger\njoy\nsurprise\nneutral\n",
  "train.tsv": "great job\t0,2\tc1\nso mad\t1\tc2\n",
  "dev.tsv": "whatever\t4\tc3\n",
  "test.tsv": "wow\t3\tc4\n",
  "ekman_mapping.json": json.dumps({"anger": ["anger"], "joy": ["admiration", "joy"], "surprise": ["surprise"]}),
  "sentiment_mapping.json": json.dumps({"positive": ["admiration", "joy"], "negative": ["anger"],
                                        "ambiguous": ["surprise"]}),
}

CLUS_FUN = """@RELATION cellcycle_FUN
@ATTRIBUTE a numeric
@ATTRIBUTE class hierarchical 01,01/01,01/02,02
@DATA
{rows}
"""

CLUS_GO = """@RELATION church_GO
@ATTRIBUTE a numeric
@ATTRIBUTE class hierarchical root/GO1,root/GO2,GO1/GO3,GO2/GO3
@DATA
{rows}
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


class DummyHierarchicalLoader(BaseHierarchicalDatasetLoader):
  dataset_name = "dummy"
  cache_dir = "dummy"

  def _download_and_load(self):
    return {"X": [[1.0], [2.0]], "labels": [["a/x"], ["b"]]}

  def _preprocess(self, raw_data):
    return raw_data

  def _split_features_and_target(self, data):
    return data["X"], data["labels"]

  def _build_hierarchy(self, data):
    return Hierarchy([(ROOT, "a"), (ROOT, "b"), ("a", "a/x")])


class TestHierarchicalLoaderFactory:
  def test_hierarchical_datasets_matches_registry_keys(self):
    assert HIERARCHICAL_DATASETS == list(HierarchicalLoaderFactory._registry.keys())
    for name in CLUS_DATASETS:
      assert f"{name}-fun" in HIERARCHICAL_DATASETS and f"{name}-go" in HIERARCHICAL_DATASETS

  def test_get_loader_is_case_insensitive_and_passes_options(self, tmp_path):
    loader = HierarchicalLoaderFactory.get_loader("MultiEURLEX", data_home=tmp_path, language="en")
    assert loader.language == "en"

  def test_clus_loaders_are_generated_per_dataset_and_ontology(self, tmp_path):
    loader = HierarchicalLoaderFactory.get_loader("cellcycle-go", data_home=tmp_path)
    assert isinstance(loader, ClusHMCLoader)
    assert type(loader).__name__ == "CellcycleGoLoader"
    assert (loader.name, loader.ontology) == ("cellcycle", "GO")

  def test_get_loader_raises_for_invalid_key(self):
    with pytest.raises(ValueError, match="not supported"):
      HierarchicalLoaderFactory.get_loader("not-a-real-dataset")


class TestBaseHierarchicalPipeline:
  def test_load_dataset_closes_labels_upward(self, tmp_path):
    loader = DummyHierarchicalLoader(data_home=tmp_path)
    X, Y, hierarchy = loader.load_dataset()
    assert X.dtype == np.float32
    assert hierarchy.nodes == ["a", "b", "a/x"]
    np.testing.assert_array_equal(Y, [[1, 0, 1], [0, 1, 0]])
    assert loader.metadata().label_names == hierarchy.nodes


class TestExtractOnce:
  def test_extracts_once(self, tmp_path):
    archive = tmp_path / "a.zip"
    archive.write_bytes(zip_bytes({"inner/file.txt": "hi"}))
    dest = _extract_once(archive, tmp_path / "out")
    assert (dest / "inner" / "file.txt").read_text() == "hi"
    (dest / "inner" / "file.txt").unlink()
    _extract_once(archive, tmp_path / "out")
    assert not (dest / "inner" / "file.txt").exists()

  def test_rejects_members_outside_destination(self, tmp_path):
    archive = tmp_path / "evil.tar.gz"
    archive.write_bytes(tar_bytes({"../escape.txt": "x"}))
    with pytest.raises(ValueError, match="outside"):
      _extract_once(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


class TestGoEmotionsHierarchical:
  def test_sentiment_ekman_emotion_tree(self, tmp_path, downloads):
    downloads.files.update(GOEMOTIONS_FIXTURE)
    X, Y, h, meta = load_hierarchical("goemotions", data_home=tmp_path, return_metadata=True)
    assert h.n == 11 and h.is_tree
    assert set(h.children()) == {"sentiment:positive", "sentiment:negative", "sentiment:ambiguous", "neutral"}
    assert h.parents("admiration") == ["ekman:joy"]
    assert h.parents("ekman:joy") == ["sentiment:positive"]
    row = dict(zip(h.nodes, Y[0]))
    assert row["sentiment:positive"] == row["ekman:joy"] == row["joy"] == row["admiration"] == 1
    assert h.constraint(Y) == "inequality"  # first comment has two emotions
    np.testing.assert_array_equal(meta.split, ["train", "train", "val", "test"])

  def test_mapping_must_cover_every_emotion(self, tmp_path, downloads):
    downloads.files.update(GOEMOTIONS_FIXTURE)
    downloads.files["ekman_mapping.json"] = json.dumps({"anger": ["anger"], "joy": ["joy"]})
    with pytest.raises(ValueError, match="exactly once"):
      load_hierarchical("goemotions", data_home=tmp_path)


class TestTwentyNewsgroups:
  def test_newsgroup_path(self):
    assert newsgroup_path("comp.sys.mac.hardware") == ("comp", "comp.sys.mac.hardware")
    assert newsgroup_path("comp.sys.mac.hardware", "full") == (
      "comp", "comp.sys", "comp.sys.mac", "comp.sys.mac.hardware")
    with pytest.raises(ValueError):
      newsgroup_path("comp.graphics", "middle")

  def test_hierarchy_from_group_names(self, tmp_path, monkeypatch):
    import sklearn.datasets

    names = ["comp.graphics", "comp.sys.mac.hardware", "sci.space"]

    def fake_fetch(data_home, subset, remove):
      target = np.array([0, 1, 2]) if subset == "train" else np.array([2])
      return Bunch(data=[f"{subset} {i}" for i in range(len(target))], target=target, target_names=names)

    monkeypatch.setattr(sklearn.datasets, "fetch_20newsgroups", fake_fetch)
    X, Y, h, meta = load_hierarchical("20newsgroups", data_home=tmp_path, return_metadata=True)
    assert h.children() == ["comp", "sci"]
    assert h.constraint(Y) == "equality"
    np.testing.assert_array_equal(meta.split, ["train"] * 3 + ["test"])
    _, _, full = load_hierarchical("20newsgroups", data_home=tmp_path, depth="full")
    assert "comp.sys.mac" in full


class TestRCV1:
  HIER = ("parent: Root      child: CCAT  child-description: CORPORATE/INDUSTRIAL\n"
          "parent: Root      child: GCAT  child-description: GOVERNMENT/SOCIAL\n"
          "parent: CCAT      child: C15   child-description: PERFORMANCE\n"
          "parent: C15       child: C151  child-description: ACCOUNTS/EARNINGS\n"
          "parent: GCAT      child: GSPO  child-description: SPORTS\n")

  def test_parse_rcv1_hierarchy(self):
    edges = parse_rcv1_hierarchy(self.HIER)
    assert edges[0] == (ROOT, "CCAT")
    assert ("C15", "C151") in edges and len(edges) == 5

  def test_requires_accepting_restricted_terms(self, tmp_path):
    with pytest.raises(TermsNotAcceptedError, match="NIST"):
      load_hierarchical("rcv1", data_home=tmp_path)

  def test_sparse_vectors_with_topic_hierarchy(self, tmp_path, downloads, monkeypatch):
    import sklearn.datasets

    downloads.files["rcv1.topics.hier.orig"] = self.HIER
    target = sp.csr_matrix(np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0]]))
    bunch = Bunch(data=sp.random(3, 5, density=0.5, format="csr", random_state=0), target=target,
                  target_names=np.array(["C15", "C151", "CCAT", "GCAT"], dtype=object))
    monkeypatch.setattr(sklearn.datasets, "fetch_rcv1", lambda data_home: bunch)
    monkeypatch.setattr(RCV1Loader, "n_train", 2)
    X, Y, h, meta = load_hierarchical("rcv1", data_home=tmp_path, accept_terms=True, return_metadata=True)
    assert sp.issparse(X) and X.dtype == np.float32
    assert set(h.nodes) == {"C15", "C151", "CCAT", "GCAT"}  # GSPO has no documents and is dropped
    assert dict(zip(h.nodes, Y[0]))["CCAT"] == 1  # closed upward from C151
    np.testing.assert_array_equal(meta.split, ["train", "train", "test"])


class TestPathDatasets:
  def test_dbpedia(self, tmp_path, downloads):
    header = "text,l1,l2,l3\n"
    downloads.files.update({
      "DBPEDIA_train.csv": header + "a footballer,Agent,Athlete,SoccerPlayer\na river,Place,BodyOfWater,River\n",
      "DBPEDIA_val.csv": header + "a striker,Agent,Athlete,SoccerPlayer\n",
      "DBPEDIA_test.csv": header + "a lake,Place,BodyOfWater,Lake\n"})
    X, Y, h, meta = load_hierarchical("dbpedia", data_home=tmp_path, return_metadata=True)
    assert h.level(1) == ["Agent", "Place"]
    assert "Place/BodyOfWater/Lake" in h
    assert h.constraint(Y) == "equality"
    np.testing.assert_array_equal(meta.split, ["train", "train", "val", "test"])

  def test_amazon_joins_title_and_text(self, tmp_path, downloads):
    header = "productId,Title,userId,Helpfulness,Score,Time,Text,Cat1,Cat2,Cat3\n"
    downloads.files.update({
      "train_40k.csv": header + "p1,Nice toy,u1,0/0,5,1,Kids love it,toys games,games,puzzles\n",
      "val_10k.csv": header + "p2,Good food,u2,0/0,4,1,Tasty,grocery,snacks,chips\n"})
    X, Y, h = load_hierarchical("amazon-hier", data_home=tmp_path)
    assert X[0] == "Nice toy. Kids love it"
    assert h.level(1) == ["grocery", "toys games"]
    assert all(checksum.startswith("md5:") for _, checksum in downloads.calls)

  def test_web_of_science_nested_zip(self, tmp_path, downloads):
    inner = zip_bytes({"X.txt": "doc one\ndoc two\ndoc three\n", "Y.txt": "0\n1\n2\n", "YL1.txt": "3\n3\n4\n"})
    downloads.files["WebOfScience.zip"] = zip_bytes({"WebOfScience/WOS5736.zip": inner})
    X, Y, h = load_hierarchical("wos", data_home=tmp_path, variant="WOS5736")
    assert list(X) == ["doc one", "doc two", "doc three"]
    assert h.children() == ["domain:3", "domain:4"]
    assert h.parents("area:1") == ["domain:3"]
    assert h.constraint(Y) == "equality"

  def test_web_of_science_rejects_area_under_two_domains(self, tmp_path, downloads):
    inner = zip_bytes({"X.txt": "a\nb\n", "Y.txt": "0\n0\n", "YL1.txt": "3\n4\n"})
    downloads.files["WebOfScience.zip"] = zip_bytes({"WebOfScience/WOS5736.zip": inner})
    with pytest.raises(ValueError, match="more than one domain"):
      load_hierarchical("wos", data_home=tmp_path, variant="WOS5736")


class TestMultiEURLEX:
  def doc(self, celex, texts, l1, l2, l3):
    return json.dumps({"celex_id": celex, "text": texts,
                       "eurovoc_concepts": {"level_1": l1, "level_2": l2, "level_3": l3}})

  def test_levels_language_and_years(self, tmp_path, downloads):
    train = "\n".join([
      self.doc("31990R0001", {"pt": "lei um", "en": "law one"}, ["100"], ["1001"], ["10011"]),
      self.doc("31995R0002", {"en": "law two"}, ["100"], ["1001"], ["10012"]),  # no Portuguese text
      self.doc("31999R0003", {"pt": "lei tres"}, ["100", "200"], ["1001", "2001"], ["10012", "20011"]),
    ])
    downloads.files["multi_eurlex.tar.gz"] = tar_bytes({
      "multi_eurlex/train.jsonl": train,
      "multi_eurlex/dev.jsonl": self.doc("32011R0004", {"pt": "lei quatro"}, ["200"], ["2001"], ["20011"]),
      "multi_eurlex/test.jsonl": self.doc("32014R0005", {"pt": "lei cinco"}, ["100"], ["1001"], ["10011"])})
    X, Y, h, meta = load_hierarchical("multieurlex", data_home=tmp_path, return_metadata=True)
    assert list(X) == ["lei um", "lei tres", "lei quatro", "lei cinco"]
    assert h.parents("10011") == ["1001"]
    assert h.parents("1001") == ["100"]
    # 10012 only appears (in Portuguese) in a law that also has 2001, so both are kept as parents
    assert h.parents("10012") == ["1001", "2001"]
    assert int(Y.sum()) == 3 + 6 + 3 + 3  # closure adds nothing: every level is in the files
    assert h.constraint(Y) == "inequality"
    np.testing.assert_array_equal(meta.groups, [1990, 1999, 2011, 2014])
    np.testing.assert_array_equal(meta.split, ["train", "train", "val", "test"])
    _, _, h_en = load_hierarchical("multieurlex", data_home=tmp_path, language="en")
    assert "20011" not in h_en


class TestArxiv:
  def paper(self, pid, cats, created):
    return json.dumps({"id": pid, "title": "A  title", "abstract": " An\nabstract. ", "categories": cats,
                       "versions": [{"version": "v1", "created": created}], "update_date": "2024-01-01"})

  def write_snapshot(self, tmp_path):
    folder = tmp_path / "arxiv"
    folder.mkdir()
    (folder / "arxiv-metadata-oai-snapshot.json").write_text("\n".join([
      self.paper("1", "cs.LG stat.ML", "Mon, 2 Apr 2007 19:18:42 GMT"),
      self.paper("2", "math.ST", "Tue, 3 Apr 2007 10:00:00 GMT"),
      self.paper("3", "cs.CL", "Wed, 1 Jan 2020 10:00:00 GMT"),
      self.paper("4", "stat.ME math.ST", "Thu, 2 Jan 2020 10:00:00 GMT"),
    ]) + "\n", encoding="utf-8")

  def test_requires_accepting_kaggle_terms(self, tmp_path):
    with pytest.raises(TermsNotAcceptedError):
      load_hierarchical("arxiv", data_home=tmp_path)

  def test_archive_category_hierarchy_and_years(self, tmp_path):
    self.write_snapshot(tmp_path)
    X, Y, h, meta = load_hierarchical("arxiv", data_home=tmp_path, accept_terms=True, return_metadata=True)
    assert X[0] == "A title. An abstract."
    assert h.children() == ["cs", "stat"]
    assert h.parents("stat.ML") == ["stat"]
    assert "math.ST" not in h
    np.testing.assert_array_equal(meta.groups, [2007, 2020, 2020])
    np.testing.assert_array_equal(meta.extras["arxiv_id"], ["1", "3", "4"])
    assert arxiv_archive("hep-th") == "hep-th"

  def test_filters(self, tmp_path):
    self.write_snapshot(tmp_path)
    X, _, _ = load_hierarchical("arxiv", data_home=tmp_path, accept_terms=True, years=(2020, 2020))
    assert len(X) == 2
    X, _, _ = load_hierarchical("arxiv", data_home=tmp_path, accept_terms=True, max_docs=1)
    assert len(X) == 1

  def test_missing_snapshot_explains_where_to_get_it(self, tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
      load_hierarchical("arxiv", data_home=tmp_path, accept_terms=True)


class TestClusHMC:
  def test_funcat_tree_from_zip(self, tmp_path):
    folder = tmp_path / "clus_hmc"
    folder.mkdir()
    (folder / "FunCat.zip").write_bytes(zip_bytes({
      "datasets_FUN/cellcycle_FUN/cellcycle_FUN.train.arff": CLUS_FUN.format(rows="1.0,01/01@02\n2.0,01/02"),
      "datasets_FUN/cellcycle_FUN/cellcycle_FUN.valid.arff": CLUS_FUN.format(rows="3.0,02"),
      "datasets_FUN/cellcycle_FUN/cellcycle_FUN.test.arff": CLUS_FUN.format(rows="?,01/01"),
    }))
    X, Y, h, meta = load_hierarchical("cellcycle-fun", data_home=tmp_path, return_metadata=True)
    assert X.shape == (4, 1) and np.isnan(X[3, 0])
    assert h.is_tree and h.parents("01/02") == ["01"]
    assert dict(zip(h.nodes, Y[0])) == {"01": 1, "02": 1, "01/01": 1, "01/02": 0}
    np.testing.assert_array_equal(meta.split, ["train", "train", "val", "test"])

  def test_gene_ontology_dag(self, tmp_path):
    folder = tmp_path / "clus_hmc" / "GO"
    folder.mkdir(parents=True)
    for tag, rows in (("train", "1.0,GO3"), ("valid", "2.0,GO1"), ("test", "3.0,GO2")):
      (folder / f"church_GO.{tag}.arff").write_text(CLUS_GO.format(rows=rows), encoding="utf-8")
    X, Y, h = load_hierarchical("church-go", data_home=tmp_path)
    assert not h.is_tree
    assert Y[0].sum() == 3  # GO3 closes upward through both parents

  def test_missing_files_explain_where_to_get_them(self, tmp_path):
    with pytest.raises(FileNotFoundError, match="dtai.cs.kuleuven.be"):
      load_hierarchical("seq-fun", data_home=tmp_path)


class TestLoadHierarchical:
  def test_raises_for_invalid_dataset(self):
    with pytest.raises(ValueError, match="not supported"):
      load_hierarchical("not-a-real-dataset")

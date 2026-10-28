import json
import re
import tarfile
import zipfile
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from quack.datasets._arff import read_clus_hmc
from quack.datasets._hierarchy import ROOT, Hierarchy
from quack.datasets._licenses import DatasetLicense, check_terms
from quack.datasets._multilabel_datasets import (
  GOEMOTIONS_FILES, GOEMOTIONS_LICENSE, _GOEMOTIONS_URL, read_goemotions)
from quack.datasets.utils import BaseHierarchicalDatasetLoader


def _extract_once(archive: Path, dest: Path) -> Path:
  """Extract a zip/tar archive into `dest` the first time it is needed."""
  marker = dest / ".extracted"
  if marker.exists():
    return dest
  dest.mkdir(parents=True, exist_ok=True)
  root = dest.resolve()
  if zipfile.is_zipfile(archive):
    with zipfile.ZipFile(archive) as zf:
      names = zf.namelist()
      _check_members(root, names)
      zf.extractall(dest)
  else:
    with tarfile.open(archive) as tf:
      _check_members(root, tf.getnames())
      try:
        tf.extractall(dest, filter="data")
      except TypeError:  # Python without tarfile extraction filters
        tf.extractall(dest)
  marker.write_text(archive.name)
  return dest


def _check_members(root: Path, names: Iterable[str]) -> None:
  for name in names:
    target = (root / name).resolve()
    if target != root and root not in target.parents:
      raise ValueError(f"archive member {name!r} would be written outside {root}")


class GoEmotionsHierarchicalLoader(BaseHierarchicalDatasetLoader):
  """GoEmotions with the sentiment > Ekman group > emotion hierarchy.

  The authors' mapping files group the 27 emotions into Ekman's six basic
  emotions, and these into positive, negative and ambiguous sentiment;
  `neutral` is a separate top-level class (37 classes in total). Comments
  can carry several emotions, so the constraint between levels is an
  inequality.

  References
  ----------
  [1] Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., &
      Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. ACL 2020.
  """
  dataset_name = "GoEmotions (hierarchical)"
  cache_dir = "goemotions"
  is_text = True
  license = GOEMOTIONS_LICENSE

  def _download_and_load(self) -> pd.DataFrame:
    for filename, checksum in GOEMOTIONS_FILES.items():
      self._fetch(_GOEMOTIONS_URL + filename, filename, checksum)
    data, self._emotions = read_goemotions(self.dataset_dir)
    self._ekman = json.loads((self.dataset_dir / "ekman_mapping.json").read_text(encoding="utf-8"))
    self._sentiment = json.loads((self.dataset_dir / "sentiment_mapping.json").read_text(encoding="utf-8"))
    return data

  def _preprocess(self, raw_data):
    return raw_data

  def _split_features_and_target(self, data):
    return data["text"].tolist(), data["labels"].tolist()

  def _build_hierarchy(self, data) -> Hierarchy:
    edges = [(ROOT, f"sentiment:{s}") for s in self._sentiment] + [(ROOT, "neutral")]
    covered: list[str] = []
    for group, members in self._ekman.items():
      owners = [s for s, ms in self._sentiment.items() if set(members) <= set(ms)]
      if len(owners) != 1:
        raise ValueError(f"Ekman group {group!r} does not nest in exactly one sentiment: {owners}")
      edges.append((f"sentiment:{owners[0]}", f"ekman:{group}"))
      edges += [(f"ekman:{group}", m) for m in members]
      covered += members
    missing = set(self._emotions) - set(covered) - {"neutral"}
    if missing or len(covered) != len(set(covered)):
      raise ValueError(f"mapping files do not cover each emotion exactly once (missing {sorted(missing)})")
    return Hierarchy(edges)

  def _split(self, data):
    return data["split"].to_numpy()


def newsgroup_path(name: str, depth: str = "top") -> tuple[str, ...]:
  """Path of a newsgroup in the hierarchy implied by its name.

  `"comp.sys.mac.hardware"` gives `("comp", "comp.sys.mac.hardware")` with
  `depth="top"`, or every dotted prefix with `depth="full"`.
  """
  parts = name.split(".")
  if depth == "top":
    return (parts[0], name)
  if depth == "full":
    return tuple(".".join(parts[:i + 1]) for i in range(len(parts)))
  raise ValueError("depth must be 'top' or 'full'")


class TwentyNewsgroupsLoader(BaseHierarchicalDatasetLoader):
  """20 Newsgroups (by-date split) with the hierarchy implied by the group names.

  Downloaded through `sklearn.datasets.fetch_20newsgroups`. With
  `depth="top"` the 20 groups sit under 7 top-level groups (alt, comp, misc,
  rec, sci, soc, talk); every message has one path, so the constraint
  between levels is an equality.

  Parameters
  ----------
  depth : {"top", "full"}, default = "top"
    `"full"` turns every dotted prefix (`comp.sys`, `comp.sys.ibm`...) into a class.
  remove : tuple of str, default = ("headers", "footers", "quotes")
    Parts of each message to strip, as in `fetch_20newsgroups`.

  References
  ----------
  [1] Lang, K. (1995). NewsWeeder: Learning to filter netnews. ICML 1995.
  """
  dataset_name = "20 Newsgroups"
  cache_dir = "twenty_newsgroups"
  is_text = True
  license = DatasetLicense(
    spdx="NOASSERTION",
    redistribution="not-declared",
    source="http://qwone.com/~jason/20Newsgroups/",
    checked="2026-10-27",
    citation="Lang, K. NewsWeeder: Learning to filter netnews. ICML 1995.",
    conditions=("cite",),
    notes="Downloaded by scikit-learn from its figshare mirror.",
  )

  def __init__(self, data_home=None, force_load: bool = False, depth: str = "top",
               remove: tuple[str, ...] = ("headers", "footers", "quotes")):
    super().__init__(data_home, force_load)
    self.depth = depth
    self.remove = remove

  def _download_and_load(self) -> pd.DataFrame:
    from sklearn.datasets import fetch_20newsgroups

    parts = []
    for subset in ("train", "test"):
      bunch = fetch_20newsgroups(data_home=str(self.dataset_dir), subset=subset, remove=self.remove)
      parts.append(pd.DataFrame({"text": bunch.data, "group": [bunch.target_names[t] for t in bunch.target],
                                 "split": subset}))
    return pd.concat(parts, ignore_index=True)

  def _preprocess(self, raw_data):
    return raw_data

  def _split_features_and_target(self, data):
    return data["text"].tolist(), [[g] for g in data["group"]]

  def _build_hierarchy(self, data) -> Hierarchy:
    edges = set()
    for group in data["group"].unique():
      parent = ROOT
      for node in newsgroup_path(group, self.depth):
        edges.add((parent, node))
        parent = node
    return Hierarchy(sorted(edges, key=lambda e: e[1]))

  def _split(self, data):
    return data["split"].to_numpy()


_RCV1_EDGE = re.compile(r"parent:\s*(\S+)\s+child:\s*(\S+)")


def parse_rcv1_hierarchy(text: str) -> list[tuple[str, str]]:
  """Edges of `rcv1.topics.hier.orig` (`parent: X  child: Y  child-description: ...`)."""
  edges = []
  for line in text.splitlines():
    match = _RCV1_EDGE.search(line)
    if match:
      parent, child = match.groups()
      edges.append((ROOT if parent.lower() == "root" else parent, child))
  return edges


class RCV1Loader(BaseHierarchicalDatasetLoader):
  """RCV1-v2: 804,414 Reuters news stories, 103 topics in a 4-root hierarchy.

  The tf-idf vectors of [1] are downloaded through `sklearn.datasets.fetch_rcv1`
  and returned as a sparse float32 matrix; the topic hierarchy comes from the
  paper's appendix. The official split is chronological: the first 23,149
  documents are the training set.

  The story text is copyright Reuters and its use requires agreements with
  NIST; the vectors carry no explicit license. Never redistribute the vectors
  or features derived from them.

  References
  ----------
  [1] Lewis, D. D., Yang, Y., Rose, T. G., & Li, F. (2004). RCV1: A New
      Benchmark Collection for Text Categorization Research. JMLR, 5, 361-397.
  """
  dataset_name = "RCV1-v2"
  cache_dir = "rcv1"
  hierarchy_url = ("http://jmlr.csail.mit.edu/papers/volume5/lewis04a/"
                   "a02-orig-topics-hierarchy/rcv1.topics.hier.orig")
  n_train = 23149
  license = DatasetLicense(
    spdx="LicenseRef-Reuters-Corpus",
    redistribution="restricted",
    source="https://trec.nist.gov/data/reuters/reuters.html",
    checked="2026-10-27",
    citation=("Lewis, D. D. et al. RCV1: A New Benchmark Collection for Text Categorization "
              "Research. JMLR 5:361-397, 2004."),
    conditions=("research-only", "cite", "no-redistribution"),
    notes="Story text is copyright Reuters/Thomson Reuters; NIST distributes it under signed agreements.",
  )

  def _download_and_load(self) -> dict:
    from sklearn.datasets import fetch_rcv1

    bunch = fetch_rcv1(data_home=str(self.dataset_dir))
    hierarchy_file = self._fetch(self.hierarchy_url, "rcv1.topics.hier.orig")
    return {"X": bunch.data, "Y": sp.csr_matrix(bunch.target), "topics": [str(t) for t in bunch.target_names],
            "edges": parse_rcv1_hierarchy(hierarchy_file.read_text(encoding="latin-1"))}

  def _preprocess(self, raw_data: dict) -> dict:
    return raw_data

  def _split_features_and_target(self, data: dict):
    Y, topics = data["Y"], data["topics"]
    label_sets = [[topics[j] for j in Y.indices[Y.indptr[k]:Y.indptr[k + 1]]] for k in range(Y.shape[0])]
    return data["X"], label_sets

  def _build_hierarchy(self, data: dict) -> Hierarchy:
    keep = set(data["topics"])
    edges = [(p, c) for p, c in data["edges"] if c in keep and (p == ROOT or p in keep)]
    hierarchy = Hierarchy(edges)
    if set(hierarchy.nodes) != keep:
      raise ValueError(f"hierarchy file and topics differ: {sorted(keep ^ set(hierarchy.nodes))[:10]}")
    return hierarchy

  def _split(self, data: dict):
    n = data["Y"].shape[0]
    return np.array(["train"] * self.n_train + ["test"] * (n - self.n_train))


class _PathsLoader(BaseHierarchicalDatasetLoader):
  """Base for sources with one root-to-leaf path per instance (single-path trees)."""
  is_text = True
  path_columns: tuple[str, ...] = ()

  def _preprocess(self, raw_data: pd.DataFrame) -> pd.DataFrame:
    data = raw_data.dropna(subset=list(self.path_columns)).reset_index(drop=True)
    data["path"] = list(zip(*(data[c].astype(str) for c in self.path_columns)))
    return data

  def _split_features_and_target(self, data):
    return data["text"].tolist(), [["/".join(p)] for p in data["path"]]

  def _build_hierarchy(self, data) -> Hierarchy:
    return Hierarchy.from_paths(data["path"].unique())

  def _split(self, data):
    return data["split"].to_numpy()


class DBPediaLoader(_PathsLoader):
  """DBPedia Classes: 342,782 Wikipedia abstracts in 3 levels (9 / 70 / 219 classes).

  One path per article, so the constraint between levels is an equality.
  Class names are the joined path (`"Agent/Athlete/BaseballPlayer"`).

  References
  ----------
  [1] Ofer, D. DBPedia Classes. Kaggle (derived from the DBpedia ontology and Wikipedia).
  """
  dataset_name = "DBPedia Classes"
  cache_dir = "dbpedia_classes"
  path_columns = ("l1", "l2", "l3")
  base_url = "https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes/resolve/main/"
  license = DatasetLicense(
    spdx="CC-BY-SA-3.0",
    redistribution="allowed-sharealike",
    source="https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes",
    checked="2026-10-27",
    citation="Ofer, D. DBPedia Classes (Kaggle), derived from DBpedia and Wikipedia abstracts.",
    conditions=("attribution", "share-alike"),
    notes=("Tagged CC0 on Hugging Face, but the text comes from Wikipedia abstracts (CC BY-SA 3.0), "
           "which is the license that applies."),
  )

  def _download_and_load(self) -> pd.DataFrame:
    parts = []
    for split, name in (("train", "DBPEDIA_train.csv"), ("val", "DBPEDIA_val.csv"), ("test", "DBPEDIA_test.csv")):
      parts.append(pd.read_csv(self._fetch(self.base_url + name, name)).assign(split=split))
    return pd.concat(parts, ignore_index=True)


class AmazonHierarchicalLoader(_PathsLoader):
  """Amazon product reviews with categories in 3 levels (6 / 64 / 510 classes).

  Title and review text are joined into one document; user ids are dropped.

  References
  ----------
  [1] Kashnitsky, Y. (2022). Amazon product reviews (mock dataset). Zenodo.
      doi:10.5281/zenodo.6657410
  """
  dataset_name = "Amazon reviews (3-level categories)"
  cache_dir = "amazon_hierarchical"
  path_columns = ("Cat1", "Cat2", "Cat3")
  files = {
    "train_40k.csv": "md5:9bda3bc6dcdb8cf68ce401a5b9782399",
    "val_10k.csv": "md5:2c22dc48f1674e5d1cc96f6491c9e2ba",
  }
  license = DatasetLicense(
    spdx="CC-BY-1.0",
    redistribution="allowed",
    source="https://zenodo.org/records/6657410",
    checked="2026-10-27",
    citation="Kashnitsky, Y. Amazon product reviews (mock dataset). Zenodo, 2022. doi:10.5281/zenodo.6657410",
    conditions=("attribution",),
    notes="Review text originates from the Amazon review data of McAuley et al.",
  )

  def _download_and_load(self) -> pd.DataFrame:
    parts = []
    for (name, checksum), split in zip(self.files.items(), ("train", "val")):
      url = f"https://zenodo.org/records/6657410/files/{name}"
      parts.append(pd.read_csv(self._fetch(url, name, checksum)).assign(split=split))
    data = pd.concat(parts, ignore_index=True)
    title = data["Title"].fillna("").astype(str) + ". " if "Title" in data else ""
    data["text"] = title + data["Text"].fillna("").astype(str)
    return data.drop(columns=[c for c in ("userId", "productId") if c in data])


class WebOfScienceLoader(BaseHierarchicalDatasetLoader):
  """Web of Science abstracts with domain > area labels.

  There is no official split. Class names are `domain:<YL1>` and `area:<Y>`,
  the numeric codes of the original files.

  Parameters
  ----------
  variant : {"WOS46985", "WOS11967", "WOS5736"}, default = "WOS46985"
    Full dataset (7 domains, 134 areas) or one of its two subsets.

  References
  ----------
  [1] Kowsari, K., Brown, D. E., Heidarysafa, M., Meimandi, K. J., Gerber, M. S.,
      & Barnes, L. E. (2017). HDLTex: Hierarchical Deep Learning for Text
      Classification. IEEE ICMLA 2017.
  """
  dataset_name = "Web of Science"
  cache_dir = "wos"
  is_text = True
  url = ("https://data.mendeley.com/public-files/datasets/9rw3vkcfy4/files/"
         "c9ea673d-5542-44c0-ab7b-f1311f7d61df/file_downloaded")
  license = DatasetLicense(
    spdx="CC-BY-4.0",
    redistribution="allowed",
    source="https://data.mendeley.com/datasets/9rw3vkcfy4/6",
    checked="2026-10-27",
    citation=("Kowsari, K. et al. HDLTex: Hierarchical Deep Learning for Text Classification. "
              "IEEE ICMLA 2017. Data: doi:10.17632/9rw3vkcfy4.6"),
    conditions=("attribution",),
    notes="The Mendeley download URL depends on a file id that may change between versions.",
  )

  def __init__(self, data_home=None, force_load: bool = False, variant: str = "WOS46985"):
    super().__init__(data_home, force_load)
    self.variant = variant

  def _download_and_load(self) -> pd.DataFrame:
    root = _extract_once(self._fetch(self.url, "WebOfScience.zip"), self.dataset_dir / "WebOfScience")
    for inner in sorted(root.rglob("*.zip")):
      _extract_once(inner, inner.with_suffix(""))
    hits = [p for p in root.rglob("X.txt") if self.variant in str(p)]
    if not hits:
      raise FileNotFoundError(f"{self.variant}/X.txt not found under {root}")
    folder = hits[0].parent

    def lines(name):
      return (folder / name).read_text(encoding="utf-8", errors="replace").splitlines()

    data = pd.DataFrame({"text": lines("X.txt"), "area": lines("Y.txt"), "domain": lines("YL1.txt")})
    return data

  def _preprocess(self, raw_data: pd.DataFrame) -> pd.DataFrame:
    data = raw_data.copy()
    data["area"] = "area:" + data["area"].str.strip()
    data["domain"] = "domain:" + data["domain"].str.strip()
    return data

  def _split_features_and_target(self, data):
    return data["text"].tolist(), [[a] for a in data["area"]]

  def _build_hierarchy(self, data) -> Hierarchy:
    pairs = set(zip(data["domain"], data["area"]))
    hierarchy = Hierarchy([(ROOT, d) for d in sorted({d for d, _ in pairs})] + sorted(pairs))
    if not hierarchy.is_tree:
      raise ValueError(f"{self.variant}: an area appears under more than one domain")
    return hierarchy


class MultiEURLEXLoader(BaseHierarchicalDatasetLoader):
  """MultiEURLEX: EU laws labelled with EuroVoc concepts at levels 1-3.

  Returns the documents of one language with labels from the three levels
  (21 / 127 / 567 classes) and the chronological split of [1] (train
  1958-2010, val 2010-2012, test 2012-2016), which shifts the label
  distribution over time. The year of each law (from its CELEX id) is
  returned as the natural group. The links between levels are not in the
  files; a concept is attached to the upper-level concepts present in every
  document that has it. Rare concepts may get more than one parent this way,
  which makes the taxonomy a DAG; the label matrix is not affected, since the
  files already list the labels of every level.

  Parameters
  ----------
  language : str, default = "pt"
    ISO code of one of the 23 languages. Laws without a text in that language are skipped.

  References
  ----------
  [1] Chalkidis, I., Fergadiotis, M., & Androutsopoulos, I. (2021). MultiEURLEX -
      A multi-lingual and multi-label legal document classification dataset for
      zero-shot cross-lingual transfer. EMNLP 2021.
  """
  dataset_name = "MultiEURLEX"
  cache_dir = "multieurlex"
  is_text = True
  url = "https://zenodo.org/records/5363165/files/multi_eurlex.tar.gz"
  checksum = "md5:9f7beeff307418146356cd259b622a2c"
  levels = (1, 2, 3)
  license = DatasetLicense(
    spdx="CC-BY-SA-4.0",
    redistribution="allowed-sharealike",
    source="https://huggingface.co/datasets/coastalcph/multi_eurlex",
    checked="2026-10-27",
    citation=("Chalkidis, I.; Fergadiotis, M.; Androutsopoulos, I. MultiEURLEX - A multi-lingual and "
              "multi-label legal document classification dataset for zero-shot cross-lingual transfer. "
              "EMNLP 2021."),
    conditions=("attribution", "share-alike"),
    notes="Zenodo declares CC BY 4.0 and the dataset card CC BY-SA 4.0; the stricter is kept. 2.8 GB archive.",
  )

  def __init__(self, data_home=None, force_load: bool = False, language: str = "pt"):
    super().__init__(data_home, force_load)
    self.language = language

  def _download_and_load(self) -> pd.DataFrame:
    root = _extract_once(self._fetch(self.url, "multi_eurlex.tar.gz", self.checksum),
                         self.dataset_dir / "multi_eurlex")
    rows = []
    for split, name in (("train", "train.jsonl"), ("val", "dev.jsonl"), ("test", "test.jsonl")):
      path = sorted(root.rglob(name))
      if not path:
        raise FileNotFoundError(f"{name} not found under {root}")
      with open(path[0], encoding="utf-8") as fh:
        for line in fh:
          doc = json.loads(line)
          text = doc["text"].get(self.language) if isinstance(doc["text"], dict) else doc["text"]
          if text:
            rows.append({"text": text, "celex_id": doc["celex_id"], "split": split,
                         **{f"level_{l}": doc["eurovoc_concepts"][f"level_{l}"] for l in self.levels}})
    return pd.DataFrame(rows)

  def _preprocess(self, raw_data):
    return raw_data

  def _split_features_and_target(self, data):
    cols = [f"level_{l}" for l in self.levels]
    return data["text"].tolist(), [sorted(set().union(*rows)) for rows in zip(*(data[c] for c in cols))]

  def _build_hierarchy(self, data) -> Hierarchy:
    return Hierarchy.from_levels([data[f"level_{l}"].tolist() for l in self.levels])

  def _groups(self, data):
    return np.array([int(c[1:5]) if c[1:5].isdigit() else -1 for c in data["celex_id"]])

  def _split(self, data):
    return data["split"].to_numpy()


def arxiv_archive(category: str) -> str:
  """Archive of an arXiv category: `"cs.LG"` gives `"cs"`; `"hep-th"` is its own archive."""
  return category.split(".", 1)[0]


class ArxivLoader(BaseHierarchicalDatasetLoader):
  """arXiv metadata: archive > category, multi-label through cross-lists, bags by year.

  The metadata snapshot is distributed on Kaggle, which requires an account.
  Download `arxiv-metadata-oai-snapshot.json` from
  https://www.kaggle.com/datasets/Cornell-University/arxiv (or with
  `kaggle datasets download Cornell-University/arxiv`) and place it in
  `~/.quack_data/arxiv/`. The document text is the title and the abstract;
  the natural group is the year of the first version.

  Parameters
  ----------
  archives : iterable of str, default = ("cs", "stat")
    Keep papers with at least one category in these archives.
  years : tuple of int, default = None
    Inclusive `(first, last)` submission years.
  max_docs : int, default = None
    Stop after this many papers (the snapshot has over 2 million).

  References
  ----------
  [1] Clement, C. B., Bierbaum, M., O'Keeffe, K. P., & Alemi, A. A. (2019).
      On the Use of ArXiv as a Dataset. arXiv:1905.00075.
  """
  dataset_name = "arXiv metadata"
  cache_dir = "arxiv"
  is_text = True
  filename = "arxiv-metadata-oai-snapshot.json"
  license = DatasetLicense(
    spdx="CC0-1.0",
    redistribution="allowed",
    source="https://www.kaggle.com/datasets/Cornell-University/arxiv",
    checked="2026-10-27",
    citation="Clement, C. B. et al. On the Use of ArXiv as a Dataset. arXiv:1905.00075, 2019.",
    access="login",
    notes=("Metadata only; arXiv states that metadata may be reused in commercial and "
           "non-commercial systems. Article full texts have other licenses and are not used."),
  )

  def __init__(self, data_home=None, force_load: bool = False, archives: Iterable[str] = ("cs", "stat"),
               years: tuple[int, int] | None = None, max_docs: int | None = None):
    super().__init__(data_home, force_load)
    self.archives = tuple(archives)
    self.years = years
    self.max_docs = max_docs

  @staticmethod
  def _year(doc: dict) -> int:
    try:
      return parsedate_to_datetime(doc["versions"][0]["created"]).year
    except (KeyError, IndexError, TypeError, ValueError):
      return int(str(doc.get("update_date", "0"))[:4] or 0)

  def _download_and_load(self) -> pd.DataFrame:
    path = self._require_file(self.filename, f"Download it from {self.license.source}")
    wanted = set(self.archives)
    rows = []
    with open(path, encoding="utf-8") as fh:
      for line in fh:
        doc = json.loads(line)
        cats = [c for c in doc["categories"].split() if arxiv_archive(c) in wanted]
        if not cats:
          continue
        year = self._year(doc)
        if self.years and not (self.years[0] <= year <= self.years[1]):
          continue
        rows.append({"text": " ".join(f"{doc['title']}. {doc['abstract']}".split()),
                     "categories": cats, "year": year, "arxiv_id": doc["id"]})
        if self.max_docs and len(rows) >= self.max_docs:
          break
    return pd.DataFrame(rows)

  def _preprocess(self, raw_data):
    self.extras_["arxiv_id"] = raw_data["arxiv_id"].to_numpy()
    return raw_data

  def _split_features_and_target(self, data):
    return data["text"].tolist(), data["categories"].tolist()

  def _build_hierarchy(self, data) -> Hierarchy:
    edges = set()
    for cats in data["categories"]:
      for c in cats:
        archive = arxiv_archive(c)
        edges.add((ROOT, archive))
        if c != archive:
          edges.add((archive, c))
    return Hierarchy(sorted(edges, key=lambda e: e[1]))

  def _groups(self, data):
    return data["year"].to_numpy()


CLUS_LICENSE = DatasetLicense(
  spdx="NOASSERTION",
  redistribution="not-declared",
  source="https://dtai.cs.kuleuven.be/clus/hmcdatasets",
  checked="2026-10-27",
  citation=("Vens, C., Struyf, J., Schietgat, L., Dzeroski, S., & Blockeel, H. Decision trees for "
            "hierarchical multi-label classification. Machine Learning 73(2):185-214, 2008."),
  access="manual",
  conditions=("cite",),
  notes="The page credits Amanda Clare for the data and states no license.",
)


class ClusHMCLoader(BaseHierarchicalDatasetLoader):
  """A yeast functional-genomics dataset of the Clus-HMC benchmark [1].

  Twelve datasets (cellcycle, church, derisi, eisen, expr, gasch1, gasch2,
  hom, pheno, seq, spo, struc) annotated with FunCat (a tree, 6 levels) or
  with the Gene Ontology (a DAG, up to 14 levels). Genes carry several
  classes, so the constraint between levels is an inequality.

  The page declares no license, so the files are not downloaded
  automatically: get the FunCat and GO archives from
  https://dtai.cs.kuleuven.be/clus/hmcdatasets and extract them into
  `~/.quack_data/clus_hmc/`. Files named like `cellcycle_FUN.train.arff`,
  `cellcycle_FUN.valid.arff` and `cellcycle_FUN.test.arff` are found in any
  subfolder. Nominal attributes are returned as category indices and missing
  values as NaN.

  References
  ----------
  [1] Vens, C., Struyf, J., Schietgat, L., Dzeroski, S., & Blockeel, H. (2008).
      Decision trees for hierarchical multi-label classification.
      Machine Learning, 73(2), 185-214.
  """
  name: str = ""
  ontology: str = ""
  cache_dir = "clus_hmc"
  license = CLUS_LICENSE
  _splits = (("train", "train"), ("val", "valid"), ("test", "test"))

  def _find(self, tag: str) -> Path | None:
    wanted = f"{self.name}_{self.ontology}.{tag}.arff".lower()
    hits = [p for p in self.dataset_dir.rglob("*.arff") if p.name.lower() == wanted]
    return sorted(hits)[0] if hits else None

  def _download_and_load(self) -> dict:
    for archive in sorted(self.dataset_dir.glob("*.zip")):
      _extract_once(archive, self.dataset_dir / archive.stem)
    found = {split: self._find(tag) for split, tag in self._splits}
    if not all(found.values()):
      stem = f"{self.name}_{self.ontology}"
      raise FileNotFoundError(
        f"{stem}.train/valid/test.arff not found in {self.dataset_dir}.\n"
        f"Download the archives from {self.license.source} and extract them into that folder.")
    parts = {split: read_clus_hmc(path) for split, path in found.items()}
    names = {tuple(p[3]) for p in parts.values()}
    if len(names) != 1:
      raise ValueError(f"{self.name}_{self.ontology}: splits have different attributes")
    return parts

  def _preprocess(self, raw_data: dict) -> dict:
    return raw_data

  def _split_features_and_target(self, data: dict):
    X = np.vstack([data[s][0] for s, _ in self._splits])
    label_sets = [labels for s, _ in self._splits for labels in data[s][1]]
    return X, label_sets

  def _build_hierarchy(self, data: dict) -> Hierarchy:
    edges = sorted({e for s, _ in self._splits for e in data[s][2]}, key=lambda e: (e[0] or "", e[1]))
    return Hierarchy(edges)

  def _split(self, data: dict):
    return np.concatenate([[s] * len(data[s][1]) for s, _ in self._splits])


CLUS_DATASETS: list[str] = ["cellcycle", "church", "derisi", "eisen", "expr", "gasch1",
                            "gasch2", "hom", "pheno", "seq", "spo", "struc"]


def _clus_loader(name: str, ontology: str) -> type:
  label = "FunCat" if ontology == "FUN" else "Gene Ontology"
  return type(f"{name.capitalize()}{ontology.capitalize()}Loader", (ClusHMCLoader,), {
    "name": name, "ontology": ontology, "dataset_name": f"{name} ({label}) [Clus-HMC]",
    "__doc__": f"{name} annotated with {label}. See `ClusHMCLoader`."})


class HierarchicalLoaderFactory:
  """Factory used to instantiate the correct hierarchical loader by dataset key.

  Every loader returns an indicator matrix over all classes of its taxonomy,
  closed upward, and declares its license in the `license` attribute. Single
  path trees (20 Newsgroups, DBPedia, Amazon, WOS) satisfy the equality
  constraint between levels; the others (GoEmotions, RCV1, MultiEURLEX,
  arXiv, Clus-HMC) only the inequality.
  """
  _registry = {
    "goemotions": GoEmotionsHierarchicalLoader,
    "20newsgroups": TwentyNewsgroupsLoader,
    "rcv1": RCV1Loader,
    "dbpedia": DBPediaLoader,
    "amazon-hier": AmazonHierarchicalLoader,
    "wos": WebOfScienceLoader,
    "multieurlex": MultiEURLEXLoader,
    "arxiv": ArxivLoader,
    **{f"{name}-{onto.lower()}": _clus_loader(name, onto) for onto in ("FUN", "GO") for name in CLUS_DATASETS},
  }

  @classmethod
  def get_loader(cls, dataset_name: str, **kwargs) -> BaseHierarchicalDatasetLoader:
    pipeline_class = cls._registry.get(dataset_name.lower())
    if not pipeline_class:
      raise ValueError(f"Dataset '{dataset_name}' not supported on hierarchical datasets.")
    return pipeline_class(**kwargs)


HIERARCHICAL_DATASETS: list[str] = list(HierarchicalLoaderFactory._registry.keys())


def load_hierarchical(dataset: str,
                      data_home: str | Path | None = None,
                      force_load: bool = False,
                      accept_terms: bool = False,
                      return_metadata: bool = False,
                      **kwargs) -> tuple:
  """Load a hierarchical dataset by name using the registered loader factory.

  Downloads the files (if not cached under `data_home`, verifying their
  checksums when known), preprocesses them and returns the instances, the
  label matrix over every class of the taxonomy and the taxonomy itself.

  Parameters
  ----------
  dataset : str
    Dataset key. Must be one of `quack.datasets.HIERARCHICAL_DATASETS`.
  data_home : str or Path, default = None
    Root folder of the cached files. Defaults to `~/.quack_data/`.
  force_load : bool, default = False
    Download the files again even if they are cached.
  accept_terms : bool, default = False
    Required for datasets whose license limits use (restricted, research
    only, login). Passing True states that the conditions shown in the error
    message were read.
  return_metadata : bool, default = False
    Also return a `DatasetMetadata` with label names, official split,
    natural groups and license.
  **kwargs
    Dataset-specific options, passed to the loader (e.g. `language` for
    `"multieurlex"`, `depth` for `"20newsgroups"`, `archives` for `"arxiv"`).

  Returns
  -------
  X : ndarray or sparse matrix
    Features as float32 (sparse for `"rcv1"`), or an object array of texts.
  Y : ndarray of shape (n_samples, n_classes)
    Indicator matrix over `hierarchy.nodes`, closed upward (int8).
  hierarchy : Hierarchy
    The taxonomy. `hierarchy.constraint(Y)` tells whether parent prevalences
    equal the sum of their children's (`"equality"`) or only bound them
    (`"inequality"`).
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
  >>> from quack.datasets import load_hierarchical
  >>> X, Y, hierarchy = load_hierarchical("goemotions")
  >>> top = [hierarchy.index[y] for y in hierarchy.level(1)]
  >>> Y[:, top].mean(axis=0)
  """
  loader = HierarchicalLoaderFactory.get_loader(dataset, data_home=data_home, force_load=force_load, **kwargs)
  check_terms(loader.license, accept_terms, dataset)
  X, Y, hierarchy = loader.load_dataset()
  if return_metadata:
    return X, Y, hierarchy, loader.metadata()
  return X, Y, hierarchy

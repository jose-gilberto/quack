"""Minimal ARFF reader for Mulan/MEKA multi-label files and Clus HMC files.

Covers what those formats use: numeric and nominal attributes, sparse rows
(`{index value, ...}`), missing values (`?`), quoted names, MEKA's
`-C n` label count, Mulan's XML label file and Clus' `hierarchical`
class attribute. String attributes are kept as text in `strings`.
"""

import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

_ATTR = re.compile(r"^@attribute\s+('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|\S+)\s+(.+)$", re.I)


@dataclass
class Attribute:
  name: str
  kind: str  # numeric | nominal | string | hierarchical
  values: list[str] = field(default_factory=list)  # nominal values or hierarchy spec


@dataclass
class ArffData:
  relation: str
  attributes: list[Attribute]
  X: object  # ndarray (dense) or csr_matrix (sparse), nominal values as indices
  strings: dict[int, list[str]]  # column index -> raw values of string/hierarchical attributes


def _unquote(s: str) -> str:
  s = s.strip()
  if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
    return s[1:-1].replace("\\'", "'").replace('\\"', '"')
  return s


def _split_values(spec: str) -> list[str]:
  """Split a nominal spec `{a, 'b c', d}` honouring quotes."""
  inner = spec.strip()[1:-1]
  return [_unquote(v) for v in next(csv.reader([inner], skipinitialspace=True, quotechar="'"))]


def _parse_header(lines: list[str]) -> tuple[str, list[Attribute], int]:
  relation = ""
  attrs: list[Attribute] = []
  for i, raw in enumerate(lines):
    line = raw.strip()
    if not line or line.startswith("%"):
      continue
    low = line.lower()
    if low.startswith("@relation"):
      relation = _unquote(line[len("@relation"):].strip())
    elif low.startswith("@attribute"):
      m = _ATTR.match(line)
      if not m:
        raise ValueError(f"cannot parse attribute line: {line!r}")
      name, spec = _unquote(m.group(1)), m.group(2).strip()
      sl = spec.lower()
      if spec.startswith("{"):
        attrs.append(Attribute(name, "nominal", _split_values(spec)))
      elif sl.startswith("hierarchical"):
        attrs.append(Attribute(name, "hierarchical", spec.split(None, 1)[1].split(",")))
      elif sl.startswith(("numeric", "real", "integer")):
        attrs.append(Attribute(name, "numeric"))
      elif sl.startswith(("string", "date")):
        attrs.append(Attribute(name, "string"))
      else:
        raise ValueError(f"unsupported attribute type in {line!r}")
    elif low.startswith("@data"):
      return relation, attrs, i + 1
  raise ValueError("no @data section")


def _value(attr: Attribute, token: str) -> float:
  token = _unquote(token)
  if token == "?":
    return np.nan
  if attr.kind == "numeric":
    return float(token)
  if attr.kind == "nominal":
    try:
      return float(attr.values.index(token))
    except ValueError:
      raise ValueError(f"value {token!r} not declared for attribute {attr.name!r}") from None
  return np.nan  # string / hierarchical go to `strings`


def read_arff(path: Path) -> ArffData:
  with open(path, encoding="utf-8", errors="replace") as fh:
    lines = fh.readlines()
  relation, attrs, start = _parse_header(lines)
  d = len(attrs)
  text_cols = [j for j, a in enumerate(attrs) if a.kind in {"string", "hierarchical"}]
  strings: dict[int, list[str]] = {j: [] for j in text_cols}
  dense_rows: list[list[float]] = []
  rows, cols, vals = [], [], []
  sparse = None
  n = 0
  for raw in lines[start:]:
    line = raw.strip()
    if not line or line.startswith("%"):
      continue
    if sparse is None:
      sparse = line.startswith("{")
    if sparse:
      body = line.strip("{}").strip()
      seen = set()
      if body:
        for item in body.split(","):
          j_str, tok = item.strip().split(None, 1)
          j = int(j_str)
          seen.add(j)
          if j in strings:
            strings[j].append(_unquote(tok))
          else:
            v = _value(attrs[j], tok)
            if v != 0:
              rows.append(n)
              cols.append(j)
              vals.append(v)
      for j in text_cols:
        if j not in seen:
          strings[j].append("")
    else:
      tokens = next(csv.reader([line], skipinitialspace=True, quotechar="'"))
      if len(tokens) != d:
        raise ValueError(f"row {n} has {len(tokens)} values, expected {d}")
      row = []
      for j, tok in enumerate(tokens):
        if j in strings:
          strings[j].append(_unquote(tok))
        row.append(_value(attrs[j], tok))
      dense_rows.append(row)
    n += 1
  if sparse:
    X = sp.csr_matrix((np.asarray(vals, float), (rows, cols)), shape=(n, d))
  else:
    X = np.asarray(dense_rows, dtype=float).reshape(n, d)
  return ArffData(relation, attrs, X, strings)


# ------------------------------------------------------------------ Mulan/MEKA
def _meka_label_count(relation: str) -> Optional[int]:
  m = re.search(r"-C\s+(-?\d+)", relation)
  return int(m.group(1)) if m else None


def _mulan_labels(xml_path: Path) -> list[str]:
  tree = ET.parse(xml_path)
  return [el.attrib["name"] for el in tree.iter() if el.tag.endswith("label") and "name" in el.attrib]


def read_multilabel(arff_path: Path, xml_path: Optional[Path] = None):
  """Read a Mulan (ARFF + XML) or MEKA (single ARFF with `-C n`) file.

  Returns `X` (features), `Y` (csr binary labels), feature names, label names.
  """
  data = read_arff(arff_path)
  names = [a.name for a in data.attributes]
  if xml_path is not None:
    label_names = _mulan_labels(xml_path)
    label_idx = [names.index(l) for l in label_names]
  else:
    c = _meka_label_count(data.relation)
    if c is None:
      raise ValueError(f"{arff_path.name}: no XML label file and no '-C n' in the relation name")
    label_idx = list(range(c)) if c > 0 else list(range(len(names) + c, len(names)))
    label_names = [names[j] for j in label_idx]
  feat_idx = [j for j in range(len(names)) if j not in set(label_idx)]
  X = data.X
  if sp.issparse(X):
    X = X.tocsc()
    Y = sp.csr_matrix(X[:, label_idx] > 0, dtype=np.int8)
    F = X[:, feat_idx].tocsr()
  else:
    Y = sp.csr_matrix(np.nan_to_num(X[:, label_idx]) > 0, dtype=np.int8)
    F = X[:, feat_idx]
  return F, Y, [names[j] for j in feat_idx], label_names


# --------------------------------------------------------------------- Clus HMC
def read_clus_hmc(path: Path, kind: str = "auto"):
  """Read a Clus HMC ARFF file.

  The class attribute is declared as `@ATTRIBUTE class hierarchical <spec>`.
  In tree files (FunCat) the spec lists paths such as `01/01/02`; in DAG
  files (GO) it lists `parent/child` edges with `root` as the top parent.
  Each data row ends with its labels separated by `@`.

  Returns `X` (dense, NaN for missing), per-row label sets, hierarchy edges
  `(parent, child)` with `None` for the root, and feature names.
  """
  data = read_arff(path)
  hier_cols = [j for j, a in enumerate(data.attributes) if a.kind == "hierarchical"]
  if len(hier_cols) != 1:
    raise ValueError(f"{path.name}: expected one hierarchical attribute, found {len(hier_cols)}")
  hj = hier_cols[0]
  spec = [s.strip() for s in data.attributes[hj].values if s.strip()]
  if kind == "auto":
    kind = "dag" if any(s.split("/")[0].lower() == "root" for s in spec) else "tree"
  edges: set[tuple[Optional[str], str]] = set()
  if kind == "tree":
    for item in spec:
      parts = item.split("/")
      for i in range(len(parts)):
        parent = "/".join(parts[:i]) if i else None
        edges.add((parent, "/".join(parts[: i + 1])))
  elif kind == "dag":
    for item in spec:
      parent, child = item.split("/")
      edges.add((None if parent.lower() == "root" else parent, child))
  else:
    raise ValueError("kind must be 'auto', 'tree' or 'dag'")
  label_sets = [[l for l in raw.split("@") if l and l.lower() != "root"] for raw in data.strings[hj]]
  feat = [j for j, a in enumerate(data.attributes) if a.kind in {"numeric", "nominal"}]
  X = data.X[:, feat]
  if sp.issparse(X):
    X = X.toarray()
  names = [data.attributes[j].name for j in feat]
  return np.asarray(X, dtype=float), label_sets, sorted(edges, key=lambda e: (e[0] or "", e[1])), names

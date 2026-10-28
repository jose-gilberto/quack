# tests/test_datasets_arff.py
import numpy as np
import pytest
import scipy.sparse as sp

from quack.datasets._arff import read_arff, read_clus_hmc, read_multilabel

DENSE = """% comment
@relation 'toy: -C 2'
@attribute l1 {0,1}
@attribute l2 {0,1}
@attribute 'feat one' numeric
@attribute color {red,'dark blue'}
@data
1,0,0.5,red
0,1,?,'dark blue'
"""

SPARSE = """@relation sparse
@attribute f1 numeric
@attribute f2 numeric
@attribute y1 {0,1}
@attribute y2 {0,1}
@data
{0 1.5,2 1}
{1 2,3 1}
{}
"""

MULAN_XML = """<?xml version="1.0" encoding="utf-8"?>
<labels xmlns="http://mulan.sourceforge.net/labels">
<label name="y1"></label>
<label name="y2"></label>
</labels>
"""

CLUS_TREE = """@RELATION toy_FUN
@ATTRIBUTE a numeric
@ATTRIBUTE b {x,y}
@ATTRIBUTE class hierarchical 01,01/01,01/02,02
@DATA
1.0,x,01/01@02
2.0,?,01/02
"""

CLUS_DAG = """@RELATION toy_GO
@ATTRIBUTE a numeric
@ATTRIBUTE class hierarchical root/GO1,root/GO2,GO1/GO3,GO2/GO3
@DATA
1.0,GO3
2.0,GO1@GO2
"""


def write(tmp_path, name, text):
  path = tmp_path / name
  path.write_text(text, encoding="utf-8")
  return path


class TestReadArff:
  def test_dense_with_nominal_quotes_and_missing(self, tmp_path):
    data = read_arff(write(tmp_path, "toy.arff", DENSE))
    assert data.relation == "toy: -C 2"
    assert [a.name for a in data.attributes] == ["l1", "l2", "feat one", "color"]
    assert data.attributes[3].values == ["red", "dark blue"]
    assert data.X.shape == (2, 4)
    assert data.X[1, 3] == 1.0  # 'dark blue' is the second value
    assert np.isnan(data.X[1, 2])

  def test_sparse_rows(self, tmp_path):
    data = read_arff(write(tmp_path, "sparse.arff", SPARSE))
    assert sp.issparse(data.X)
    np.testing.assert_array_equal(data.X.toarray(), [[1.5, 0, 1, 0], [0, 2, 0, 1], [0, 0, 0, 0]])

  def test_undeclared_nominal_value_raises(self, tmp_path):
    with pytest.raises(ValueError, match="not declared"):
      read_arff(write(tmp_path, "bad.arff", DENSE.replace("0,1,?,'dark blue'", "0,1,?,green")))

  def test_missing_data_section_raises(self, tmp_path):
    with pytest.raises(ValueError, match="@data"):
      read_arff(write(tmp_path, "empty.arff", "@relation x\n@attribute a numeric\n"))


class TestReadMultilabel:
  def test_meka_label_count_in_relation(self, tmp_path):
    X, Y, features, labels = read_multilabel(write(tmp_path, "toy.arff", DENSE))
    assert labels == ["l1", "l2"]
    assert features == ["feat one", "color"]
    np.testing.assert_array_equal(Y.toarray(), [[1, 0], [0, 1]])
    assert X.shape == (2, 2)

  def test_mulan_xml_labels(self, tmp_path):
    arff = write(tmp_path, "sparse.arff", SPARSE)
    X, Y, features, labels = read_multilabel(arff, write(tmp_path, "sparse.xml", MULAN_XML))
    assert labels == ["y1", "y2"]
    assert features == ["f1", "f2"]
    assert sp.issparse(X)
    np.testing.assert_array_equal(Y.toarray(), [[1, 0], [0, 1], [0, 0]])

  def test_without_xml_or_label_count_raises(self, tmp_path):
    with pytest.raises(ValueError, match="-C n"):
      read_multilabel(write(tmp_path, "sparse.arff", SPARSE))


class TestReadClusHMC:
  def test_tree_paths(self, tmp_path):
    X, label_sets, edges, names = read_clus_hmc(write(tmp_path, "toy_FUN.train.arff", CLUS_TREE))
    assert names == ["a", "b"]
    assert np.isnan(X[1, 1])
    assert label_sets == [["01/01", "02"], ["01/02"]]
    assert (None, "01") in edges and ("01", "01/01") in edges and (None, "02") in edges

  def test_dag_edges(self, tmp_path):
    X, label_sets, edges, _ = read_clus_hmc(write(tmp_path, "toy_GO.train.arff", CLUS_DAG))
    assert set(edges) == {(None, "GO1"), (None, "GO2"), ("GO1", "GO3"), ("GO2", "GO3")}
    assert label_sets == [["GO3"], ["GO1", "GO2"]]
    assert X.shape == (2, 1)

  def test_requires_one_hierarchical_attribute(self, tmp_path):
    with pytest.raises(ValueError, match="hierarchical attribute"):
      read_clus_hmc(write(tmp_path, "plain.arff", SPARSE))

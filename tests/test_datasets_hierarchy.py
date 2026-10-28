# tests/test_datasets_hierarchy.py
import numpy as np
import pytest

from quack.datasets import ROOT, Hierarchy


@pytest.fixture
def tree():
  #        root
  #      /      \
  #   clothing   shoes
  #   /     \
  # shirt  trousers
  return Hierarchy.from_paths([("clothing", "shirt"), ("clothing", "trousers"), ("shoes",)])


@pytest.fixture
def dag():
  # c has two parents (a and b); d hangs from c.
  return Hierarchy([(None, "a"), (None, "b"), ("a", "c"), ("b", "c"), ("c", "d")])


class TestHierarchyStructure:
  def test_nodes_are_in_breadth_first_order(self, tree):
    assert tree.nodes == ["clothing", "shoes", "clothing/shirt", "clothing/trousers"]
    assert tree.index["clothing/shirt"] == 2
    assert len(tree) == tree.n == 4

  def test_navigation(self, tree):
    assert tree.children() == ["clothing", "shoes"]
    assert tree.children("clothing") == ["clothing/shirt", "clothing/trousers"]
    assert tree.parents("clothing/shirt") == ["clothing"]
    assert tree.parents("clothing") == []
    assert tree.ancestors("clothing/shirt") == {"clothing"}
    assert tree.descendants("clothing") == {"clothing/shirt", "clothing/trousers"}
    assert "shoes" in tree and ROOT not in tree

  def test_levels_and_leaves(self, tree):
    assert tree.depth("clothing") == 1
    assert tree.depth("clothing/trousers") == 2
    assert tree.level(1) == ["clothing", "shoes"]
    assert tree.max_depth == 2
    assert tree.leaves == ["shoes", "clothing/shirt", "clothing/trousers"]
    assert tree.internal == ["clothing"]

  def test_is_tree(self, tree, dag):
    assert tree.is_tree
    assert not dag.is_tree
    assert dag.ancestors("d") == {"a", "b", "c"}

  def test_edges_include_root(self, tree):
    assert (ROOT, "clothing") in tree.edges()
    assert ("clothing", "clothing/shirt") in tree.edges()

  def test_node_only_seen_as_parent_hangs_from_root(self):
    h = Hierarchy([("a", "b")])
    assert h.children() == ["a"]

  def test_custom_node_order(self):
    h = Hierarchy([(None, "a"), (None, "b")], nodes=["b", "a"])
    assert h.nodes == ["b", "a"]

  def test_node_order_must_cover_every_class(self):
    with pytest.raises(ValueError, match="every class"):
      Hierarchy([(None, "a"), (None, "b")], nodes=["a"])

  def test_rejects_cycles_and_self_loops(self):
    with pytest.raises(ValueError, match="cycle"):
      Hierarchy([(None, "a"), ("a", "b"), ("b", "a")])
    with pytest.raises(ValueError, match="self-loop"):
      Hierarchy([("a", "a")])
    with pytest.raises(ValueError, match="root"):
      Hierarchy([("a", ROOT)])

  def test_repr(self, tree):
    assert repr(tree) == "Hierarchy(tree, n=4, top=2, leaves=3, depth=2)"


class TestHierarchyFromLevels:
  def test_links_inferred_from_co_occurrence(self):
    level_1 = [["A"], ["A"], ["B"], ["A", "B"]]
    level_2 = [["a1"], ["a2"], ["b1"], ["a1", "b1"]]
    h = Hierarchy.from_levels([level_1, level_2])
    assert h.parents("a1") == ["A"]
    assert h.parents("b1") == ["B"]
    assert h.is_tree

  def test_class_without_common_parent_hangs_from_root(self):
    h = Hierarchy.from_levels([[["A"], ["B"]], [["x"], ["x"]]])
    assert h.parents("x") == []
    assert "x" in h.children()


class TestLabelMatrix:
  def test_closed_upward(self, tree):
    Y = tree.label_matrix([["clothing/shirt"], ["shoes"], []])
    np.testing.assert_array_equal(Y, [[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    assert Y.dtype == np.int8

  def test_dag_closes_through_every_parent(self, dag):
    Y = dag.label_matrix([["d"]])
    assert Y.sum() == 4

  def test_unknown_label_raises(self, tree):
    with pytest.raises(KeyError, match="not in the hierarchy"):
      tree.label_matrix([["hats"]])

  def test_close_upward_accepts_dense(self, tree):
    Y = np.array([[0, 0, 0, 1]])
    np.testing.assert_array_equal(tree.close_upward(Y), [[1, 0, 0, 1]])


class TestConstraints:
  def test_single_path_full_depth_is_equality(self, tree):
    Y = tree.label_matrix([["clothing/shirt"], ["clothing/trousers"], ["shoes"]])
    assert tree.constraint(Y) == "equality"

  def test_stopping_at_internal_node_is_inequality(self, tree):
    Y = tree.label_matrix([["clothing"], ["shoes"]])
    assert tree.constraint(Y) == "inequality"

  def test_multiple_paths_is_inequality(self, tree):
    Y = tree.label_matrix([["clothing/shirt", "shoes"]])
    assert tree.constraint(Y) == "inequality"

  def test_dag_is_inequality(self, dag):
    assert dag.constraint(dag.label_matrix([["d"]])) == "inequality"

  def test_sum_matrix(self, tree):
    S = tree.sum_matrix()
    p_leaves = np.array([0.5, 0.3, 0.2])  # shoes, shirt, trousers
    np.testing.assert_allclose(S @ p_leaves, [0.5, 0.5, 0.3, 0.2])

  def test_sum_matrix_requires_tree(self, dag):
    with pytest.raises(ValueError, match="tree"):
      dag.sum_matrix()

  def test_violations(self, tree):
    assert tree.violations([0.5, 0.5, 0.3, 0.2]) == 0
    assert tree.violations([0.1, 0.5, 0.3, 0.2]) == 2

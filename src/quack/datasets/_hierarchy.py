from collections import deque
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

ROOT = "<root>"
"""Name of the implicit root node. It is not one of the classes."""


class Hierarchy:
  """Class taxonomy (tree or DAG) used by the hierarchical dataset loaders.

  The root is implicit and is not counted among the `n` classes. Label
  matrices built by `label_matrix` have one column per class, in `nodes`
  order, and are closed upward: an instance labelled with a class is also
  labelled with every ancestor of that class.

  Parameters
  ----------
  edges : iterable of (str or None, str)
    `(parent, child)` pairs. Use `None` or `ROOT` as the parent of the
    top-level classes. A node that appears only as a parent hangs from the root.
  nodes : sequence of str, default = None
    Column order of the label matrices. Defaults to breadth-first order
    (by depth, then by name).

  Attributes
  ----------
  nodes : list of str
    Class names, in column order.
  index : dict
    Maps each class name to its column.

  Examples
  --------
  >>> h = Hierarchy.from_paths([("clothing", "shirt"), ("clothing", "trousers"), ("shoes",)])
  >>> h.nodes
  ['clothing', 'shoes', 'clothing/shirt', 'clothing/trousers']
  >>> h.label_matrix([["clothing/shirt"]]).tolist()
  [[1, 0, 1, 0]]
  """

  def __init__(self, edges: Iterable[tuple[str | None, str]], nodes: Sequence[str] | None = None):
    children: dict[str, list[str]] = {ROOT: []}
    parents: dict[str, list[str]] = {}
    for parent, child in edges:
      parent = ROOT if parent is None else parent
      if child == ROOT:
        raise ValueError("the root cannot be a child")
      if parent == child:
        raise ValueError(f"self-loop on {child!r}")
      children.setdefault(parent, [])
      children.setdefault(child, [])
      if child not in children[parent]:
        children[parent].append(child)
      parents.setdefault(child, [])
      if parent not in parents[child]:
        parents[child].append(parent)
    for node in list(children):
      if node != ROOT and node not in parents:
        parents[node] = [ROOT]
        children[ROOT].append(node)
    self._children = children
    self._parents = parents
    self._check_acyclic()
    self._depth = self._bfs_depth()

    order = list(nodes) if nodes is not None else sorted(
      (y for y in children if y != ROOT), key=lambda y: (self._depth[y], y))
    if sorted(order) != sorted(y for y in children if y != ROOT):
      raise ValueError("`nodes` must list every class exactly once, without the root")
    self.nodes: list[str] = order
    self.index: dict[str, int] = {y: i for i, y in enumerate(order)}

  @classmethod
  def from_paths(cls, paths: Iterable[Sequence[str]], sep: str = "/") -> "Hierarchy":
    """Tree built from root-to-node paths such as `("comp", "sys", "mac")`.

    Node names are the joined prefixes (`"comp/sys"`), so equal names under
    different parents never collide.
    """
    edges = set()
    for path in paths:
      parent = ROOT
      for i in range(len(path)):
        node = sep.join(path[:i + 1])
        edges.add((parent, node))
        parent = node
    return cls(sorted(edges, key=lambda e: e[1]))

  @classmethod
  def from_levels(cls, level_sets: Sequence[Sequence[Sequence[str]]]) -> "Hierarchy":
    """Infer the links between levels from per-level label sets of the same instances.

    `level_sets[l][k]` is the set of level-`l` labels of instance `k`. A class
    at level `l + 1` is attached to the level-`l` classes present in every
    instance that has it, or to the root if there is none. Used when a source
    lists the labels of each level but not the parent of each class.
    """
    edges = [(ROOT, top) for top in sorted({y for row in level_sets[0] for y in row})]
    for upper, lower in zip(level_sets[:-1], level_sets[1:]):
      common: dict[str, set[str]] = {}
      for up_row, low_row in zip(upper, lower):
        for child in low_row:
          common[child] = common[child] & set(up_row) if child in common else set(up_row)
      for child in sorted(common):
        edges.extend((p, child) for p in (sorted(common[child]) or [ROOT]))
    return cls(edges)

  @property
  def n(self) -> int:
    """Number of classes (the root excluded)."""
    return len(self.nodes)

  def __len__(self) -> int:
    return self.n

  def __contains__(self, y: str) -> bool:
    return y in self.index

  def children(self, y: str = ROOT) -> list[str]:
    """Direct children of `y` (of the root by default)."""
    return list(self._children[y])

  def parents(self, y: str) -> list[str]:
    """Direct parents of `y`, without the root."""
    return [p for p in self._parents[y] if p != ROOT]

  def ancestors(self, y: str) -> set[str]:
    """All ancestors of `y`, without the root."""
    out: set[str] = set()
    stack = list(self._parents[y])
    while stack:
      p = stack.pop()
      if p != ROOT and p not in out:
        out.add(p)
        stack.extend(self._parents[p])
    return out

  def descendants(self, y: str) -> set[str]:
    """All descendants of `y`."""
    out: set[str] = set()
    stack = list(self._children[y])
    while stack:
      c = stack.pop()
      if c not in out:
        out.add(c)
        stack.extend(self._children[c])
    return out

  def depth(self, y: str) -> int:
    """Shortest distance from the root; top-level classes have depth 1."""
    return self._depth[y]

  def level(self, depth: int) -> list[str]:
    """Classes at a given depth, in column order."""
    return [y for y in self.nodes if self._depth[y] == depth]

  @property
  def max_depth(self) -> int:
    return max(self._depth[y] for y in self.nodes)

  @property
  def leaves(self) -> list[str]:
    """Classes without children."""
    return [y for y in self.nodes if not self._children[y]]

  @property
  def internal(self) -> list[str]:
    """Classes with children."""
    return [y for y in self.nodes if self._children[y]]

  @property
  def is_tree(self) -> bool:
    """True when every class has exactly one parent."""
    return all(len(self._parents[y]) == 1 for y in self.nodes)

  def edges(self) -> list[tuple[str, str]]:
    """`(parent, child)` pairs, with `ROOT` as the parent of top-level classes."""
    return [(p, c) for c in self.nodes for p in self._parents[c]]

  def sum_matrix(self) -> np.ndarray:
    """Summing matrix `S` of shape `(n, n_leaves)`.

    `S[i, j] = 1` when class `i` is leaf `j` or one of its ancestors, so that
    `p = S @ p_leaves` for single-path, full-depth labelling. Trees only.
    """
    if not self.is_tree:
      raise ValueError("the summing matrix requires a tree")
    leaves = self.leaves
    S = np.zeros((self.n, len(leaves)), dtype=np.int8)
    for j, leaf in enumerate(leaves):
      S[self.index[leaf], j] = 1
      for a in self.ancestors(leaf):
        S[self.index[a], j] = 1
    return S

  def label_matrix(self, label_sets: Iterable[Iterable[str]]) -> np.ndarray:
    """Binary indicator matrix `(n_samples, n)` from per-instance label sets, closed upward."""
    rows, cols = [], []
    n_rows = 0
    for k, labels in enumerate(label_sets):
      n_rows = k + 1
      for y in labels:
        if y not in self.index:
          raise KeyError(f"label {y!r} is not in the hierarchy")
        rows.append(k)
        cols.append(self.index[y])
    Y = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_rows, self.n))
    return self.close_upward(Y)

  def close_upward(self, Y) -> np.ndarray:
    """Add every ancestor of every positive label. Returns a dense int8 matrix."""
    rows, cols = [], []
    for y in self.nodes:
      i = self.index[y]
      rows.append(i)
      cols.append(i)
      for a in self.ancestors(y):
        rows.append(i)
        cols.append(self.index[a])
    A = sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(self.n, self.n))
    closed = sp.csr_matrix(Y, dtype=np.int32) @ A
    return (closed.toarray() > 0).astype(np.int8)

  def constraint(self, Y) -> str:
    """Kind of consistency constraint the labels in `Y` satisfy.

    Returns `"equality"` when the taxonomy is a tree and every instance
    follows exactly one path from a top-level class down to a leaf; then the
    prevalence of a class equals the sum of its children's prevalences in
    every bag. Otherwise returns `"inequality"`: a child's prevalence never
    exceeds its parent's.
    """
    if not self.is_tree:
      return "inequality"
    Y = np.asarray(Y)
    top = [self.index[y] for y in self.children(ROOT)]
    if not np.all(Y[:, top].sum(axis=1) == 1):
      return "inequality"
    for y in self.internal:
      kids = [self.index[c] for c in self.children(y)]
      has = Y[:, self.index[y]] == 1
      n_kids = Y[:, kids].sum(axis=1)
      if np.any(n_kids[has] != 1) or np.any(n_kids[~has] != 0):
        return "inequality"
    return "equality"

  def violations(self, prevalence: Sequence[float], tol: float = 1e-9) -> int:
    """Number of parent/child pairs whose estimated prevalence has the child above the parent."""
    p = np.asarray(prevalence, dtype=float)
    return int(sum(p[self.index[c]] > p[self.index[q]] + tol for q, c in self.edges() if q != ROOT))

  def _bfs_depth(self) -> dict[str, int]:
    depth = {ROOT: 0}
    queue = deque([ROOT])
    while queue:
      u = queue.popleft()
      for v in self._children[u]:
        if v not in depth:
          depth[v] = depth[u] + 1
          queue.append(v)
    return depth

  def _check_acyclic(self) -> None:
    indegree = {y: len(self._parents.get(y, [])) for y in self._children}
    queue = deque([ROOT])
    seen = 0
    while queue:
      u = queue.popleft()
      seen += 1
      for v in self._children[u]:
        indegree[v] -= 1
        if indegree[v] == 0:
          queue.append(v)
    if seen != len(self._children):
      raise ValueError("the taxonomy has a cycle")

  def __repr__(self) -> str:
    kind = "tree" if self.is_tree else "DAG"
    return (f"Hierarchy({kind}, n={self.n}, top={len(self.children(ROOT))}, "
            f"leaves={len(self.leaves)}, depth={self.max_depth})")

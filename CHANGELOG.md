# Changelog

All notable changes to `quack` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While `quack` is in `0.x`, minor releases may contain breaking changes.

## [0.1.1]

### Fixed

- **Topsoe distance in the convex solver.** `BaseMixtureQuantifier` expressed
  the `'TS'` objective as `kl_div(2p, t) + kl_div(2t, p)`, which is not the
  Topsoe distance. The solver converged, but to the minimum of a different
  function, so `DyS(distance_metric="TS")` (the default) could return
  estimates far from the optimum — up to `0.49` away from the correct
  prevalence on a single bag in our benchmark, and a mean absolute error
  ~5x worse than the Golden Section Search fallback. The objective is now
  `KL(p || m) + KL(t || m)` with `m = (p + t)/2`, which is exactly the
  Topsoe distance and matches `_compute_distance` and the GSS fallback.
- **`NotFittedError` was never raised.** Trailing-underscore attributes
  (`conditional_matrix_`, `score_range_`, `bins_`, `feature_spaces_`,
  `unique_rows_`, `class_distances_matrix_`, `quadratic_matrix_`,
  `train_class_samples_`, `feature_subsets_`, `sub_quantifiers_`) were
  initialized in `__init__`. Because `sklearn.utils.validation.check_is_fitted`
  treats any such attribute as proof of a fitted estimator, calling
  `predict` before `fit` raised an obscure `TypeError` instead of a
  `NotFittedError`. They are now set during `fit` only.
- **`ReadMe.fit` no longer overwrites the `n_features` hyper-parameter.**
  The resolved subspace size is exposed as the fitted attribute
  `n_features_`, restoring the scikit-learn contract that `fit` must not
  mutate `__init__` parameters (which `clone` and `GridSearchCV` rely on).
- **`quack.datasets._reviews_datasets`** imported `src.quack.utils`, making
  the module unimportable outside a source checkout.
- **Docs workflow** watched `mkdocs.yml`, a file that does not exist (the
  project uses `mkdocs.yaml`), so configuration changes never triggered a
  docs deploy. It now also watches `src/quack/**`, since the API reference
  is generated from docstrings.

### Changed

- **Packaging metadata.** `requires-python` is now `>=3.10` (the code uses
  `X | Y` annotations evaluated at runtime, so 3.8/3.9 never worked); the
  license classifier said MIT while `LICENSE` is BSD 3-Clause, now declared
  as `BSD-3-Clause`; the version is read from `quack.core.__VERSION__`
  instead of being duplicated; project URLs, keywords and Python version
  classifiers were added.
- **Dependency lower bounds lowered** to the oldest versions the test suite
  is verified against (`numpy>=1.26`, `scipy>=1.11`, `scikit-learn>=1.4`,
  `pandas>=2.1`, `cvxpy>=1.4`). The previous floors (`numpy>=2.5`,
  `pandas>=3.0.3`, `scikit-learn>=1.9`) forced an upgrade of the whole
  scientific stack on every downstream project.
- Private CVXPY cache attributes lost their trailing underscore
  (`_cvx_problem_` to `_cvx_problem`, and likewise for the others) so they no
  longer register as fitted state.

### Added

- `quack.__version__`.
- `py.typed` marker, so type checkers honour the annotations already present
  in the public API.
- `tests/test_estimator_contract.py`: unfitted estimators must raise
  `NotFittedError`, `fit` must not mutate hyper-parameters, and the convex
  solver must not be worse than a brute-force search on the very distance it
  claims to minimize (the check that catches the Topsoe bug above).
- CI matrix over Python 3.10-3.13, plus a job that installs the declared
  dependency lower bounds and runs the suite against them.

### Documentation

- `HDy` documents that `QuaPy` implements the historical variant (median over
  bins 10..110), so the two libraries only agree at a fixed `n_bins`.
- `precision` in the threshold selectors documents that `None` reproduces
  `QuaPy`'s candidate grid exactly.
- README: corrected the visualization imports, the Python/dependency
  requirements, the metrics list (`nae` was missing) and added a License
  section.

## [0.1.0]

- Initial alpha release.

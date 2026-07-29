# Contributing to Quack

First off, thank you for considering contributing to `quack` 🦆! This project is community-driven, and contributions of any size, bug reports, documentation fixes, new quantifiers, dataset loaders, or tests, are welcome.

This document describes the branching model, development workflow, and quality standards we follow.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Ways to Contribute](#ways-to-contribute)
3. [Branching Model (GitFlow)](#branching-model-gitflow)
4. [Development Setup](#development-setup)
5. [Commit Message Convention](#commit-message-convention)
6. [Pull Request Process](#pull-request-process)
7. [Coding Standards](#coding-standards)
8. [Testing](#testing)
9. [Documentation](#documentation)
10. [Reporting Bugs](#reporting-bugs)
11. [Proposing New Features](#proposing-new-features)
12. [Releasing](#releasing)

---

## Code of Conduct

Be respectful, constructive, and patient. We follow the spirit of the [Contributor Covenant](https://www.contributor-covenant.org/). Harassment, discrimination, or disrespectful conduct will not be tolerated.

---

## Ways to Contribute

- Report bugs or unexpected behavior.
- Improve documentation, docstrings, or tutorials.
- Implement new quantification methods, bag generators, dataset loaders, or metrics.
- Add or improve test coverage.
- Propose enhancements via issues before large changes.

If you're unsure where to start, look for issues labeled `good first issue` or `help wanted`.

---

## Branching Model (GitFlow)

`quack` follows a **GitFlow**-inspired branching strategy to keep `main` always stable and releasable.

| Branch | Purpose | Base | Merges into |
|---|---|---|---|
| `main` | Latest stable, released code. Every commit here is tagged. | — | — |
| `develop` | Integration branch for the next release. All feature work lands here first. | `main` | `main` (via release branch) |
| `feature/<short-description>` | New quantifiers, loaders, metrics, or enhancements. | `develop` | `develop` |
| `fix/<short-description>` | Non-urgent bug fixes. | `develop` | `develop` |
| `release/x.y.z` | Stabilization branch before a version release (version bump, changelog, final QA). | `develop` | `main` **and** `develop` |
| `hotfix/x.y.z` | Urgent fixes to production (`main`). | `main` | `main` **and** `develop` |
| `docs/<short-description>` | Documentation-only changes. | `develop` | `develop` |

### Branch naming examples

```
feature/hdy-quantifier
feature/uci-loader-adult
fix/acc-denominator-zero-division
docs/uci-datasets-tutorial
release/0.2.0
hotfix/0.1.1
```

### Typical workflow

```bash
# 1. Sync with develop
git checkout develop
git pull origin develop

# 2. Create your feature branch
git checkout -b feature/my-new-quantifier

# 3. Work, commit, push
git push origin feature/my-new-quantifier

# 4. Open a Pull Request targeting `develop` (never `main` directly)
```

> **Note:** External contributors should fork the repository and open PRs from their fork's equivalent branches, targeting `develop` on `jose-gilberto/quack`.

---

## Development Setup

```bash
git clone https://github.com/<your-fork>/quack.git
cd quack
git checkout develop

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e ".[test,docs]"
```

Run the test suite to confirm your environment is healthy:

```bash
pytest
```

---

## Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/) to keep history readable and enable automated changelog generation:

```
<type>(<optional scope>): <short summary>

[optional body]

[optional footer(s)]
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`

Examples:

```
feat(quantifiers): add DyS with Topsoe distance support
fix(base): guard against zero division in ACC adjustment formula
docs(datasets): add UCI datasets tutorial page
test(threshold): cover MedianSweep fallback branch
refactor(dmm): extract shared calibration logic to base class
```

---

## Pull Request Process

1. **Target `develop`** (unless it's a `hotfix/*`, which targets `main`).
2. Keep PRs **focused and small** — one quantifier, one loader, or one fix per PR whenever possible.
3. Ensure:
   - [ ] All quantifiers inherit from `BaseQuantifier` and implement `.fit(X, y)` / `.predict(X)`.
   - [ ] `.predict()` returns a normalized 1D NumPy array summing to `1.0`.
   - [ ] New code has corresponding tests (see [Testing](#testing)).
   - [ ] `pytest` passes locally with no regressions in coverage.
   - [ ] Public classes/functions have NumPy-style docstrings (see existing modules for the pattern).
   - [ ] No new heavy dependencies were introduced (stick to `numpy`, `scipy`, `scikit-learn`, `cvxpy`).
4. Fill out the PR template (description, motivation, related issue, screenshots/plots if relevant).
5. A maintainer will review, request changes if needed, and merge using **squash merge** to keep `develop` history clean.
6. CI (`.github/workflows/tests.yaml`) must pass before merge.

---

## Coding Standards

- **Style**: 2-space indentation, matching the existing codebase (see `src/quack/quantifiers/base.py` for reference).
- **Type hints**: all public function/method signatures should be typed.
- **Docstrings**: NumPy style (`Parameters`, `Returns`, `Attributes`, `References`, `Examples`), consistent with `mkdocstrings` configuration in `mkdocs.yaml`.
- **Vectorization**: prefer NumPy/SciPy vectorized operations over explicit Python loops when iterating over bags or computing prevalence aggregations.
- **Abstraction**: new quantifiers must cleanly extend `BaseQuantifier`, `BaseCalibratedQuantifier`, or `BaseMixtureQuantifier` as appropriate — do not duplicate the calibration/GSS/CVXPY machinery already provided by the base classes.
- **Dependencies**: rely strictly on the standard Scientific Python stack (`numpy`, `scipy`, `scikit-learn`, `cvxpy`). Discuss in an issue before introducing anything new.

---

## Testing

- Tests live under `tests/`, mirroring the `src/quack/` package structure (e.g., `tests/test_uci_datasets.py` for `src/quack/datasets/_uci_datasets.py`).
- Use `pytest`. Group related assertions into classes (`TestX`) as done in existing test files.
- **Never rely on real network access** in tests. For dataset loaders:
  - Use `monkeypatch` to stub `download_file` / `urllib.request.urlopen`.
  - Use lightweight `DummyLoader`-style fixtures for pipeline logic (see `tests/test_uci_datasets.py`).
- Network-bound modules (individual UCI/Forman loader `_download_and_load` methods) are excluded from coverage via `[tool.coverage.run] omit` in `pyproject.toml` — don't chase coverage there; focus tests on the deterministic logic (factories, base classes, math).
- Run the full suite with coverage before opening a PR:

```bash
pytest
```

---

## Documentation

Docs live in `docs/` and are built with `mkdocs` + `mkdocstrings` (see `mkdocs.yaml`).

```bash
pip install -e ".[docs]"
mkdocs serve   # http://127.0.0.1:8000
```

When adding a new quantifier, metric, or dataset loader:

1. Add a NumPy-style docstring with `References` if the method comes from a paper.
2. Register it in the relevant `docs/api/*.md` file using `mkdocstrings` `:::` syntax.
3. If it introduces a new user-facing workflow, add or extend a page under `docs/tutorials/`.
4. Update `docs/tutorials.md` and the `nav` section of `mkdocs.yaml` if a new page was created.

---

## Reporting Bugs

Before opening an issue, please:

1. Search existing issues to avoid duplicates.
2. Include a minimal reproducible example (dataset, quantifier config, expected vs. actual output).
3. Include your environment (`python --version`, `pip show quack scikit-learn numpy scipy`).

---

## Proposing New Features

For substantial changes (new quantifier families, breaking API changes, new modules like ensembles), please open an issue first describing:

- The motivation / use case.
- Relevant references (papers), if applicable.
- A sketch of the proposed API, consistent with `BaseQuantifier`'s `.fit`/`.predict` contract.

This avoids duplicated effort and lets maintainers weigh in on design before implementation.

---

## Releasing

Releases are cut by maintainers following GitFlow:

1. Branch `release/x.y.z` from `develop`.
2. Bump `version` in `pyproject.toml`, update the changelog.
3. Final QA / doc review on the release branch.
4. Merge `release/x.y.z` into `main`, tag `vx.y.z`, then merge back into `develop`.
5. `docs.yaml` workflow auto-deploys documentation on pushes to `main` touching `docs/**` or `mkdocs.yml`.

---

Thanks again for helping improve `quack`! 🦆
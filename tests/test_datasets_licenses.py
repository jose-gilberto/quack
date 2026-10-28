# tests/test_datasets_licenses.py
import re

import pytest

from quack.datasets import ACCESS, REDISTRIBUTION, DatasetLicense, TermsNotAcceptedError, get_dataset_licenses
from quack.datasets._hierarchical_datasets import HierarchicalLoaderFactory
from quack.datasets._licenses import check_terms
from quack.datasets._multilabel_datasets import MultilabelLoaderFactory
from quack.datasets._regression_datasets import RegressionLoaderFactory

ALL_LOADERS = [(key, loader) for factory in (MultilabelLoaderFactory, HierarchicalLoaderFactory,
                                             RegressionLoaderFactory)
               for key, loader in factory._registry.items()]


def make_license(**overrides):
  fields = dict(spdx="CC-BY-4.0", redistribution="allowed", source="https://example.org",
                checked="2026-10-27", citation="Someone, 2020.")
  fields.update(overrides)
  return DatasetLicense(**fields)


class TestDatasetLicense:
  def test_rejects_unknown_categories(self):
    with pytest.raises(ValueError, match="redistribution"):
      make_license(redistribution="maybe")
    with pytest.raises(ValueError, match="access"):
      make_license(access="ftp")

  @pytest.mark.parametrize("redistribution, access, needs", [
    ("allowed", "open", False),
    ("allowed-sharealike", "open", False),
    ("not-declared", "manual", False),
    ("allowed-noncommercial", "open", True),
    ("allowed-research-only", "open", True),
    ("restricted", "open", True),
    ("allowed", "login", True),
  ])
  def test_requires_acceptance(self, redistribution, access, needs):
    assert make_license(redistribution=redistribution, access=access).requires_acceptance is needs

  def test_allows_mirror(self):
    assert make_license().allows_mirror
    assert make_license(redistribution="allowed-noncommercial").allows_mirror
    assert not make_license(redistribution="not-declared").allows_mirror
    assert not make_license(redistribution="allowed-research-only").allows_mirror
    assert not make_license(mirror_blocked="permission granted to the original host only").allows_mirror

  def test_notice_lists_terms(self):
    text = make_license(conditions=("attribution",), notes="Be careful.").notice("toy")
    assert text.startswith("toy")
    assert "CC-BY-4.0" in text and "attribution" in text and "Be careful." in text

  def test_check_terms(self):
    check_terms(make_license(), accept_terms=False, dataset="toy")
    restricted = make_license(redistribution="restricted")
    with pytest.raises(TermsNotAcceptedError, match="accept_terms=True"):
      check_terms(restricted, accept_terms=False, dataset="toy")
    check_terms(restricted, accept_terms=True, dataset="toy")

  def test_terms_error_is_a_permission_error(self):
    assert issubclass(TermsNotAcceptedError, PermissionError)


class TestRegisteredLicenses:
  @pytest.mark.parametrize("key, loader", ALL_LOADERS, ids=[k for k, _ in ALL_LOADERS])
  def test_every_loader_declares_a_complete_license(self, key, loader):
    lic = loader.license
    assert isinstance(lic, DatasetLicense)
    assert lic.redistribution in REDISTRIBUTION and lic.access in ACCESS
    assert lic.source.startswith("http") and lic.citation
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", lic.checked)
    assert loader.dataset_name

  def test_undeclared_licenses_are_never_mirrored(self):
    for key, loader in ALL_LOADERS:
      if loader.license.redistribution in {"not-declared", "restricted", "allowed-research-only"}:
        assert not loader.license.allows_mirror, key

  def test_get_dataset_licenses_covers_every_key(self):
    table = get_dataset_licenses()
    assert len(table) == len(ALL_LOADERS)
    assert set(table["task"]) == {"multilabel", "hierarchical", "regression"}
    assert table.loc["rcv1", "requires_acceptance"]
    assert not table.loc["told-br", "requires_acceptance"]

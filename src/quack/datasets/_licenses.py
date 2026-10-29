from dataclasses import dataclass, field


REDISTRIBUTION: dict[str, str] = {
  "allowed": "Open license that permits redistribution.",
  "allowed-sharealike": "Redistribution permitted; derivatives must keep the same license.",
  "allowed-noncommercial": "Redistribution permitted for non-commercial use only.",
  "allowed-research-only": "Use and distribution permitted for research purposes only.",
  "not-declared": "No license found; download from the original source only.",
  "restricted": "Terms restrict redistribution; download from the original source only.",
}
"""What each `DatasetLicense.redistribution` value means."""

ACCESS: dict[str, str] = {
  "open": "Downloaded automatically from the original source.",
  "manual": "The user downloads the files and places them in the dataset folder.",
  "login": "The source requires an account and accepting its terms on the website.",
}
"""What each `DatasetLicense.access` value means."""

_MIRRORABLE = {"allowed", "allowed-sharealike", "allowed-noncommercial"}
_NEEDS_ACCEPTANCE = {"allowed-noncommercial", "allowed-research-only", "restricted"}


class TermsNotAcceptedError(PermissionError):
  """Raised when a dataset with use conditions is loaded without `accept_terms=True`."""


@dataclass(frozen=True)
class DatasetLicense:
  """License terms of a dataset, as read from its official page.

  Every complex-scenario loader declares one as the `license` class
  attribute. The fields record what the license allows, where it was read
  and when, so that a published copy (e.g. on Zenodo, as done for the
  Forman datasets) is only made when the license permits it.

  Parameters
  ----------
  spdx : str
    SPDX identifier (`"CC-BY-4.0"`), `"LicenseRef-..."` for custom terms, or
    `"NOASSERTION"` when the source declares no license.
  redistribution : str
    One of the keys of `quack.datasets.REDISTRIBUTION`.
  source : str
    URL of the page where the license was read.
  checked : str
    ISO date of that reading.
  citation : str
    Reference to cite when using the dataset.
  access : str, default = "open"
    One of the keys of `quack.datasets.ACCESS`.
  conditions : tuple of str, default = ()
    Short tags for use conditions (`"attribution"`, `"share-alike"`,
    `"research-only"`, ...).
  mirror_blocked : str, default = ""
    Reason a copy cannot be published even though the license would allow
    it (e.g. a permission granted only to the original distributor).
  notes : str, default = ""
    Anything a user should know before using the data.
  """
  spdx: str
  redistribution: str
  source: str
  checked: str
  citation: str
  access: str = "open"
  conditions: tuple[str, ...] = field(default_factory=tuple)
  mirror_blocked: str = ""
  notes: str = ""

  def __post_init__(self):
    if self.redistribution not in REDISTRIBUTION:
      raise ValueError(f"unknown redistribution {self.redistribution!r}; use one of {sorted(REDISTRIBUTION)}")
    if self.access not in ACCESS:
      raise ValueError(f"unknown access {self.access!r}; use one of {sorted(ACCESS)}")

  @property
  def requires_acceptance(self) -> bool:
    """True when loading requires `accept_terms=True`."""
    return self.redistribution in _NEEDS_ACCEPTANCE or self.access == "login"

  @property
  def allows_mirror(self) -> bool:
    """True when the license permits publishing a processed copy of the data."""
    return self.redistribution in _MIRRORABLE and not self.mirror_blocked

  def notice(self, dataset_name: str = "") -> str:
    """Human-readable summary of the terms, shown when acceptance is required."""
    lines = [
      f"{dataset_name}".strip(),
      f"  License: {self.spdx} ({REDISTRIBUTION[self.redistribution]})",
    ]
    if self.conditions:
      lines.append(f"  Conditions: {', '.join(self.conditions)}")
    lines.append(f"  Access: {ACCESS[self.access]}")
    lines.append(f"  Checked on {self.checked} at {self.source}")
    if self.notes:
      lines.append(f"  Notes: {self.notes}")
    lines.append(f"  Cite: {self.citation}")
    return "\n".join(line for line in lines if line)


def check_terms(license: DatasetLicense, accept_terms: bool, dataset: str) -> None:
  """Raise `TermsNotAcceptedError` if the dataset has conditions and they were not accepted.

  Parameters
  ----------
  license : DatasetLicense
    Terms of the dataset being loaded.
  accept_terms : bool
    Value passed by the user to the `load_*` function.
  dataset : str
    Dataset key, used in the error message.
  """
  if license.requires_acceptance and not accept_terms:
    raise TermsNotAcceptedError(
      f"{license.notice(dataset)}\n\n"
      f"'{dataset}' has use conditions. Read them above and pass accept_terms=True to load it."
    )

import hashlib
import os
import urllib
import urllib.request
from pathlib import Path
from tqdm import tqdm


def file_hash(filename: str, algorithm: str = "sha256") -> str:
  """Compute the hex digest of a local file.

  Parameters
  ----------
  filename : str
    Path of the file to hash.
  algorithm : str, default = "sha256"
    Any algorithm accepted by `hashlib.new` (e.g. `"sha256"`, `"md5"`).

  Returns
  -------
  digest : str
    Hexadecimal digest of the file contents.
  """
  h = hashlib.new(algorithm)
  with open(filename, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def check_file_hash(filename: str, checksum: str) -> bool:
  """Check a file against a checksum written as `"<algorithm>:<hexdigest>"`.

  Parameters
  ----------
  filename : str
    Path of the file to check.
  checksum : str
    Expected digest prefixed by its algorithm, e.g. `"sha256:9a59a4..."` or
    `"md5:2bf729..."`, the format used by the dataset loaders.

  Returns
  -------
  matches : bool
    True when the file exists and its digest equals the expected one.
  """
  algorithm, _, expected = checksum.partition(":")
  if not expected:
    raise ValueError(f"checksum must look like '<algorithm>:<hexdigest>', got {checksum!r}")
  if not os.path.exists(filename):
    return False
  return file_hash(filename, algorithm) == expected.lower()


def download_file(url: str, filename: str, exist_ok: bool = False, checksum: str | None = None) -> None:
  """Download a file from an specific URL and rename the local base filename.

  Args:
    url (str): URL that the file will be downloaded from.
    filename (str): filename of the local archive that will be downloaded.
    exist_ok (bool, optional): Flag to download the file even if it is already
      exists on local filesystem. Defaults to False.
    checksum (str, optional): Expected digest as `"<algorithm>:<hexdigest>"`.
      A cached file that does not match is downloaded again; a fresh download
      that does not match is deleted and raises `ValueError`. Defaults to None
      (no verification).
  """
  def _progress_bar(block_num, bs, size):
    if pbar["tqdm"] is None:
      pbar["tqdm"] = tqdm(
        total=size if size > 0 else 0,
        unit="B", unit_scale=True, unit_divisor=1024,
        desc=os.path.basename(filename)
      )

    downloaded = block_num * bs

    if pbar["tqdm"].total and downloaded > pbar["tqdm"].total:
      downloaded = pbar["tqdm"].total

    pbar["tqdm"].n = downloaded
    pbar["tqdm"].refresh()

  if os.path.exists(filename) and exist_ok is False:
    if checksum is None or check_file_hash(filename, checksum):
      return
    print('Cached %s does not match its checksum, downloading again' % filename)

  os.makedirs(os.path.dirname(filename), exist_ok=True)
  pbar = {"tqdm": None}

  print('Downloading from %s' % url)

  try:
    urllib.request.urlretrieve(url, filename=filename, reporthook=_progress_bar)
  finally:
    if pbar["tqdm"] is not None:
      pbar["tqdm"].close()

  print('')

  if checksum is not None and not check_file_hash(filename, checksum):
    actual = file_hash(filename, checksum.partition(":")[0])
    os.remove(filename)
    raise ValueError(
      f"{os.path.basename(filename)} downloaded from {url} does not match its checksum "
      f"(expected {checksum}, got {actual}). The source may have changed."
    )


def get_quack_home() -> Path:
  """
  Get the home directory of QuacK, this is the folder where QuacK saves the
  datasets and metadata, such as the downloaded datasets from online repositories.

  The directory is `~/.quack_data/`

  Returns:
    home (str): base directory of quack data is this machine.
  """
  home = Path.home() / '.quack_data'
  os.makedirs(home, exist_ok=True)
  return home

import os
import urllib
import urllib.request
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, filename: str, exist_ok: bool = False) -> None:
    """Download a file from an specific URL and rename the local base filename.

    Args:
        url (str): URL that the file will be downloaded from. 
        filename (str): filename of the local archive that will be downloaded.
        exist_ok (bool, optional): Flag to download the file even if it is already
            exists on local filesystem. Defaults to False.
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
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pbar = {"tqdm": None}

    print('Downloading from %s' % url)

    try:
        urllib.request.urlretrieve(url, filename=filename, reporthook=_progress_bar)
    finally:
        if pbar["tqdm"] is not None:
            pbar["tqdm"].close()
    
    print('')


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
